"""Synthetic filesystem contracts only: no model/ODE calls or real acceptance."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import pytest

from acceptance import AcceptanceError, _f32, _prefix_hash, accept_study, expected_episodes
from study import BOUNDS, digest, episode_summary, fitness, hypervolume, lhs, observed, prior_summary


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_complete(root):
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text())
    p=root/'sources/protocol.json';dump(p,protocol)
    code=root/'sources/upstream/snar.py';code.parent.mkdir(parents=True);code.write_text('# synthetic fixture only\n')
    lock={'source_modified':False,'official_class':'summit.benchmarks.snar.SnarBenchmark',
          'files':[{'path':'snar.py','sha256':sha(code)}]}
    lp=root/'sources/source_lock.json';dump(lp,lock)
    dump(root/'registration.json',{'protocol_fingerprint':digest(protocol),'oracle_source_root':str(code.parent),
        'sources':[{'path':str(x),'sha256':sha(x)} for x in (p,lp)]})
    source={'repository':protocol['benchmark']['repository'],'commit':protocol['benchmark']['commit'],
        'source_lock_sha256':sha(lp),'official_source_modified':False,'official_class':lock['official_class'],
        'executed_modules':[{'module':'summit.benchmarks.snar','path':'snar.py','sha256':sha(code)}]}
    base='a'*64;checkpoint='c'*64;cfg='d'*64;runtime={'synthetic_only':True}
    dump(root/'base_policy.json',{'base_state_hash':base,'checkpoint_hash':checkpoint,
        'configuration_fingerprint':cfg,'runtime':runtime})
    specs=expected_episodes(protocol);stages=defaultdict(list);summaries={};query_rows={};ids_saved={}
    states={b:[base,digest([b,1]),digest([b,2])] for b in ('full','es_only')}
    def fake_graph(directory,weight,ids):
        path=directory/'native_graph.npz'
        with zipfile.ZipFile(path,'w') as z:
            for name in ('sources.npy','targets.npy','weights.npy','node_types.npy'):z.writestr(name,b'synthetic-array-fixture')
        prefix=_prefix_hash(ids,True);stamp='unit-live-state'
        return {'weight_hash':weight,'source_stamp':{'state_id':stamp},
            'capture':{'prefix_hash':prefix,'future_outcomes_used':False},
            'graph':{'future_outcomes_used':False,'target':{'benchmark':'summit_snar','source_prefix_hash':prefix},
                'native':{'contract':'native_local_jacobian_action_logprob_mlp_cut_v2',
                    'source_selection_rule':protocol['graph']['source_selection_rule'],'sink_count':1,'sink_influence_seed_is_probability':False,
                    'backend_validation':{'passed':True,'checks':[{'passed':True}]*4,'policy_state_id':stamp,
                        'epsilon':.001,'rtol':.05,'atol':.0001},
                    'fidelity':{str(i):{'fvu_undefined':0,'output_fvu':.1} for i in range(32)}}},
            'node_records':[{'kind':'score','synthetic_only':True}],'graph_file':str(path),'graph_sha256':sha(path)}
    def episode(spec,prior,weight):
        name=spec['name'];folder=root/'episodes'/name;folder.mkdir(parents=True)
        header={k:spec[k] for k in ('name','arm','seed','budget')}
        header.update(prior_fingerprint=digest(prior),protocol_fingerprint=digest(protocol),weight_hash=weight)
        dump(folder/'episode.json',header);rows=[];design=lhs(spec['seed'],5)
        guided=spec['arm'] in ('uq_esopt','uq_only')
        for i in range(spec['budget']):
            params=design[i] if i<5 else {'tau':1.,'equiv_pldn':2.,'conc_dfnb':.3,'temperature':60.}
            generation_path=None;features_path=None;probability=.2 if guided and i>=5 else None
            if i>=5 and spec['arm'] not in ('random','gp_ei_scalarized'):
                cp=folder/f'query{i:03d}_proposal0';cp.mkdir()
                text=json.dumps(params,separators=(',',':'));ids=[[3]+[ord(x)+10 for x in text]+[1]]
                gen={'success':True,'parsed_action':params,'input_ids_with_completion':ids,
                    'prompt_token_count':1,'completion_count':len(ids[0])-1,'prefix_hash':_prefix_hash(ids),
                    'model_stamp':{'checkpoint_hash':checkpoint},'configuration_fingerprint':cfg,'policy_runtime':runtime}
                payload={'query_count':i,'budget':spec['budget'],'remaining_budget':spec['budget']-i,
                    'history':[observed(r) for r in rows][-6:],'history_omitted_count':max(0,i-6),
                    'shared_prior':prior_summary(prior,4)}
                generation_path=str(cp/'generation.json');features_path=str(cp/'candidate.json')
                dump(Path(generation_path),{'generation':gen,'weight_hash':weight,'prompt':{'prompt_payload':payload}})
                candidate={'success':True,'parameters':params,'generation_path':generation_path,'risk':probability}
                if probability is not None:candidate.update(features={'graph.unit':1.},graph=fake_graph(cp,weight,ids))
                dump(Path(features_path),candidate);ids_saved[generation_path]=ids
            qid=f'{name}/q{i:03d}';scope=spec['scope'];attempt=len(stages[scope])+1
            # Deliberately alternating improvements/non-improvements for label tests.
            bias=spec.get('generation',0)*20+spec.get('mutation',0)%2
            objectives={'sty':100.+5*(i//2)+bias,'e_factor':900.}
            request={'query_id':qid,'parameters':params,'seed':0,'noise_level':0.,'source_lock_sha256':sha(lp)}
            rsha=digest(request);aid=scope+':'+str(attempt)
            result={'query_id':qid,'parameters':params,'objectives':objectives,'oracle_attempt_id':aid,
                'oracle_attempts':attempt,'elapsed_seconds':.001,'noise_level':0.,'seed':0,
                'cache_hit':False,'physical_oracle_calls_this_request':1,'request_sha256':rsha}
            stages[scope].append((request,rsha,aid,result))
            row={'query_id':qid,'parameters':params,'objectives':objectives,
                'kind':'paired_lhs_initial_design' if i<5 else 'llm_proposal' if generation_path else spec['arm'],
                'risk':probability,'proposal_count':int(generation_path is not None),'invalid_proposals':0,
                'chosen_generation':generation_path,'chosen_features_path':features_path,'oracle_receipt':result,'elapsed_seconds':.01}
            before=hypervolume([*prior,*rows]);after=hypervolume([*prior,*rows,row])
            row['hv_increment']=after-before;row['no_hvi']=int(after-before<=1e-12)
            dump(folder/f'query{i:03d}_intent.json',{'query_id':qid,'parameters':params,'chosen_generation':generation_path,'prior_and_history_hv':before})
            dump(folder/f'query{i:03d}.json',row);rows.append(row);query_rows[qid]=row
        summary={**header,**episode_summary(rows,prior)}
        if guided:summary['valid_proposal_graph_fraction']=1.
        dump(folder/'summary.json',summary);summaries[name]=summary
        return rows
    archive=[]
    for name,spec in sorted(specs.items()):
        if spec.get('role')=='test':continue
        weight=base if name.startswith('collection_') else states[spec['branch']][spec['generation']] if spec['role']=='dev' else digest(name)
        archive.extend(episode(spec,[],weight))
    dump(root/'shared_prior_archive.json',archive)
    dump(root/'adaptation_complete.json',{'queries':550,'archive_fingerprint':digest(archive)})
    selected={}
    for branch in ('full','es_only'):
        records=[]
        for g in range(2):
            rewards=[]
            for mutation in protocol['evolution']['mutation_seeds'][g]:
                sm=summaries[f'es_{branch}_population_G{g}_m{mutation}']
                rewards.append(fitness(sm) if branch=='full' else sm['mean_querywise_hv']-.1*sm['invalid_proposal_rate'])
            v=[_f32(x) for x in rewards];avg=_f32(_f32(sum(v))/2);std=_f32(math.sqrt(sum((x-avg)**2 for x in v)/2))
            records.append({'generation':g,'state_hash_before':states[branch][g],'state_hash_after':states[branch][g+1],
                'seeds':protocol['evolution']['mutation_seeds'][g],'rewards':rewards,
                'normalized_rewards':[_f32(_f32(x-avg)/_f32(std+1e-8)) for x in v],
                'normalization_epsilon':1e-8,'ddof':0,'reward_normalization':'population_zscore',
                'alpha':.0005,'sigma':protocol['evolution']['sigma'][g],
                'metadata':{'branch':branch,'protocol_fingerprint':digest(protocol)},'parameter_delta_l2':{'unit.weight':.001}})
            history={'base_state_hash':base,'parameter_scope':'full','policy_model_id':protocol['model']['id']+'@'+protocol['model']['revision'],
                     'history':list(records),'current_state_hash':states[branch][g+1]}
            dump(root/f'{branch}_es_history_G{g+1}.json',history)
        dev=[]
        for g in range(3):
            sms=[summaries[f'es_{branch}_dev_G{g}_{s}'] for s in protocol['evolution']['checkpoint_dev_seeds']]
            scores=[fitness(s) if branch=='full' else s['mean_querywise_hv']-.1*s['invalid_proposal_rate'] for s in sms]
            d={'generation':g,'fitness':sum(scores)/len(scores),'scores':scores,'weight_hash':states[branch][g]}
            dump(root/f'{branch}_es_development_G{g}.json',d);dev.append(d)
        selected[branch]=max(dev,key=lambda x:(x['fitness'],-x['generation']))
        dump(root/f'{branch}_checkpoint_selection.json',{'test_used':False,'development':dev,'selected':selected[branch]})
        dump(root/f'{branch}_selected_es_history.json',{**history,'history':records[:selected[branch]['generation']],'current_state_hash':selected[branch]['weight_hash']})
        dump(root/f'{branch}_selected_clean_replay.json',{'verified':True,'weight_hash':selected[branch]['weight_hash'],'generation':selected[branch]['generation']})
        dump(root/f'{branch}_evolution_complete.json',{'queries':200,'selected':selected[branch]})
    for name,spec in sorted(specs.items()):
        if spec.get('role')!='test':continue
        arm=spec['arm'];weight=selected['full']['weight_hash'] if arm=='uq_esopt' else selected['es_only']['weight_hash'] if arm=='es_only' else None if arm in ('random','gp_ei_scalarized') else base
        episode(spec,archive,weight)
    stages['fit_risk_default']=[]
    for scope,items in stages.items():
        folder=root/'oracle_journals'/scope
        identity={'schema':'summit_snar_oracle_session_v1','classification':'benchmark','scientific_main_result':True,
            'noise_level':0.,'seed':0,'protocol':{'path':str(p),'sha256':sha(p)},'source':source,'session_id':scope,
            'runtime':{'CUDA_VISIBLE_DEVICES':'','threads':{'OMP_NUM_THREADS':'2'}},
            'parameter_bounds':{k:list(v) for k,v in BOUNDS.items()},'max_queries':max(1,len(items))}
        dump(folder/'identity.json',identity);events=[]
        for request,rsha,aid,result in items:
            qid=request['query_id'];key=hashlib.sha256(qid.encode()).hexdigest()
            dump(folder/'queries'/f'{key}.intent.json',{**request,'request_sha256':rsha,'attempt_id':aid})
            dump(folder/'queries'/f'{key}.terminal.json',{'query_id':qid,'request_sha256':rsha,'status':'succeeded','result':result})
            events += [{'schema':'summit_snar_oracle_event_v1','session_id':scope,'event':'attempt_started','attempt_id':aid,'query_id':qid,'request_sha256':rsha},
                {'schema':'summit_snar_oracle_event_v1','session_id':scope,'event':'attempt_returned','attempt_id':aid,**result}]
        (folder/'queries').mkdir(exist_ok=True)
        events.append({'schema':'summit_snar_oracle_event_v1','session_id':scope,'event':'session_closed'})
        journal=folder/'oracle_attempts.jsonl';journal.write_text(''.join(json.dumps(x)+'\n' for x in events))
        dump(folder/'summary.json',{'schema':'summit_snar_oracle_summary_v1','status':'closed','classification':'benchmark',
            'scientific_main_result':True,'session_id':scope,'oracle_attempts':len(items),'completed_queries':len(items),'reserved_queries':len(items),
            'errors':0,'unknown_outcomes':0,'reserved_without_started_event':0,'dft_calls':0,'pending_query_ids':[],
            'identity_sha256':sha(folder/'identity.json'),'journal_sha256':sha(journal)})
    rows={}
    for split,seed in [('train',1101),('dev',2101)]:
        rows[split]=[]
        for i in (5,6):
            name=f'collection_{split}_{seed}';qp=root/'episodes'/name/f'query{i:03d}.json';row=query_rows[f'{name}/q{i:03d}']
            directory=Path(row['chosen_generation']).parent;gp=directory/'risk_graph.json';features={'graph.unit':1.}
            dump(gp,{'available':True,'features':features,'metadata':fake_graph(directory,base,ids_saved[row['chosen_generation']])})
            generation=json.loads(Path(row['chosen_generation']).read_text())['generation']
            rows[split].append({'episode':name,'prefix_hash':generation['prefix_hash'],'label':row['no_hvi'],
                'features':features,'query_path':str(qp),'graph_path':str(gp)})
    # Unit prefixes must be distinct between splits, just like the real dedup gate.
    for r in rows['dev']:
        qp=json.loads(Path(r['query_path']).read_text());gp=Path(qp['chosen_generation']);saved=json.loads(gp.read_text())
        saved['generation']['input_ids_with_completion'][0][0]=4;saved['generation']['prefix_hash']=_prefix_hash(saved['generation']['input_ids_with_completion'])
        dump(gp,saved);r['prefix_hash']=saved['generation']['prefix_hash']
        gpath=Path(r['graph_path']);gr=json.loads(gpath.read_text());gr['metadata']=fake_graph(gpath.parent,base,saved['generation']['input_ids_with_completion']);dump(gpath,gr)
    dump(root/'risk_fit.json',{'test_used':False,'provenance':{'test_used_for_fit':False},'rows':rows,'feature_names':['graph.unit']})
    (root/'snar_risk.pt').write_bytes(b'synthetic fixture risk checkpoint; not real weights')
    dump(root/'collection_complete.json',{'episodes':5,'queries':150})
    for arm in protocol['evaluation']['arms']:dump(root/f'test_{arm}_complete.json',{'episodes':5,'oracle_queries':250})
    return protocol


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    root=tmp_path_factory.mktemp('synthetic_snar_only');protocol=synthetic_complete(root)
    return root,protocol


def test_complete_synthetic_matrix_raw_queries_and_official_shaped_journals(complete):
    root,protocol=complete;report=accept_study(root,protocol)
    assert report['passed'] and report['episodes']==60 and report['unique_physical_attempts']==2300
    assert report['adaptation_oracle_calls']==550 and report['test_oracle_calls']==1750
    assert report['scientific_oracle_calls_performed_by_audit']==0


def test_zero_completion_is_valid_failed_proposal_but_never_successful_action(complete):
    root,protocol=complete;folder=root/'episodes/test_qwen_base_5101'
    q=folder/'query005.json';intent=folder/'query005_intent.json';summary=folder/'summary.json'
    first=folder/'query005_proposal0';second=folder/'query005_proposal1'
    paths=[q,intent,summary,first/'candidate.json',first/'generation.json']
    originals={p:p.read_bytes() for p in paths}
    try:
        second.mkdir()
        generation=json.loads(originals[first/'generation.json']);candidate=json.loads(originals[first/'candidate.json'])
        candidate['generation_path']=str(second/'generation.json')
        dump(second/'generation.json',generation);dump(second/'candidate.json',candidate)
        empty=json.loads(originals[first/'generation.json'])
        empty['generation'].update(success=False,parsed_action=None,input_ids_with_completion=[[3]],
            prompt_token_count=1,completion_count=0,prefix_hash=_prefix_hash([[3]]))
        dump(first/'generation.json',empty)
        dump(first/'candidate.json',{'success':False,'parameters':None,'generation_path':str(first/'generation.json'),'risk':None})
        row=json.loads(originals[q]);row.update(proposal_count=2,invalid_proposals=1,
            chosen_generation=str(second/'generation.json'),chosen_features_path=str(second/'candidate.json'))
        dump(q,row);it=json.loads(originals[intent]);it['chosen_generation']=row['chosen_generation'];dump(intent,it)
        rows=[json.loads(p.read_text()) for p in sorted(folder.glob('query[0-9][0-9][0-9].json'))]
        prior=json.loads((root/'shared_prior_archive.json').read_text())
        dump(summary,{**json.loads((folder/'episode.json').read_text()),**episode_summary(rows,prior)})
        assert accept_study(root,protocol)['passed']
        empty['generation']['success']=True;dump(first/'generation.json',empty)
        with pytest.raises(AcceptanceError,match='token boundary'):
            accept_study(root,protocol)
    finally:
        for p,data in originals.items():p.write_bytes(data)
        if second.exists():shutil.rmtree(second)


@pytest.mark.parametrize('mutation',['label','receipt','summary','pending','selected','source','prior','split','graph_hash'])
def test_reject_tampered_scope_outputs_and_unknowns(complete,mutation):
    root,protocol=complete
    paths={'label':root/'episodes/test_random_5101/query005.json',
        'receipt':root/'episodes/test_random_5101/query005.json',
        'summary':root/'episodes/test_random_5101/summary.json',
        'pending':root/'oracle_journals/evaluate_random/summary.json',
        'selected':root/'full_selected_clean_replay.json','source':root/'sources/upstream/snar.py',
        'prior':root/'shared_prior_archive.json','split':root/'risk_fit.json',
        'graph_hash':root/'episodes/test_uq_esopt_5101/query005_proposal0/candidate.json'}
    path=paths[mutation];original=path.read_bytes()
    try:
        if mutation=='source':path.write_bytes(original+b'# changed\n')
        else:
            value=json.loads(original)
            if mutation=='label':value['no_hvi']=1-value['no_hvi']
            elif mutation=='receipt':value['oracle_receipt']['objectives']['sty']+=1
            elif mutation=='summary':value['final_hv']+=1
            elif mutation=='pending':value['unknown_outcomes']=1
            elif mutation=='selected':value['verified']=False
            elif mutation=='prior':value.pop()
            elif mutation=='split':value['rows']['dev'][0]['prefix_hash']=value['rows']['train'][0]['prefix_hash']
            elif mutation=='graph_hash':value['graph']['graph_sha256']='0'*64
            dump(path,value)
        with pytest.raises(AcceptanceError):accept_study(root,protocol)
    finally:path.write_bytes(original)
