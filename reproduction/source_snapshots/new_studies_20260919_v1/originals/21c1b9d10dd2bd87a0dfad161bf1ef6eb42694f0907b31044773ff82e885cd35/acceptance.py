"""Read-only terminal acceptance. No torch, model loading, or oracle invocation."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import zipfile

from study import BOUNDS, digest, episode_summary, fitness, hypervolume, lhs, observed, prior_summary


class AcceptanceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AcceptanceError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _same(a, b):
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a)==set(b) and all(_same(a[k],b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def _f32(x): return struct.unpack('f',struct.pack('f',x))[0]


def _prefix_hash(ids,mask=False):
    require(isinstance(ids,list) and len(ids)==1 and ids[0] and len(ids[0])<=2048
            and all(type(x) is int and x>=0 for x in ids[0]),'Invalid complete original token sequence')
    states={'input_ids':ids[0]}
    if mask: states['attention_mask']=[1]*len(ids[0])
    h=hashlib.sha256()
    for key,values in sorted(states.items()):
        h.update(json.dumps([key,[1,len(values)],'torch.int64'],separators=(',',':')).encode()+b'\0')
        h.update(struct.pack('<'+'q'*len(values),*values))
    return h.hexdigest()


class Evidence:
    def __init__(self): self.files={}
    def raw(self,path):
        path=Path(path).resolve();before=path.stat()
        with path.open('rb') as f: data=f.read()
        after=path.stat()
        key=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        require(key(before)==key(after) and len(data)==before.st_size,'Artifact changed while read: '+str(path))
        ref={'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        require(str(path) not in self.files or self.files[str(path)]==ref,'Artifact changed during acceptance')
        self.files[str(path)]=ref
        return data
    def read(self,path):
        def pairs(items):
            out={}
            for k,v in items:
                require(k not in out,'Duplicate JSON key');out[k]=v
            return out
        def invalid(value): raise AcceptanceError('Nonfinite JSON constant')
        return json.loads(self.raw(path),object_pairs_hook=pairs,parse_constant=invalid)
    def journal(self,path):
        raw=self.raw(path)
        require(not raw or raw.endswith(b'\n'),'Partial JSONL journal')
        return [json.loads(line,parse_constant=lambda v:(_ for _ in ()).throw(AcceptanceError('Nonfinite event'))) for line in raw.splitlines()]


def expected_episodes(protocol):
    result={}
    def put(name,arm,seed,budget,scope,**extra):
        require(name not in result,'Duplicate registered episode')
        result[name]=dict(name=name,arm=arm,seed=seed,budget=budget,scope=scope,**extra)
    for split,key in [('train','train_seeds'),('dev','dev_seeds')]:
        for seed in protocol['risk'][key]:
            put(f'collection_{split}_{seed}','qwen_base',seed,30,'collect_default',split=split)
    evo=protocol['evolution']
    require(evo['generations']==2 and evo['population_size']==2,'This acceptance covers the registered G2/P2 study')
    for branch,arm in [('full','uq_esopt'),('es_only','es_only')]:
        for generation in range(3):
            for seed in evo['checkpoint_dev_seeds']:
                put(f'es_{branch}_dev_G{generation}_{seed}',arm,seed,20,'evolve_'+branch,branch=branch,generation=generation,role='dev')
        for generation,seeds in enumerate(evo['mutation_seeds']):
            require(len(seeds)==2 and len(set(seeds))==2,'Mutation seeds must be distinct P2')
            for mutation in seeds:
                put(f'es_{branch}_population_G{generation}_m{mutation}',arm,evo['fitness_episode_seeds'][generation],20,
                    'evolve_'+branch,branch=branch,generation=generation,mutation=mutation,role='population')
    for arm in protocol['evaluation']['arms']:
        for seed in protocol['evaluation']['seeds']:
            put(f'test_{arm}_{seed}',arm,seed,50,'evaluate_'+arm,role='test')
    require(len(result)==60 and sum(x['budget'] for x in result.values())==2300,'Expected60 episodes/2300 new oracle calls')
    require(set(protocol['evaluation']['arms'])=={'random','gp_ei_scalarized','qwen_base','uq_esopt','uq_only','es_only','random_controller'},'Changed seven-arm study')
    require(protocol['accounting']['planned_adaptation_oracle_calls']==550 and protocol['accounting']['planned_test_oracle_calls']==1750,'Changed planned counts')
    return result


def _oracle_evidence(root, protocol, expected, ev, source_lock, source_lock_sha):
    expected_scopes={x['scope'] for x in expected.values()}
    folders={p.name:p for p in (root/'oracle_journals').iterdir() if p.is_dir()}
    require(expected_scopes <= set(folders) <= expected_scopes|{'fit_risk_default'},'Unexpected/missing oracle stage journal')
    by_query={};counts={};source_identity=None
    for scope,folder in sorted(folders.items()):
        identity=ev.read(folder/'identity.json');summary=ev.read(folder/'summary.json')
        require(identity['schema']=='summit_snar_oracle_session_v1' and identity['classification']=='benchmark'
                and identity['scientific_main_result'] is True and identity['noise_level']==0 and identity['seed']==0,'Wrong official oracle identity')
        p=identity['protocol'];require(ev.read(p['path'])==protocol and ev.files[str(Path(p['path']).resolve())]['sha256']==p['sha256'],'Oracle protocol source differs')
        source=identity['source']
        require(source['repository']==protocol['benchmark']['repository'] and source['commit']==protocol['benchmark']['commit']
                and source['source_lock_sha256']==source_lock_sha and source['official_source_modified'] is False,'Oracle source identity differs')
        require(source['official_class']==source_lock['official_class'],'Wrong official oracle class')
        locked={r['path']:r['sha256'] for r in source_lock['files']}
        require(source['executed_modules'] and all(locked.get(r['path'])==r['sha256'] for r in source['executed_modules']),'Executed oracle source not in lock')
        if source_identity is None:source_identity=source
        require(source==source_identity,'Mixed official oracle source implementations')
        require(identity['runtime']['CUDA_VISIBLE_DEVICES']=='' and all(str(v)=='2' for v in identity['runtime']['threads'].values()),'Oracle was not the registered CPU runtime')
        require(identity['parameter_bounds']=={k:list(v) for k,v in BOUNDS.items()},'Oracle domain changed')
        events=ev.journal(folder/'oracle_attempts.jsonl');session=identity['session_id']
        require(all(e['schema']=='summit_snar_oracle_event_v1' and e['session_id']==session for e in events),'Mixed journal session')
        require(events and events[-1]['event']=='session_closed','Oracle journal not terminally closed')
        allowed={'attempt_started','attempt_returned','cache_hit','session_paused','session_closed'}
        require(all(e['event'] in allowed for e in events),'Failed/unknown oracle journal event')
        starts=[e for e in events if e['event']=='attempt_started'];returns=[e for e in events if e['event']=='attempt_returned']
        require(len({e['attempt_id'] for e in starts})==len(starts) and len({e['attempt_id'] for e in returns})==len(returns),'Duplicate physical attempt')
        started={e['attempt_id']:e for e in starts};returned={e['attempt_id']:e for e in returns}
        require(set(started)==set(returned),'Pending or orphan physical attempt')
        intents=sorted((folder/'queries').glob('*.intent.json'));terminals=sorted((folder/'queries').glob('*.terminal.json'))
        require(len(intents)==len(terminals)==len(starts),'Unmatched query intent/terminal/physical start')
        require(summary['schema']=='summit_snar_oracle_summary_v1' and summary['status']=='closed'
                and summary['classification']=='benchmark' and summary['scientific_main_result'] is True
                and summary['session_id']==session,'Bad oracle summary')
        for k in ('oracle_attempts','completed_queries','reserved_queries'):require(summary[k]==len(starts),'Oracle summary count mismatch')
        for k in ('errors','unknown_outcomes','reserved_without_started_event','dft_calls'):require(summary[k]==0,'Oracle failure/unknown or foreign cost')
        require(summary['pending_query_ids']==[] and summary['identity_sha256']==ev.files[str((folder/'identity.json').resolve())]['sha256']
                and summary['journal_sha256']==ev.files[str((folder/'oracle_attempts.jsonl').resolve())]['sha256'],'Oracle closure hashes differ')
        require(identity['max_queries']>=len(starts),'Oracle exceeded its cap')
        for file in intents:
            intent=ev.read(file);qid=intent['query_id'];key=hashlib.sha256(qid.encode()).hexdigest()
            require(file.name==key+'.intent.json' and qid not in by_query,'Duplicate/wrong query identity')
            terminal=ev.read(file.with_name(key+'.terminal.json'))
            request={k:v for k,v in intent.items() if k not in ('request_sha256','attempt_id')}
            require(digest(request)==intent['request_sha256'] and request['source_lock_sha256']==source_lock_sha
                    and request['seed']==0 and request['noise_level']==0,'Original request hash/source differs')
            require(terminal['status']=='succeeded' and terminal['query_id']==qid and terminal['request_sha256']==intent['request_sha256'],'Failed or mismatched terminal')
            result=terminal['result'];aid=intent['attempt_id']
            require(aid in started and result['oracle_attempt_id']==aid and result['query_id']==qid
                    and result['parameters']==intent['parameters'] and result['request_sha256']==intent['request_sha256']
                    and result['physical_oracle_calls_this_request']==1 and result['cache_hit'] is False,'Invalid physical result')
            require(started[aid]['query_id']==qid and started[aid]['request_sha256']==intent['request_sha256']
                    and all(returned[aid].get(k)==v for k,v in result.items()),'Journal and terminal differ')
            by_query[qid]={'scope':scope,'result':result,'terminal_sha256':ev.files[str(file.with_name(key+'.terminal.json').resolve())]['sha256']}
        counts[scope]=len(starts)
    require(counts.get('fit_risk_default',0)==0,'Risk fitting called the oracle')
    return by_query,counts


def _evolution(root,protocol,episodes,ev,base):
    out={};cfg=protocol['evolution']
    for branch in ('full','es_only'):
        histories=[ev.read(root/f'{branch}_es_history_G{g}.json') for g in (1,2)]
        full=histories[-1]
        require(full['base_state_hash']==base and full['parameter_scope']=='full' and full['policy_model_id']==protocol['model']['id']+'@'+protocol['model']['revision'],'ES did not start from declared original policy')
        require(len(full['history'])==2 and histories[0]['history']==full['history'][:1],'Incomplete/mixed ES histories')
        states=[base]
        for g,row in enumerate(full['history']):
            require(row['generation']==g and row['state_hash_before']==states[-1] and _sha(row['state_hash_after']),'ES hash chain differs')
            rewards=[]
            for mutation in cfg['mutation_seeds'][g]:
                s=episodes[f'es_{branch}_population_G{g}_m{mutation}']
                rewards.append(fitness(s) if branch=='full' else s['mean_querywise_hv']-.1*s['invalid_proposal_rate'])
            require(row['seeds']==cfg['mutation_seeds'][g] and _same(row['rewards'],rewards)
                    and row['alpha']==cfg['alpha'] and row['sigma']==cfg['sigma'][g],'ES used different population/fitness/config')
            values=[_f32(x) for x in rewards];avg=_f32(_f32(sum(values))/2)
            std=_f32(math.sqrt(sum((x-avg)**2 for x in values)/2))
            normalized=[_f32(_f32(x-avg)/_f32(std+row['normalization_epsilon'])) for x in values]
            require(len(row['normalized_rewards'])==2 and all(math.isclose(a,b,abs_tol=1e-6,rel_tol=1e-6) for a,b in zip(row['normalized_rewards'],normalized))
                    and row['ddof']==0 and row['reward_normalization']=='population_zscore','ES float32 normalization differs')
            require(row['metadata']['branch']==branch and row['metadata']['protocol_fingerprint']==digest(protocol),'Mixed ES branch/protocol')
            require(row['parameter_delta_l2'] and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in row['parameter_delta_l2'].values()),'Missing actual parameter update evidence')
            require((row['state_hash_after']!=row['state_hash_before'])==any(v>0 for v in row['parameter_delta_l2'].values()),'Null/update hash evidence inconsistent')
            states.append(row['state_hash_after'])
            require(histories[g]['current_state_hash']==states[-1],'History final hash differs')
        dev=[]
        for g in range(3):
            scores=[]
            for seed in cfg['checkpoint_dev_seeds']:
                s=episodes[f'es_{branch}_dev_G{g}_{seed}'];require(s['weight_hash']==states[g],'Development evaluated wrong generation')
                scores.append(fitness(s) if branch=='full' else s['mean_querywise_hv']-.1*s['invalid_proposal_rate'])
            record=ev.read(root/f'{branch}_es_development_G{g}.json')
            expected={'generation':g,'fitness':sum(scores)/len(scores),'scores':scores,'weight_hash':states[g]}
            require(_same(record,expected),'Development fitness record differs');dev.append(record)
        selected=max(dev,key=lambda x:(x['fitness'],-x['generation']))
        selection=ev.read(root/f'{branch}_checkpoint_selection.json')
        require(selection['test_used'] is False and _same(selection['development'],dev) and _same(selection['selected'],selected),'Test-informed or different checkpoint selection')
        selected_history=ev.read(root/f'{branch}_selected_es_history.json')
        require(selected_history['history']==full['history'][:selected['generation']] and selected_history['current_state_hash']==selected['weight_hash']
                and selected_history['base_state_hash']==base,'Selected replay history differs')
        proof=ev.read(root/f'{branch}_selected_clean_replay.json')
        require(proof.get('verified') is True and proof['weight_hash']==selected['weight_hash'] and proof['generation']==selected['generation'],'Missing/mismatched actual clean replay proof')
        complete=ev.read(root/f'{branch}_evolution_complete.json')
        require(complete['queries']==200 and _same(complete['selected'],selected),'ES branch completion differs')
        out[branch]=selected
    return out


def _graph(meta,features,ev,weight_hash,protocol,prefix_hash):
    require(meta['weight_hash']==weight_hash and meta['capture']['prefix_hash']==prefix_hash,'Graph weight/prefix differs')
    graph=meta['graph'];native=graph['native'];target=graph['target']
    require(graph['future_outcomes_used'] is False and meta['capture']['future_outcomes_used'] is False
            and target['benchmark']=='summit_snar' and target['source_prefix_hash']==prefix_hash,'Wrong/future-informed graph target')
    require(native['contract']=='native_local_jacobian_action_logprob_mlp_cut_v2'
            and native['source_selection_rule']==protocol['graph']['source_selection_rule']
            and native['sink_count']==1 and native['sink_influence_seed_is_probability'] is False,'Wrong native graph algorithm')
    gate=native['backend_validation']
    require(gate['passed'] is True and len(gate['checks'])==4 and all(x['passed'] is True for x in gate['checks'])
            and gate['policy_state_id']==meta['source_stamp']['state_id'],'Missing exact-state actual native gate')
    for k in ('epsilon','rtol','atol'):
        require(gate[k]==protocol['graph']['validation_'+k],'Changed FD gate threshold')
    fidelity=native['fidelity']
    require(len(fidelity)==32 and all(v['fvu_undefined']==0 and math.isfinite(v['output_fvu']) and v['output_fvu']<=.5 for v in fidelity.values()),'Current-prefix native fidelity failed')
    require(meta['node_records'] and all(k.startswith(('graph.','hidden.','sampling.')) for k in features),'Missing graph nodes or mixed label/features')
    path=Path(meta['graph_file']);ev.raw(path)
    require(meta.get('graph_sha256')==ev.files[str(path.resolve())]['sha256'],'Native graph artifact hash differs')
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None and {'sources.npy','targets.npy','weights.npy','node_types.npy'}<=set(z.namelist()),'Broken/missing native graph arrays')


def accept_study(output,protocol):
    """Raise on any missing/mixed/unknown evidence; return a source-bound audit."""
    root=Path(output).resolve();ev=Evidence();expected=expected_episodes(protocol)
    registration=ev.read(root/'registration.json')
    require(registration['protocol_fingerprint']==digest(protocol) and registration['sources'],'Unregistered protocol/source')
    source_lock=None;source_lock_sha=None
    for item in registration['sources']:
        data=ev.raw(item['path']);require(hashlib.sha256(data).hexdigest()==item['sha256'],'Registered source changed')
        if Path(item['path']).name=='source_lock.json':
            require(source_lock is None,'Ambiguous official source lock')
            source_lock=ev.read(item['path']);source_lock_sha=item['sha256']
    require(source_lock is not None and source_lock['source_modified'] is False,'Missing official source lock')
    if registration.get('oracle_source_root'):
        for item in source_lock['files']:
            p=Path(registration['oracle_source_root'])/item['path']
            require(hashlib.sha256(ev.raw(p)).hexdigest()==item['sha256'],'Pinned official source changed')
    physical,oracle_counts=_oracle_evidence(root,protocol,expected,ev,source_lock,source_lock_sha)
    folders={p.name:p for p in (root/'episodes').iterdir() if p.is_dir()}
    require(set(folders)==set(expected),'Missing/extra formal episode directory')
    prior=ev.read(root/'shared_prior_archive.json');adaptation=ev.read(root/'adaptation_complete.json')
    require(len(prior)==550 and adaptation['queries']==550 and adaptation['archive_fingerprint']==digest(prior),'Common adaptation prior differs')
    base_record=ev.read(root/'base_policy.json');base=base_record['base_state_hash'];require(_sha(base),'Invalid base weight hash')
    summaries={};all_rows={};adapt_rows=[]
    for name,spec in sorted(expected.items()):
        folder=folders[name];header=ev.read(folder/'episode.json');p=prior if spec.get('role')=='test' else []
        require(all(header[k]==spec[k] for k in ('name','arm','seed','budget')) and header['protocol_fingerprint']==digest(protocol)
                and header['prior_fingerprint']==digest(p),'Episode scope/prior identity differs')
        files=sorted(folder.glob('query[0-9][0-9][0-9].json'))
        require([x.name for x in files]==[f'query{i:03d}.json' for i in range(spec['budget'])],'Missing/extra query records')
        rows=[];design=lhs(spec['seed'],5);all_candidates=[]
        for i,file in enumerate(files):
            row=ev.read(file);qid=f'{name}/q{i:03d}';require(row['query_id']==qid and qid not in all_rows,'Wrong/duplicate query ID')
            require(qid in physical and physical[qid]['scope']==spec['scope'],'Query not in its official stage journal')
            original=physical[qid]['result'];receipt=row['oracle_receipt']
            require(all(receipt.get(k)==v for k,v in original.items() if k not in ('cache_hit','physical_oracle_calls_this_request'))
                    and (receipt['cache_hit'],receipt['physical_oracle_calls_this_request']) in ((False,1),(True,0)),'Query result does not match its immutable official terminal')
            require(row['parameters']==original['parameters'] and row['objectives']==original['objectives'],'Published objective/parameters differ from official result')
            require(set(row['parameters'])==set(BOUNDS) and all(type(row['parameters'][k]) in (int,float) and math.isfinite(row['parameters'][k]) and lo<=row['parameters'][k]<=hi for k,(lo,hi) in BOUNDS.items()),'Invalid executed parameter')
            if i<5:require(row['parameters']==design[i] and row['kind']=='paired_lhs_initial_design','Paired initialization differs')
            before=hypervolume([*p,*rows]);after=hypervolume([*p,*rows,row])
            require(_same(row['hv_increment'],after-before) and row['no_hvi']==int(after-before<=protocol['objective']['non_improvement_tolerance']),'Incorrect HV increment or observed risk label')
            risk=row.get('risk');require(risk is None or type(risk) in (int,float) and math.isfinite(risk) and 0<=risk<=1,'Invalid uncertainty value')
            require(type(row['proposal_count']) is int and 0<=row['invalid_proposals']<=row['proposal_count']<=2,'Proposal accounting differs')
            if spec['arm'] not in ('uq_esopt','uq_only'):require(risk is None,'Ablation/base silently used risk predictions')
            intent=ev.read(folder/f'query{i:03d}_intent.json')
            require(intent['query_id']==qid and intent['parameters']==row['parameters'] and _same(intent['prior_and_history_hv'],before)
                    and intent['chosen_generation']==row['chosen_generation'],'Published query intent differs')
            candidates=[]
            for directory in sorted(folder.glob(f'query{i:03d}_proposal*')):
                require(directory.name in {f'query{i:03d}_proposal0',f'query{i:03d}_proposal1'},'Unregistered extra proposal')
                candidate=ev.read(directory/'candidate.json');saved=ev.read(directory/'generation.json');generation=saved['generation']
                require(Path(candidate['generation_path']).resolve()==(directory/'generation.json').resolve()
                        and saved['weight_hash']==header['weight_hash'] and generation['model_stamp']['checkpoint_hash']==base_record['checkpoint_hash']
                        and generation['configuration_fingerprint']==base_record['configuration_fingerprint']
                        and generation['policy_runtime']==base_record['runtime'],'Candidate policy/configuration differs')
                require(_prefix_hash(generation['input_ids_with_completion'])==generation['prefix_hash'],'Stored generated token bytes differ')
                start=generation['prompt_token_count'];length=len(generation['input_ids_with_completion'][0])
                require(type(start) is int and 1<=start<=length and generation['completion_count']==length-start<=128
                        and (generation['success'] is False or start<length),'Original token boundary differs')
                payload=saved['prompt']['prompt_payload']
                require(payload['query_count']==i and payload['budget']==spec['budget'] and payload['remaining_budget']==spec['budget']-i
                        and payload['history']==[observed(r) for r in rows][-protocol['policy']['history_window']:]
                        and payload['history_omitted_count']==max(0,i-protocol['policy']['history_window'])
                        and payload['shared_prior']==prior_summary(p,4),'Candidate prompt includes wrong/future data')
                require(candidate['success']==generation['success'] and 'no_hvi' not in candidate,'Candidate observation/label leakage')
                if candidate['success']:require(candidate['parameters']==generation['parsed_action'],'Candidate differs from generated JSON action')
                if candidate['risk'] is not None:
                    require(candidate['success'] and spec['arm'] in ('uq_esopt','uq_only'),'Unexpected risk-controlled candidate')
                    _graph(candidate['graph'],candidate['features'],ev,header['weight_hash'],protocol,
                           _prefix_hash(generation['input_ids_with_completion'],True))
                candidates.append(candidate)
            require(len(candidates)==row['proposal_count'] and sum(not c['success'] for c in candidates)==row['invalid_proposals'],'Proposal count/invalid evidence differs')
            if row['chosen_generation'] is not None:
                chosen=[c for c in candidates if c['generation_path']==row['chosen_generation']]
                require(len(chosen)==1 and chosen[0]['success'] and chosen[0]['parameters']==row['parameters']
                        and chosen[0]['risk']==risk,'Chosen query not its actual proposed action/risk')
            else:require(risk is None,'Unproposed query has fabricated uncertainty')
            all_candidates.extend(candidates)
            rows.append(row);all_rows[qid]=row
        actual=ev.read(folder/'summary.json');computed={**header,**episode_summary(rows,p)}
        require(all(k in actual and _same(actual[k],v) for k,v in computed.items()),'Episode summary not reconstructed from its own queries')
        if spec['arm'] in ('uq_esopt','uq_only'):
            valid=[c for c in all_candidates if c['success']];fraction=sum(c['risk'] is not None for c in valid)/len(valid) if valid else 0.
            require(_same(actual.get('valid_proposal_graph_fraction'),fraction) and fraction>=protocol['risk']['minimum_valid_graph_fraction'],'Native graph coverage failed')
        summaries[name]=actual
        if spec.get('role')!='test':adapt_rows.extend(rows)
    require(set(all_rows)==set(physical) and len(physical)==2300,'Extra/omitted physical calls')
    require(prior==adapt_rows,'Shared prior is not the exact ordered complete adaptation archive')
    selections=_evolution(root,protocol,summaries,ev,base)
    for name,spec in expected.items():
        if spec.get('role')=='test':
            expected_hash=selections['full']['weight_hash'] if spec['arm']=='uq_esopt' else selections['es_only']['weight_hash'] if spec['arm']=='es_only' else None if spec['arm'] in ('random','gp_ei_scalarized') else base
            require(summaries[name]['weight_hash']==expected_hash,'Evaluation uses the wrong selected/base policy')
        elif name.startswith('collection_'):require(summaries[name]['weight_hash']==base,'Risk collection not original policy')
    risk=ev.read(root/'risk_fit.json');require(risk['test_used'] is False and risk['provenance']['test_used_for_fit'] is False,'Risk fit used test data')
    tr=risk['rows']['train'];dv=risk['rows']['dev'];require(tr and dv,'Empty risk split')
    require(not ({r['episode'] for r in tr}&{r['episode'] for r in dv}) and not ({r['prefix_hash'] for r in tr}&{r['prefix_hash'] for r in dv}),'Risk train/dev leakage')
    for split,rows in [('train',tr),('dev',dv)]:
        for row in rows:
            require(row['episode'] in expected and expected[row['episode']].get('split')==split,'Risk includes another scientific split')
            original=ev.read(row['query_path']);require(row['label']==original['no_hvi'],'Risk label differs from executed observation')
            require(set(row['features'])==set(risk['feature_names']) and all(k.startswith(('graph.','hidden.','sampling.')) for k in row['features']),'Outcome/evidence mixed into risk feature schema')
            graph=ev.read(row['graph_path']);require(graph['available'] is True and graph['features']==row['features'],'Risk graph differs')
            saved=ev.read(original['chosen_generation']);generation=saved['generation']
            require(saved['weight_hash']==base and generation['prefix_hash']==row['prefix_hash'],'Risk graph has different base action')
            _graph(graph['metadata'],graph['features'],ev,base,protocol,_prefix_hash(generation['input_ids_with_completion'],True))
    require({r['label'] for r in tr}=={0,1},'Risk fit lacks both observed training classes')
    for arm in protocol['evaluation']['arms']:
        receipt=ev.read(root/f'test_{arm}_complete.json');require(receipt=={'episodes':5,'oracle_queries':250},'Test stage completion differs')
    require(ev.read(root/'collection_complete.json')=={'episodes':5,'queries':150},'Collection completion differs')
    ev.raw(root/'snar_risk.pt')  # Small risk checkpoint only; no large policy/TC tensor reads.
    report={'schema':'summit_snar_terminal_acceptance_v1','passed':True,'complete':True,
            'protocol_fingerprint':digest(protocol),'episodes':60,'adaptation_episodes':25,'test_episodes':35,
            'adaptation_oracle_calls':550,'test_oracle_calls':1750,'unique_physical_attempts':2300,
            'failed_or_unknown_physical_attempts':0,'oracle_stage_counts':oracle_counts,'selected':selections,
            'risk_rows':{'train':len(tr),'dev':len(dv)},'scientific_oracle_calls_performed_by_audit':0,
            'model_calls_performed_by_audit':0,'large_policy_weights_rehashed':False,
            'weight_verification_scope':'Recorded original ES hash chains, dev-only selection and actual clean-replay proofs checked; this CPU audit does not replay tensors.',
            'evidence_files':sorted(ev.files.values(),key=lambda r:r['path'])}
    report['fingerprint']=digest(report)
    return report
