"""Recalculate the published mechanism tables using only cached scientific data.

No network, model, oracle, fitting or source imports. Recorded remote artifact
hashes are provenance; this checker does not claim to rehash unpublished files.
"""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,math,statistics

P=Path(__file__).resolve().parent
def read(name):return json.loads((P/name).read_text())
def rows(name):return list(csv.DictReader((P/name).open(newline='')))
def close(a,b):
    if a is None or b is None:assert a is b,(a,b)
    else:assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-14),(a,b)
def num(v):return None if v=='' else float(v)
def auc(pairs):
    pos=[p for p,y in pairs if y==1];neg=[p for p,y in pairs if y==0]
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg)) if pos and neg else None
def supported_valid(a):return sum(x['n'] for x in a['graph_by_generation_success'] if x['success']=='True' and x['status']=='succeeded')
def valid(a):return sum(x['n'] for x in a['graph_by_generation_success'] if x['success']=='True')

def main():
    data=read('trajectory_evidence.json');analysis=read('analysis.json');pro=read('source_provenance.json')
    assert data['source_snapshot_sha256']==pro['source_snapshot_sha256']==analysis['snapshot']['sha256']=='2d1c1ff98a9f107289f039048f9430652d27b67d662da44d698e73d582db92fb'
    tests=data['fixed_test_runs'];training=data['full_train_dev'];assert len(tests)==68 and len(training)==6
    assert len({(r['spec']['task_id'],r['spec']['arm']) for r in tests})==68
    by={(r['spec']['task_id'],r['spec']['arm']):r for r in tests};table=rows('fixed17_all68_mechanisms.csv')
    assert len(table)==68
    artifact_refs={r['path']:r['sha256'] for r in pro['scientific_artifact_references_recorded_by_original_extractor']}
    for row in tests+training:
        s,a=row['spec'],row['analysis'];steps=a['rpc']['steps']
        assert len(steps)==10 and len({x['rpc_id'] for x in steps})==10 and all(x['ok'] is True for x in steps)
        curve=[0]+[x['official_SUN'] for x in steps]
        audc=sum(x+y for x,y in zip(curve,curve[1:]))/100
        close(audc,s['AUDC'] if row in tests else row['fitness'])
        assert a['journal_candidate_ORB_started']==a['journal_candidate_ORB_returned']==10
        assert a['journal_MACE_started']==a['journal_MACE_returned']
        if a['surrogate_cost_key_present']:close(a['reported_surrogate_oracle_attempts'],a['journal_MACE_started'])
        else:assert a['reported_surrogate_oracle_attempts'] is None
        assert sum(g['proposals'] for g in a['groups'])==a['rows'] and len(a['groups'])==a['decision_groups']
        if row in tests:
            assert s['budget']==10 and s['seed']==2 and curve[-1]==s['SUN']
            for k in ('result','receipt'):assert artifact_refs[s[k]['path']]==s[k]['sha256']
        else:assert artifact_refs[s['job_record']['path']]==s['job_record']['sha256']
    for row in table:
        raw=by[row['task_id'],row['arm']];s,a=raw['spec'],raw['analysis']
        expect={'SUN':s['SUN'],'AUDC':s['AUDC'],'score_buffer_calls':a['rpc']['tool_counts'].get('score_buffer',0),
            'reported_MACE':a['reported_surrogate_oracle_attempts'],'journal_MACE':a['journal_MACE_started'],
            'generate_calls':a['rpc']['tool_counts'].get('generate_structures',0),'select_calls':a['rpc']['tool_counts'].get('select_for_evaluation',0),
            'select_errors':a['rpc']['tool_failure_counts'].get('select_for_evaluation',0),
            'explicit_hash_repeats':a['rpc']['repeated_successful_explicit_hash_selections'],
            'graph_supported':supported_valid(a),'valid_generated_proposals':valid(a),'all_proposals':a['rows'],
            'NN_extra_proposals':sum(v for k,v in a['actual_extra_proposals_by_first_trigger'].items() if k.startswith('NN_')),
            'schema_extra_proposals':a['actual_extra_proposals_by_first_trigger'].get('observed_local_schema',0),
            'risk_rank_nonfirst_valid':a['ranking'].get('selected_nonfirst_valid_by_risk_ranking',0)}
        for k,v in expect.items():close(num(row[k]),v)
        assert row['result_path']==s['result']['path'] and row['result_sha256']==s['result']['sha256']
    cases=read('fixed17_cases.json');assert len(cases['rows'])==17
    for row in cases['rows']:
        f=by[row['task_id'],'full_support_aware']['spec'];u=by[row['task_id'],'uq_only_support_aware']['spec']
        ds=f['SUN']-u['SUN'];da=f['AUDC']-u['AUDC'];close(ds,row['SUN_delta_full_minus_UQ']);close(da,row['AUDC_delta_full_minus_UQ'])
        assert row['group']==('positive' if ds>0 else 'negative' if ds<0 else 'tie')
        assert (ds>0)==(da>1e-12) and (ds<0)==(da < -1e-12)
    for arm,total in cases['totals'].items():
        rr=[r for r in tests if r['spec']['arm']==arm];assert len(rr)==17
        assert sum(r['spec']['SUN'] for r in rr)==total['SUN'];close(statistics.mean(r['spec']['AUDC'] for r in rr),total['mean_AUDC'])
        ag=analysis['test_arm_summary'][arm]
        sums={'proposals':sum(r['analysis']['rows'] for r in rr),'groups':sum(r['analysis']['decision_groups'] for r in rr),
              'candidate_ORB':170,'MACE_physical_started':sum(r['analysis']['journal_MACE_started'] for r in rr),
              'scalar_threshold_crossings':sum(r['analysis']['scalar_above_actual_threshold'] for r in rr),
              'duplicate_explicit_hash_selections':sum(r['analysis']['rpc']['repeated_successful_explicit_hash_selections'] for r in rr),
              'fixed_recovery_events':sum(len(r['analysis']['fixed_failure_recovery_events']) for r in rr),'trajectories':17}
        for k,v in sums.items():assert ag[k]==v,(arm,k,ag[k],v)
        for k,field in [('tools','tool_counts'),('tool_failures','tool_failure_counts')]:
            count=Counter()
            for r in rr:count.update(r['analysis']['rpc'][field])
            assert dict(count)==ag[k]
        for k,field in [('graph','graph_status'),('requested','controller_retry_requested_by_source_and_head'),('extra','actual_extra_proposals_by_first_trigger'),('rank','ranking')]:
            count=Counter()
            for r in rr:count.update(r['analysis'][field])
            assert dict(count)==ag[k],(arm,k)
    # Each scientific event appears once. Last executed select is preferred;
    # fallback remains explicit and unsupported predictions remain missing.
    rebuilt=[];train_summary={r['request_id']:r for r in analysis['training_dev']}
    assert len(train_summary)==6
    for tr in training:
        s,a=tr['spec'],tr['analysis'];req=tr['request'];assert req['request_id']==s['request_id']
        assert req['case']['split']==s['partition'] and s['task_id']==('Al-Au-Hf' if s['partition']=='train' else 'Al-Pd-Sm')
        assert tr['job_seed_fields']['budget']==10
        assert not any(v for k,v in a['controller_retry_requested'].items() if k.startswith('NN_'))
        assert not any(v for k,v in a['actual_extra_proposals_by_first_trigger'].items() if k.startswith('NN_'))
        assert a['scalar_above_actual_threshold']==a['ranking'].get('selected_nonfirst_valid_by_risk_ranking',0)==0
        ts=train_summary[s['request_id']]
        mapping={'AUDC':tr['fitness'],'proposals':a['rows'],'valid_generation':valid(a),'graph_supported':supported_valid(a),
                 'unknown_FVU':a['graph_failure_reason_counts'].get('FVU',0),'NN_retry_requests':0,
                 'actual_schema_extra':a['actual_extra_proposals_by_first_trigger'].get('observed_local_schema',0),
                 'NN_rank_nonfirst':0,'scalar_threshold_crossings':0,'MACE_physical':a['journal_MACE_started'],
                 'score_buffer_calls':a['rpc']['tool_counts'].get('score_buffer',0)}
        for k,v in mapping.items():close(ts[k],v)
        risk_rows=a['training_dev_risk_rows']
        for step in a['rpc']['steps']:
            rid=step['rpc_id'];contributing=[r for r in risk_rows if r['disposition']=='executed' and rid in (r['next_outcome_rpc_ids'] or {}).get('scientific_evaluation_failure',[])]
            assert contributing,(s['request_id'],rid)
            select=[r for r in contributing if r['tool']=='select_for_evaluation'];r=(select or contributing)[-1]
            supported=r['supported'] is True
            rebuilt.append({'request_id':s['request_id'],'partition':s['partition'],'generation':s['generation'],'candidate_index':s['candidate_index'],
                'event_rpc_id':rid,'label':r['future_failure'],'decision_id':r['id'],'tool':r['tool'],'had_select_row':bool(select),
                'supported':supported,'risk':r['risk'] if supported else None,
                'not_new_risk':r['type_risk'].get('not_new') if supported else None,
                'unstable_risk':r['type_risk'].get('unstable') if supported else None,'contributing_rows':len(contributing)})
    event_rows=rows('train_dev_event_deduplicated.csv');assert len(event_rows)==len(rebuilt)==60
    key=lambda r:(r['request_id'],int(r['event_rpc_id']))
    assert len({key(r) for r in rebuilt})==60
    for stored,actual in zip(sorted(event_rows,key=key),sorted(rebuilt,key=key)):
        for k,v in actual.items():
            if k in ('risk','not_new_risk','unstable_risk'):close(num(stored[k]),v)
            else:assert stored[k]==('' if v is None else str(v)),(k,stored[k],v)
    es=read('train_dev_event_deduplicated_summary.json');assert es['events']==60 and es['new_fits']==es['threshold_changes']==0
    train=[r for r in rebuilt if r['partition']=='train'];prior=statistics.mean(r['label'] for r in train)
    assert len(train)==40;close(prior,es['train_empirical_failure_prior'])
    for summary in es['summary']:
        rr=[r for r in rebuilt if r['request_id']==summary['request_id']];pairs=[(r['risk'],r['label']) for r in rr if r['supported'] and r['risk'] is not None]
        assert summary['events']==10 and summary['positive_events']==sum(r['label'] for r in rr) and summary['supported_events']==len(pairs)
        close(auc(pairs),summary['AUROC']);close(statistics.mean((p-y)**2 for p,y in pairs) if pairs else None,summary['Brier'])
        close(statistics.mean((prior-y)**2 for p,y in pairs) if pairs else None,summary['constant_train_prior_Brier_same_supported'])
        close(.25 if pairs else None,summary['constant_half_Brier_same_supported']);assert sum(p>=.6 for p,y in pairs)==summary['risk_above_0_6']==0
    esdata=data['full_ES'];assert esdata==analysis['ES_selection_and_updates']
    assert esdata['actual_dev_generations']==[1,2] and esdata['summary']['initial_dev_evaluated'] is None
    assert sorted((r['spec']['generation'],r['spec']['candidate_index']) for r in training if r['spec']['partition']=='train')==[(1,0),(1,1),(2,0),(2,1)]
    assert sorted(r['spec']['generation'] for r in training if r['spec']['partition']=='dev')==[1,2]
    update1,update2=esdata['updates'];receipt=esdata['training_receipt']
    assert update1['state_hash_before']==receipt['initial_actual_model_state_hash']
    assert update1['state_hash_after']==update2['state_hash_before'] and update2['state_hash_after']==receipt['selected_actual_model_state_hash']
    assert receipt['selected_generation']==2 and receipt['uncertainty_reward_weight']==0 and receipt['selected_matches_initial'] is False
    for g,u in enumerate((update1,update2),1):
        assert u['parameter_blocks']==u['nonzero_delta_blocks']==723 and u['delta_L2_combined']>0
        expected=[r['fitness'] for r in sorted(training,key=lambda r:r['spec']['candidate_index'] or 0) if r['spec']['generation']==g and r['spec']['partition']=='train']
        assert u['rewards']==expected
    # Hash verification is over these derived public files, not private weights.
    for name,expected in pro['byte_preserved_exports'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==expected
    manifest=P/'manifest.json';n=0
    if manifest.exists():
        for row in read('manifest.json')['files']:
            f=P/row['path'];assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['export_sha256'];n+=1
    return {'passed':True,'fixed_test_trajectories':68,'full_train_dev_trajectories':6,'candidate_ORB_calls_in_audited_trajectories':740,
            'deduplicated_train_dev_events':60,'dev_events_per_generation':10,'full_training_NN_retries_or_rank_changes':0,
            'recorded_nonzero_parameter_blocks_per_update':[723,723],'G0_dev_evaluated_at_snapshot':False,
            'cross_seed_variance_estimated':False,'manifest_files_checked':n,'new_scientific_calls':0}

if __name__=='__main__':print(json.dumps(main(),sort_keys=True))
