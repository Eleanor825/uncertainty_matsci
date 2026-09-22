"""Portable aggregate/status validation; no dependencies, raw-log replay or fits."""
from pathlib import Path
import hashlib,json,math
H=Path(__file__).resolve().parent
def read(name):return json.loads((H/name).read_text())
def main():
    manifest=read('manifest.json')
    for row in manifest['files']:
        b=(H/row['path']).read_bytes();assert len(b)==row['bytes']and hashlib.sha256(b).hexdigest()==row['sha256']
    s=read('status.json');assert s['completed_within_seed_pairs']==0 and s['cross_seed_effect_comparison']is False
    assert s['online_improvement_established']is False and s['ES_parameter_updates_added']==s['new_heldout_Full_tests']==0
    complete={x['condition']:x for x in s['complete_development_episodes']};assert set(complete)=={'Q75','Common2'}
    for name,nfail,seed in [('Q75',10,304),('Common2',1,305)]:
        x=complete[name];f=x['fitness'];assert x['accepted']is True and x['policy_seed']==seed and x['split']=='development'
        assert x['budget']==x['action_returns']==x['ticks_returned']==f['agent_attempts']==10
        assert x['failed_action_returns']==f['failure_count']==nfail and x['successful_action_returns']==10-nfail
        assert f['scoreNormalized']==0 and f['completedSuccessfully']is False and f['lambda_cal']==0
        assert math.isclose(f['failure_fraction'],nfail/10)and math.isclose(f['F'],-.1*nfail/10)and f['F']==f['J']
    assert complete['Q75']['supported_activity_gate_passed']is True
    partial={x['condition']:x for x in s['partial_development_episodes']};assert set(partial)=={'Posterior05','Internal2'}
    for name,n,nfail in [('Posterior05',7,7),('Internal2',1,0)]:
        x=partial[name];assert x['status']=='partial_not_final'and x['accepted']is None and x['fitness']is None
        assert x['action_returns']==x['ticks_returned']==n and x['failed_action_returns']==nfail
    assert all(x['passed_prefixes']==8 and x['failed_prefixes']==0 for x in s['qualification'])
    assert s['grounded_v6']['scientific_execution_confirmed']is False
    cv=read('cv_not_run.json');assert cv['completion']['status']=='not_run_insufficient_fold_support'and not cv['passed']
    assert all(x==0 for x in cv['completion']['counts'].values())and not cv['fold_omitted']and not cv['gate_relaxed']
    failed=[]
    for fold in cv['folds']:
        a=fold['training_support_original_gate'];passed=min(a['positive_rows'],a['negative_rows'])>=10 and min(a['positive_episodes'],a['negative_episodes'])>=2
        assert passed==a['passed']
        if not passed:failed.append(a)
    assert len(failed)==1 and failed[0]['positive_rows']==33 and failed[0]['negative_rows']==7
    p=read('training_prefix_feasibility.json');rule=p['selection_rule']
    assert rule['fixed_extra_indices_per_episode']==[2,3,4,5,7,8,9,10]and rule['label_selected']is False
    assert rule['maximum_new_training_rows']==24 and rule['maximum_total_training_rows']==84
    assert p['actual_extra_graphs_completed']==0 and p['actual_enriched_dataset_ready']is False and p['CV_v1_modified_or_retried']is False
    assert p['new_graphs']==p['NN_updates']==p['new_environment_calls']==p['dev_or_test_trajectory_reads']==0
    episodes={x['episode_id']:x for x in p['episodes']};assert len(episodes)==3
    for fold in p['folds']:
        other=[x for key,x in episodes.items()if key!=fold['heldout_episode']];a=fold['label_only_potential_if_all_fixed_extra_graphs_pass']
        assert sum(x['potential_fixed28_observed_label_counts']['positive']for x in other)==a['positive_rows']
        assert sum(x['potential_fixed28_observed_label_counts']['negative']for x in other)==a['negative_rows']
    # Files contain no operational host/process/credential material or machine paths.
    for path in H.iterdir():
        if path.suffix not in ('.json','.md','.csv'):continue
        text=path.read_text()
        assert not any(value in text for value in ('/mnt/','/Users/','GPU-ca6','de-41039-','tj-3041039-','"hostname"','"host"','"pid"','"password"','"token"'))
    print(json.dumps({'status':'PASS','complete_development_episodes':2,'incomplete_pairs':2,'CV_optimizer_updates':0,'extra_graphs_completed':0,'private_operational_fields_exported':False,'new_scientific_calls':0,'validated_files':len(manifest['files'])}))
if __name__=='__main__':main()
