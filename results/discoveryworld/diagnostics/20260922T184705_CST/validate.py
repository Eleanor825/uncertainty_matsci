"""Recompute published closed-action risk and behavior arithmetic; stdlib only."""
from pathlib import Path
import hashlib,json,math
H=Path(__file__).resolve().parent
def read(n):return json.loads((H/n).read_text())
def near(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
def main():
    manifest=read('manifest.json')
    for f in manifest['files']:
        b=(H/f['path']).read_bytes();assert len(b)==f['bytes']and hashlib.sha256(b).hexdigest()==f['sha256']
    a=read('audit.json');assert a['schema']=='dw_closed_development_cached_risk_audit_v1'
    assert all(a[k]==0 for k in ('LLM_calls','NN_forward_calls','environment_calls','graph_calls','optimizer_updates','filesystem_mutations'))
    assert a['partial_episodes_read']is False and a['threshold_search']is False and a['threshold_examined']==.5
    assert {e['condition']for e in a['episodes']}=={'Q75','Common2'};total=0
    for e in a['episodes']:
        rows=e['rows'];assert len(rows)==10 and [r['attempt_index']for r in rows]==list(range(1,11))
        assert e['world_seed']==2 and e['budget']==10 and e['final_task_score']==0 and not e['test_data_used']
        for r in rows:
            assert r['unexecuted_candidate_outcome_labels_created']is False and r['none_serialization_repair']is False
            assert r['risk_mask_reason']is None and r['label_mask_reason']is None
            assert r['primary_failure_label']==r['operational_returned_failure']==int(not r['official_success'])
            assert 0<=r['risk']<=1 and r['risk_alarm_05']==(r['risk']>=.5)
            assert near(r['immediate_score_delta'],r['score_after']-r['score_before'])
        m=e['primary_label_metrics'];n=len(rows);y=[r['primary_failure_label']for r in rows];p=[r['risk']for r in rows]
        assert m['n']==10 and m['masked']==0 and m['failures']==sum(y)
        assert near(m['Brier'],sum((x-z)**2 for x,z in zip(p,y))/n)
        clipped=[min(1-1e-15,max(1e-15,x))for x in p]
        assert near(m['NLL'],-sum(z*math.log(x)+(1-z)*math.log1p(-x)for x,z in zip(clipped,y))/n)
        confusion={'TP':0,'FP':0,'TN':0,'FN':0}
        for x,z in zip(p,y):confusion[('T'if (x>=.5)==bool(z)else'F')+('P'if x>=.5 else'N')]+=1
        assert all(m[k]==v for k,v in confusion.items())
        assert e['operational_return_metrics_including_serialization_repairs']==m
        counts=e['behavior_counts']
        for name in ('both_candidates_parsed','second_action_differs_from_first','selected_nonfirst','executed_action_differs_from_first',
                     'recorded_neural_rank_change','neural_revision_requested','repeats_previous_failed_action','successful_action_without_score_increase'):
            assert counts[name]==sum(bool(r[name])for r in rows)
        run=longest=0;prev=None
        for r in rows:
            repeated=bool(prev and not r['official_success']and not prev['official_success']and r['action_digest']==prev['action_digest'])
            assert repeated==r['repeats_previous_failed_action'];run=(run+1 if repeated else 1)if not r['official_success']else 0;longest=max(longest,run);prev=r
        assert counts['longest_identical_failed_action_run']==longest
        assert counts['second_candidates']==sum(r['candidate_count']==2 for r in rows)
        assert counts['successful_actions']==sum(r['official_success']for r in rows)
        assert counts['successful_action_without_score_increase']==sum(r['official_success']and r['immediate_score_delta']<=0 for r in rows)
        total+=n
    q=next(e for e in a['episodes']if e['condition']=='Q75');c=next(e for e in a['episodes']if e['condition']=='Common2')
    assert q['policy_seed']==304 and c['policy_seed']==305
    assert q['behavior_counts']['second_candidates']==3 and q['behavior_counts']['second_action_differs_from_first']==0
    assert q['behavior_counts']['recorded_neural_rank_change']==2 and q['behavior_counts']['executed_action_differs_from_first']==0
    assert c['behavior_counts']['second_action_differs_from_first']==6 and c['behavior_counts']['successful_action_without_score_increase']==9
    for path in H.iterdir():
        if path.suffix not in ('.json','.md','.csv'):continue
        text=path.read_text();assert not any(x in text for x in ('/mnt/','/Users/','de-41039-','tj-3041039-','"host"','"hostname"','"pid"','"password"'))
    print(json.dumps({'status':'PASS','selected_action_rows_recomputed':total,'closed_development_episodes_reused':2,'additional_trajectories':0,
        'Q75_effective_action_changes':0,'Common2_successes_without_score_increase':9,'new_scientific_calls':0,'validated_files':len(manifest['files'])}))
if __name__=='__main__':main()
