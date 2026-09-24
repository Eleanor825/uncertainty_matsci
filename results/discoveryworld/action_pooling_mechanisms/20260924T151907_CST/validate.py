from pathlib import Path
from collections import Counter
import hashlib,json,math
P=Path(__file__).resolve().parent
m=json.loads((P/'manifest.json').read_text())
for n,pin in m['files'].items():
 b=(P/n).read_bytes();assert len(b)==pin['bytes']and hashlib.sha256(b).hexdigest()==pin['sha256']
s=json.loads((P/'snapshot.json').read_text());rows=s['completion_metrics_for_crosscheck'];assert len(rows)==len({r['job_id']for r in rows})==120
assert {(r['budget'],r['world'],r['policy_seed'],r['arm'])for r in rows}=={(b,w,k,a)for b in(10,20,30)for w in(3,4)for k in range(501,511)for a in('Native','ActionPool2')}
for b in(10,20,30):
 for arm in('Native','ActionPool2'):
  rr=[r for r in rows if r['budget']==b and r['arm']==arm];g=s['groups'][str(b)+'/'+arm];assert g['episodes']==len(rr)==20 and g['steps']==sum(r['agent_attempts']for r in rr)==20*b
  assert g['execution_successes']==sum(round(r['agent_attempts']*r['execution_success_rate'])for r in rr)
  assert 0<=g['positive_progress_steps']<=g['steps']and 0<=g['success_without_positive_progress']<=g['execution_successes']and g['adapter_rejected']==0
  assert g['fallback:None']==g['steps']
  if arm=='ActionPool2':assert g['two_valid_same_packet']+g['two_distinct_valid_packets']==g['steps']and g['supported_candidates']==2*g['steps']and 0<=g['actual_packet_changes']<=g['index_changed']<=g['steps']
  else:assert g['actual_packet_changes']==g['index_changed']==0
assert s['groups']['30/ActionPool2']['actual_packet_changes']==138 and s['groups']['30/ActionPool2']['success_without_positive_progress']==364
assert s['same120_completion_hashes_as_published_final_results']and not s['causal_value_of_unexecuted_candidate_observed']and s['new_scientific_calls']==0
print(json.dumps({'status':'passed','trajectories':120,'groups':6,'completion_reference_set_and_aggregate_consistency_checked':True,'new_scientific_calls':0},sort_keys=True))
