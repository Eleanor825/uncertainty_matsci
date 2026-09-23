from pathlib import Path
import hashlib,json,re
p=Path(__file__).resolve().parent
d=json.loads((p/'replay_evidence.json').read_bytes());m=json.loads((p/'manifest.json').read_bytes())
for name,r in m['files'].items():
 b=(p/name).read_bytes();assert len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
gates=('public','global_rng','world_rng','object_rng','UUID_rng')
assert d['scope']['world_seed']==2 and d['scope']['fixed_executed_actions']==13 and d['scope']['public_B30_budget_preserved']
assert d['total_recorded_diagnostic_calls']=={'GPU_calls':0,'LLM_calls':0,'NN_calls':0,'action_returns':78,'agent_requests':78,'bootstrap_ticks':6,'environment_loads':6,'tick_returns':78}
for mode in d['modes']:
 assert len(mode['replicas'])==3 and all(r['closed']for r in mode['replicas'])
 for pair in mode['pairwise']:
  assert [r['step']for r in pair['rows']]==list(range(14))
  for k in (*gates,'numpy_rng'):assert pair['all_'+k+'_equal']==all(r[k+'_equal']for r in pair['rows'])
  assert all(r['left_object_count']==r['right_object_count']==1188 for r in pair['rows'])
  if mode['mode']=='original':assert pair['first_public_divergence_step']==13 and not pair['all_public_equal']and not pair['all_global_rng_equal']and not pair['all_object_rng_equal']and pair['all_world_rng_equal']and pair['all_UUID_rng_equal']
  else:assert all(pair['all_'+k+'_equal']for k in gates)and pair['first_public_divergence_step']is None
  assert not pair['all_numpy_rng_equal']
a=d['instrument_limitations'];assert not a['NumPy_equality_is_gate']and a['NPC_position_arrays_empty']and a['NPC_position_equality_is_vacuous_and_excluded']and not a['complete_private_or_future_equivalence_proven']
w=d['world4_status_snapshot'];assert len(w['registered_jobs'])==10 and {r['policy_seed']for r in w['registered_jobs']}=={336,337}and w['reported_completed_outcomes']==0 and w['scientific_execution_or_outcomes_not_established_by_this_receipt']
assert not d['interpretation']['causal_controller_benefit_established']and d['interpretation']['publication_new_scientific_calls']==0
for name in m['files']:
 b=(p/name).read_bytes();assert not any((b'/'+x+b'/')in b for x in (b'mnt',b'Users',b'root'));assert not any(x in b for x in (b'Bearer'+b' ',b'PRIVATE'+b' KEY'))
print(json.dumps({'portable_validation_passed':True,'closed_replicas':6,'step_comparisons':56,'replay_agent_requests':78,'publication_scientific_calls':0,'world4_completed_outcomes_reported':0},sort_keys=True))
