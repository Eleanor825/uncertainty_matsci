"""Validate a fixed sanitized partial snapshot without network or scientific calls."""
from pathlib import Path
from collections import Counter
import hashlib, json, math

P=Path(__file__).resolve().parent
def read(name):return json.loads((P/name).read_text())
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
m=read('manifest.json')
assert set(m['files'])=={p.name for p in P.iterdir()if p.is_file()and p.name!='manifest.json'}
for name,pin in m['files'].items():
 raw=(P/name).read_bytes();assert len(raw)==pin['bytes']and hashlib.sha256(raw).hexdigest()==pin['sha256']
s=read('snapshot.json');rows=s['rows'];arms=('Native1','Common2','Public2','FullPool2','ActionPool2')
assert len(rows)==len({r['job_id']for r in rows})==s['expected_trajectories']==20
assert {(r['task'],r['budget'],r['arm'],r['seed'])for r in rows}=={(t,b,a,501)for t in('Mg-Sn-Sr','Au-K-Tb')for b in(10,20)for a in arms}
assert dict(Counter(r['status']for r in rows))=={'accepted':2,'pending':16,'started_incomplete':2}
assert not s['complete']and s['accepted']==2 and s['failed']==0
assert not s['unfinished_metrics_imputed']and not s['unexecuted_candidate_labels_imputed']
assert sum(r['budget']for r in rows)==s['registered_candidate_ORB_attempts']==300
accepted=[r for r in rows if r['status']=='accepted']
assert {r['arm']for r in accepted}=={'Native1','Common2'}
assert all(r['task']=='Mg-Sn-Sr'and r['budget']==10 for r in accepted)
assert sum(r['costs']['reconstructed_scientific_costs']['candidate_oracle_attempts']for r in accepted)==s['accepted_candidate_ORB_attempts']==20
for r in rows:
 if r['status']!='accepted':
  assert r['metrics']is None and r['costs']is None and r['completion']is None
  continue
 v=r['metrics'];curve=v['discovery_curve'];b=r['budget'];c=r['costs'];o=c['online_costs'];e=c['reconstructed_scientific_costs']
 assert len(curve)==b+1 and curve[0]==[0,0]and [x[0]for x in curve]==list(range(b+1))
 assert v['SUN']==curve[-1][1]and close(v['mSUN'],v['SUN']/b)
 assert close(v['AUDC'],sum(curve[i][1]+curve[i+1][1]for i in range(b))/(b*b))
 assert e['candidate_oracle_attempts']==b and e['initialization_oracle_attempts']==96
 assert o['candidate_generation_intents']==o['candidate_generation_returns']+o['candidate_generation_context_rejections']
 assert o['candidate_generation_returns']==e['llm_calls']==o['tool_decisions']*(1 if r['arm']=='Native1'else 2)
 assert o['prompt_tokens']==e['prompt_tokens']and o['completion_tokens']==e['completion_tokens']
 assert all(o[k]==0 for k in('capture_forward_intents','capture_forward_returns','selector_calls','risk_used_decisions','graph_calls','ES_updates','fits','counterfactual_outcome_labels'))
 assert c['whole_job_wall_seconds']>=c['cold_load_seconds']+e['initialization_wall_seconds']+e['wall_seconds']
 assert len(r['completion']['sha256'])==64 and r['original_raw_scientific_evidence_reconstructed']
by={r['arm']:r for r in accepted};pair=s['completed_pair']
for key in('SUN','mSUN','AUDC'):assert close(pair['delta_'+key],by['Common2']['metrics'][key]-by['Native1']['metrics'][key])
assert close(pair['delta_whole_job_wall_seconds'],by['Common2']['costs']['whole_job_wall_seconds']-by['Native1']['costs']['whole_job_wall_seconds'])
assert pair['matched_pairs']==1 and not pair['internal_information_effect_tested']and not pair['ES_effect_tested']and not pair['statistical_significance_claimed']
assert s['method']['model']['selected_generation']==0 and s['method']['ES_updates']==0
assert s['method']['evaluator']['orb_num_workers']==1 and s['method']['evaluator']['mace_num_workers']==4
assert s['method']['internal_arms_results_in_this_snapshot']==0 and not s['new_holdout_claimed']
assert s['native_completion_bytes_reconstructed_and_sha_verified']
assert m['new_scientific_calls']==s['new_scientific_calls_for_publication']==0
md=(P/'report.md').read_text()
assert sum(line.startswith('| Mg-Sn-Sr |')or line.startswith('| Au-K-Tb |')for line in md.splitlines())==20
for r in accepted:
 v=r['metrics'];assert f"| {r['task']} | {r['budget']} | {r['arm']} | 已完成 | {v['SUN']} | {v['mSUN']:.3f} | {v['AUDC']:.3f} | {r['costs']['whole_job_wall_seconds']:.3f} |" in md
print(json.dumps({'status':'passed','expected':20,'accepted':2,'started_incomplete':2,'pending':16,'completed_pairs':1,'accepted_candidate_ORB_attempts':20,'internal_effect_tested':False,'new_scientific_calls':0},sort_keys=True))
