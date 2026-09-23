"""Validate the sanitized fixed snapshot without network or scientific calls."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_bytes())
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
m=read('manifest.json');assert set(m['files'])=={p.name for p in P.iterdir()if p.is_file()and p.name!='manifest.json'}
for n,pin in m['files'].items():
 raw=(P/n).read_bytes();assert len(raw)==pin['bytes']and hashlib.sha256(raw).hexdigest()==pin['sha256']
s=read('snapshot.json');rows=s['rows'];arms=('Native1','Common2','Public2','FullPool2','ActionPool2')
assert len(rows)==len({r['job_id']for r in rows})==20
assert {(r['task'],r['budget'],r['arm'],r['seed'])for r in rows}=={(t,b,a,501)for t in('Mg-Sn-Sr','Au-K-Tb')for b in(10,20)for a in arms}
assert Counter(r['status']for r in rows)=={'accepted':4,'technical_failure':2,'unclaimed':14}
assert s['accepted']==4 and s['technical_failed']==2 and s['unclaimed']==14 and not s['complete']
assert sum(r['budget']for r in rows)==s['registered_candidate_ORB_attempts']==300
accepted=[r for r in rows if r['status']=='accepted'];assert sum(r['budget']for r in accepted)==s['accepted_candidate_ORB_attempts']==40
for r in rows:
 if r['status']!='accepted':
  assert r['metrics']is None and r['costs']is None and r['completion']is None
  if r['status']=='technical_failure':assert r['task']=='Mg-Sn-Sr'and r['budget']==10 and r['arm']in('Public2','FullPool2')and r['failure']['error']=='Wrong action-position boundary'and not r['failure']['scientific_zero_result']
  continue
 assert r['arm']in('Native1','Common2')and r['budget']==10 and r['accepted_completion_bytes_sha_verified']
 v=r['metrics'];b=r['budget'];curve=v['discovery_curve'];o=r['costs']['online_costs'];e=r['costs']['reconstructed_scientific_costs']
 assert [x[0]for x in curve]==list(range(b+1))and curve[0]==[0,0]and v['SUN']==curve[-1][1]
 assert close(v['mSUN'],v['SUN']/b)and close(v['AUDC'],sum(curve[i][1]+curve[i+1][1]for i in range(b))/(b*b))
 assert e['candidate_oracle_attempts']==b
 assert o['candidate_generation_intents']==o['candidate_generation_returns']+o['candidate_generation_context_rejections']
 assert o['candidate_generation_returns']==e['llm_calls']==o['tool_decisions']*(1 if r['arm']=='Native1'else 2)
 assert all(o[k]==0 for k in('capture_forward_intents','capture_forward_returns','risk_used_decisions','selector_calls','graph_calls','ES_updates','fits','counterfactual_outcome_labels'))
for pair in s['completed_pairs']:
 z={r['arm']:r for r in accepted if r['task']==pair['task']}
 for k in('SUN','mSUN','AUDC'):assert close(pair['differences'][k],z['Common2']['metrics'][k]-z['Native1']['metrics'][k])
 assert not pair['internal_risk_used']and not pair['statistical_significance_claimed']
assert len(s['completed_pairs'])==2 and s['method']['accepted_internal_risk_trajectories']==0
assert not s['new_holdout_claimed']and not s['later_core_results_included']and not s['unfinished_metrics_imputed']and not s['unexecuted_candidate_labels_imputed']
p=read('provenance.json');assert p['boundary_diagnosis']['original_dataclass_CPU_boundary_reproduction']and not p['boundary_diagnosis']['models_or_environments_loaded']
md=(P/'report.md').read_text();table=[x for x in md.splitlines()if x.startswith('| Mg-Sn-Sr |')or x.startswith('| Au-K-Tb |')];assert len(table)==20
assert m['new_scientific_calls']==s['new_scientific_calls_for_publication']==0
print(json.dumps({'status':'passed','accepted':4,'technical_failed':2,'unclaimed':14,'completed_control_pairs':2,'internal_effect_tested':False,'new_scientific_calls':0},sort_keys=True))
