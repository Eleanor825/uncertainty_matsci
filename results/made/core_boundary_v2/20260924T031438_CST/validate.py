from pathlib import Path
import json,hashlib,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_bytes())
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
m=read('manifest.json');assert set(m['files'])=={p.name for p in P.iterdir()if p.is_file()and p.name!='manifest.json'}
for n,pin in m['files'].items():
 raw=(P/n).read_bytes();assert len(raw)==pin['bytes']and hashlib.sha256(raw).hexdigest()==pin['sha256']
s=read('snapshot.json');rows=s['completed_comparison'];z={r['arm']:r for r in rows};assert set(z)=={'Native1','Common2','ActionPool2'}
assert all(r['status']=='accepted'and r['task']=='Au-K-Tb'and r['budget']==10 and r['seed']==501 for r in rows)
assert [z[a]['metrics']['SUN']for a in('Native1','Common2','ActionPool2')]==[0,3,1]
assert [z[a]['metrics']['AUDC']for a in('Native1','Common2','ActionPool2')]==[0,.39,.17]
for r in rows:assert close(r['metrics']['mSUN'],r['metrics']['SUN']/10)and r['costs']['reconstructed_scientific_costs']['candidate_oracle_attempts']==10
for a in('Native1','Common2'):
 for k in('SUN','mSUN','AUDC'):assert close(s['ActionPool2_minus_control'][a][k],z['ActionPool2']['metrics'][k]-z[a]['metrics'][k])
c=s['same_packet_mechanism']['counts'];details=s['same_packet_mechanism']['details'];assert c['selection_events']==len(details)==15 and c['risk_used']==9 and c['common_fallback']==6
assert c['tool_args_key_changed']==sum(x['tool_args_key_changed']for x in details)==5
assert c['risk_used_with_two_distinct_valid_actions']==7 and c['score_calls']==20
o=z['ActionPool2']['costs']['online_costs'];assert o['capture_forward_returns']==o['capture_forward_intents']==40 and o['candidate_generation_returns']==30 and o['risk_used_decisions']==9
assert o['ES_updates']==o['graph_calls']==o['fits']==o['counterfactual_outcome_labels']==0
p=s['incomplete_progress'];assert p['status']=='incomplete'and p['candidate_evaluations_observed']==p['partial_official_SUN']==5 and p['final_metrics']is None and not p['compared_with_completed_Native']
assert not s['internal_uncertainty_net_benefit_established']and not s['statistical_significance_claimed']and not s['new_holdout_claimed']and not s['global_core_completed_count_inferred']
assert s['new_scientific_calls_for_publication']==m['new_scientific_calls']==0
md=(P/'report.md').read_text();assert all(f"| {a} |"in md for a in z)
print(json.dumps({'status':'passed','new_completed':1,'new_incomplete':1,'completed_control_trajectories':2,'internal_net_benefit_claimed':False,'new_scientific_calls':0},sort_keys=True))
