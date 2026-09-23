"""Validate this fixed publication without GPU, network or private server files."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib,json,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_bytes())
m=read('manifest.json')
assert set(m['files'])=={p.name for p in P.iterdir()if p.is_file()and p.name!='manifest.json'}
for n,r in m['files'].items():
 b=(P/n).read_bytes();assert len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256'],n
v=read('snapshot.json');s=read('summary.json');proof=read('scientific_sha256.json');rows=v['rows']
assert hashlib.sha256((json.dumps(v,sort_keys=True)+'\n').encode()).hexdigest()==proof['exported_payload_sha256']
assert len(rows)==len({r['job_id']for r in rows})==20 and v['expected']==s['expected']==20
arms=('Native','G0GraphRisk','G2ControllerOff','G2Full','Hidden145Fixed2');systems=('Mg-Sn-Sr','Au-K-Tb');budgets=(5,15)
assert {(r['task_id'],r['budget'],r['arm'],r['seed'])for r in rows}=={(t,b,a,1)for t in systems for b in budgets for a in arms}
accepted=[r for r in rows if r['status']in('accepted_original','accepted_recovery')]
assert len(accepted)==v['accepted']==s['accepted']and s['unfinished']==20-len(accepted)
counts=Counter(r['status']for r in rows);assert dict(counts)==s['status_counts']
assert s['accepted_original']==counts['accepted_original']and s['accepted_recovery']==counts['accepted_recovery']
assert sum(r['original_init_failure_preserved']for r in rows)==v['original_failure_attempts_preserved']==s['original_init_failures_preserved']
assert read('accepted_metrics.json')==[{k:x for k,x in r.items()if k!='curve'}for r in accepted]
assert read('curves.json')==[{k:r[k]for k in ('job_id','task_id','budget','seed','arm','curve','result_ref')}for r in accepted]
assert proof['scientific_registration']==v['scientific_registration']
assert proof['scientific_result_refs']==[{'job_id':r['job_id'],**{k:r[k]for k in ('result_ref','completion_ref','adoption_ref')}}for r in accepted]
for r in rows:
 if r in accepted:
  c=r['curve'];B=r['budget'];assert len(c)==B+1 and [x[0]for x in c]==list(range(B+1))and c[0]==[0,0]
  assert r['SUN']==c[-1][1]and math.isclose(r['mSUN'],r['SUN']/B,abs_tol=1e-12)
  assert math.isclose(r['AUDC'],sum(c[i][1]+c[i+1][1]for i in range(B))/B**2,abs_tol=1e-12)
  assert r['costs']['candidate_oracle_attempts']==B and r['result_ref']and r['completion_ref']
  assert (r['adoption_ref']is not None)==(r['status']=='accepted_recovery')
  assert r['selected_generation']==(2 if r['arm']in('G2ControllerOff','G2Full')else 0)
 else:assert all(r[k]is None for k in ('SUN','AUDC','mSUN','result_ref','completion_ref','adoption_ref','costs','curve'))
 for k in ('result_ref','completion_ref','adoption_ref'):
  x=r[k]
  if x:assert not Path(x['path']).is_absolute()and '..'not in Path(x['path']).parts and len(x['sha256'])==64 and x['bytes']>0
assert sum(r['budget']for r in accepted)==s['accepted_candidate_oracle_attempts']
assert sum(r['budget']for r in rows)==s['planned_candidate_oracle_attempts']==200
assert s['by_budget']=={str(b):{'accepted':sum(r['budget']==b for r in accepted),'expected':sum(r['budget']==b for r in rows)}for b in budgets}
stamp=datetime.fromtimestamp(v['observed_at'],timezone(timedelta(hours=8)))
assert s['snapshot_CST']==stamp.isoformat()and s['snapshot_UTC']==stamp.astimezone(timezone.utc).isoformat()
lookup={(r['task_id'],r['budget'],r['arm']):r for r in rows};pairs=[]
for system in systems:
 for budget in budgets:
  native=lookup[system,budget,'Native']
  for arm in arms[1:]:
   method=lookup[system,budget,arm];closed=native in accepted and method in accepted
   pairs.append({'task_id':system,'budget':budget,'seed':1,'arm':arm,'native_job_id':native['job_id'],'method_job_id':method['job_id'],'native_status':native['status'],'method_status':method['status'],'pair_status':'both_accepted'if closed else'incomplete_pair','native_SUN':native['SUN'],'method_SUN':method['SUN'],'delta_SUN':method['SUN']-native['SUN']if closed else None,'native_AUDC':native['AUDC'],'method_AUDC':method['AUDC'],'delta_AUDC':method['AUDC']-native['AUDC']if closed else None})
assert read('paired_comparisons.json')==pairs
assert len(accepted)==20 and s['unfinished']==0 and s['all_canonical_jobs_accepted']is True and s['final_for_registered_Tiny20_batch']is True
stats=read('comparison_summaries.json');pm=stats['paired_arm_means'];bm=stats['by_budget_arm']
assert len(pm)==4 and {x['arm']for x in pm}==set(arms[1:])
assert len(bm)==10 and {(x['budget'],x['arm'])for x in bm}=={(b,a)for b in budgets for a in arms}
for g in pm:
 assert g['matched_blocks']==4 and g['systems']==2 and g['seeds']==1 and g['budgets']==list(budgets)
 for metric in ('SUN','mSUN','AUDC'):
  n=[lookup[t,b,'Native'][metric]for t in systems for b in budgets]
  a=[lookup[t,b,g['arm']][metric]for t in systems for b in budgets]
  assert math.isclose(g['native_mean'][metric],sum(n)/4,abs_tol=1e-12)
  assert math.isclose(g['method_mean'][metric],sum(a)/4,abs_tol=1e-12)
  assert math.isclose(g['mean_paired_delta'][metric],sum(x-y for x,y in zip(a,n))/4,abs_tol=1e-12)
for g in bm:
 assert g['systems']==g['accepted_trajectories']==2 and g['seeds']==1
 for metric in ('SUN','mSUN','AUDC'):
  n=[lookup[t,g['budget'],'Native'][metric]for t in systems]
  a=[lookup[t,g['budget'],g['arm']][metric]for t in systems]
  assert math.isclose(g['mean'][metric],sum(a)/2,abs_tol=1e-12)
  assert math.isclose(g['mean_paired_delta_vs_Native'][metric],sum(x-y for x,y in zip(a,n))/2,abs_tol=1e-12)
assert stats['repeated_systems_across_budgets']is True and stats['independent_systems']==2 and stats['policy_seeds']==1
assert stats['fresh_holdout']is False and stats['significance_test_performed']is False and stats['Hidden145Fixed2_is_separate_method']is True
assert v['new_scientific_calls']==s['new_scientific_calls']==m['new_scientific_calls']==proof['new_scientific_calls']==0
print(json.dumps({'status':'passed','accepted':len(accepted),'unfinished':20-len(accepted),'canonical_jobs':20,'candidate_ORB_attempts_in_accepted':sum(r['budget']for r in accepted),'completed_method_native_pairs':sum(p['pair_status']=='both_accepted'for p in pairs),'new_scientific_calls':0}))
