"""Validate this fixed publication without GPU, network or private server files."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_bytes())
m=read('manifest.json')
assert set(m['files'])=={p.name for p in P.iterdir() if p.is_file() and p.name!='manifest.json'}
for n,r in m['files'].items():
 b=(P/n).read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'],n
v=read('snapshot.json');s=read('summary.json');proof=read('scientific_sha256.json');rows=v['rows']
assert hashlib.sha256((json.dumps(v,sort_keys=True)+'\n').encode()).hexdigest()==proof['exported_payload_sha256']
assert len(rows)==len({r['job_id']for r in rows})==20 and v['expected']==20
arms=('Native','G0GraphRisk','G2ControllerOff','G2Full','Hidden145Fixed2')
assert {(r['task_id'],r['budget'],r['arm'],r['seed'])for r in rows}=={(t,b,a,1)for t in ('Mg-Sn-Sr','Au-K-Tb')for b in (5,15)for a in arms}
accepted=[r for r in rows if r['status']in('accepted_original','accepted_recovery')]
assert len(accepted)==v['accepted']==s['accepted']==9 and s['unfinished']==11
assert dict(Counter(r['status']for r in rows))==s['status_counts']
assert sum(r['original_init_failure_preserved']for r in rows)==v['original_failure_attempts_preserved']==8
assert read('accepted_metrics.json')==[{k:x for k,x in r.items()if k!='curve'}for r in accepted]
assert read('curves.json')==[{k:r[k]for k in ('job_id','task_id','budget','seed','arm','curve','result_ref')}for r in accepted]
for r in rows:
 if r in accepted:
  c=r['curve'];B=r['budget'];assert len(c)==B+1 and [x[0]for x in c]==list(range(B+1)) and c[0]==[0,0]
  assert r['SUN']==c[-1][1] and math.isclose(r['mSUN'],r['SUN']/B,abs_tol=1e-12)
  assert math.isclose(r['AUDC'],sum(c[i][1]+c[i+1][1]for i in range(B))/B**2,abs_tol=1e-12)
  assert r['costs']['candidate_oracle_attempts']==B and r['result_ref'] and r['completion_ref']
  assert (r['adoption_ref']is not None)==(r['status']=='accepted_recovery')
  assert r['selected_generation']==(2 if r['arm']in('G2ControllerOff','G2Full')else 0)
 else:assert all(r[k]is None for k in ('SUN','AUDC','mSUN','result_ref','completion_ref','adoption_ref','costs','curve'))
assert sum(r['budget']for r in accepted)==s['accepted_candidate_oracle_attempts']==65
assert s['by_budget']=={'5':{'accepted':7,'expected':10},'15':{'accepted':2,'expected':10}}
assert v['new_scientific_calls']==s['new_scientific_calls']==m['new_scientific_calls']==0
print(json.dumps({'status':'passed','accepted':9,'unfinished':11,'canonical_jobs':20,'candidate_ORB_attempts_in_accepted':65,'new_scientific_calls':0}))
