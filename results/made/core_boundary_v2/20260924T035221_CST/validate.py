from pathlib import Path
from collections import Counter
import hashlib,json,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_bytes())
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
m=read('manifest.json');assert set(m['files'])=={p.name for p in P.iterdir()if p.is_file()and p.name!='manifest.json'}
for n,pin in m['files'].items():
 b=(P/n).read_bytes();assert len(b)==pin['bytes']and hashlib.sha256(b).hexdigest()==pin['sha256']
s=read('snapshot.json');rows=s['rows'];assert len(rows)==len({r['job_id']for r in rows})==120
assert Counter(r['status']for r in rows)=={'accepted_new':7,'accepted_reused':4,'started_incomplete':2,'unclaimed':107}
assert {(r['task'],r['budget'],r['seed'],r['arm'])for r in rows}=={(t,b,z,a)for t in('Mg-Sn-Sr','Au-K-Tb')for b in(10,20,30)for z in range(501,511)for a in('Native1','ActionPool2')}
pairs=s['completed_pairs'];assert len(pairs)==5 and all(p['budget']==10 for p in pairs)
for p in pairs:
 for a in('Native1','ActionPool2'):
  r=p[a];assert r['status'].startswith('accepted')and r in rows and r['costs']['candidate_oracle_attempts']==10
  assert close(r['metrics']['mSUN'],r['metrics']['SUN']/10)
 for k in('SUN','AUDC'):assert close(p['difference'][k],p['ActionPool2']['metrics'][k]-p['Native1']['metrics'][k])
for v in s['by_budget']+s['B10_by_chemistry']:
 g=[p for p in pairs if p['budget']==v['budget']and('task'not in v or p['task']==v['task'])];assert len(g)==v['completed_pairs']
 if not g:assert v['means']is None and v['win_tie_loss']is None;continue
 for k in('SUN','AUDC'):
  assert close(v['means']['difference'][k],sum(p['difference'][k]for p in g)/len(g))
  for a in('Native1','ActionPool2'):assert close(v['means'][a][k],sum(p[a]['metrics'][k]for p in g)/len(g))
  assert v['win_tie_loss'][k]=={'wins':sum(p['difference'][k]>1e-12 for p in g),'ties':sum(abs(p['difference'][k])<=1e-12 for p in g),'losses':sum(p['difference'][k]<-1e-12 for p in g)}
assert close(s['by_budget'][0]['means']['difference']['SUN'],1.4)and close(s['by_budget'][0]['means']['difference']['AUDC'],.186)
assert s['by_budget'][0]['win_tie_loss']['SUN']=={'wins':4,'ties':0,'losses':1}
for c in s['seed501_only_Common2_checks']:
 assert c['seed']==501 and not c['Common2_uses_internal_risk']
 for k in('SUN','AUDC'):assert close(c['difference'][k],c['ActionPool2']['metrics'][k]-c['Common2']['metrics'][k])
assert not s['statistical_significance_claimed']and not s['uncertainty_net_increment_established']and not s['missing_metrics_imputed']and not s['incomplete_pairs_included_in_summary']
assert m['new_scientific_calls']==s['new_scientific_calls_for_publication']==0
print(json.dumps({'status':'passed','accepted':11,'new_accepted':7,'reused':4,'complete_pairs':5,'SUN_win_tie_loss':[4,0,1],'AUDC_win_tie_loss':[5,0,0],'new_scientific_calls':0},sort_keys=True))
