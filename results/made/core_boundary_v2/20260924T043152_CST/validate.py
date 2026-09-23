from pathlib import Path
import hashlib,json,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_bytes())
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
m=read('manifest.json');assert set(m['files'])=={p.name for p in P.iterdir()if p.is_file()and p.name!='manifest.json'}
for n,pin in m['files'].items():
 raw=(P/n).read_bytes();assert len(raw)==pin['bytes']and hashlib.sha256(raw).hexdigest()==pin['sha256']
s=read('snapshot.json');pairs=s['completed_pairs'];assert len(pairs)==7 and all(p['budget']==10 for p in pairs)
assert len(s['observed_records'])==17 and len({r['job_id']for r in s['observed_records']})==17
for p in pairs:
 for a in('Native1','ActionPool2'):assert p[a]['status']=='accepted'and close(p[a]['metrics']['mSUN'],p[a]['metrics']['SUN']/10)
 for k in('SUN','AUDC'):assert close(p['difference'][k],p['ActionPool2']['metrics'][k]-p['Native1']['metrics'][k])
for task,z in [(None,s['B10_statistics']),*((c['task'],c['statistics'])for c in s['B10_by_chemistry'])]:
 group=[p for p in pairs if task is None or p['task']==task];assert z['complete_pairs_in_coverage']==len(group)
 for k in('SUN','AUDC'):
  for a in('Native1','ActionPool2'):assert close(z['means'][a][k],sum(p[a]['metrics'][k]for p in group)/len(group))
  assert close(z['means']['difference'][k],sum(p['difference'][k]for p in group)/len(group))
  assert z['win_tie_loss'][k]=={'wins':sum(p['difference'][k]>1e-12 for p in group),'ties':sum(abs(p['difference'][k])<=1e-12 for p in group),'losses':sum(p['difference'][k]<-1e-12 for p in group)}
assert close(s['B10_statistics']['means']['Native1']['SUN'],22/7)and close(s['B10_statistics']['means']['ActionPool2']['SUN'],37/7)
assert close(s['B10_statistics']['means']['difference']['AUDC'],.25)
assert s['B10_statistics']['win_tie_loss']['SUN']=={'wins':6,'ties':0,'losses':1}
assert s['B10_statistics']['win_tie_loss']['AUDC']=={'wins':7,'ties':0,'losses':0}
assert s['global_queue_counts']is None and s['eight_GPU_current_state']is None and not s['later_results_merged']
assert not s['statistical_significance_claimed']and not s['uncertainty_net_increment_established']and not s['missing_metrics_imputed']
for c in s['seed501_only_Common2_checks']:
 assert c['seed']==501 and not c['Common2_uses_internal_risk']
 for k in('SUN','AUDC'):assert close(c['difference'][k],c['ActionPool2']['metrics'][k]-c['Common2']['metrics'][k])
assert m['new_scientific_calls']==s['new_scientific_calls_for_publication']==0
print(json.dumps({'status':'passed','complete_pairs_in_coverage':7,'SUN_win_tie_loss':[6,0,1],'AUDC_win_tie_loss':[7,0,0],'global_queue_or8_GPU_state_claimed':False,'new_scientific_calls':0},sort_keys=True))
