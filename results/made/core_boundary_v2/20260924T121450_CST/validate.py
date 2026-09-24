from pathlib import Path
from collections import Counter
import hashlib,json,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_bytes())
def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
m=read('manifest.json');assert set(m['files'])=={p.name for p in P.iterdir()if p.is_file()and p.name!='manifest.json'}
for n,pin in m['files'].items():
 raw=(P/n).read_bytes();assert len(raw)==pin['bytes']and hashlib.sha256(raw).hexdigest()==pin['sha256']
s=read('snapshot.json');pairs=s['pairs'];rows=[r for p in pairs for r in p['arms'].values()];assert len(pairs)==60 and len(rows)==len({r['job_id']for r in rows})==120
assert Counter(r['state']for r in rows)==s['all_status_counts']=={'accepted':64,'active_or_claimed':2,'pending':54}
assert Counter(r['state']for r in rows if r['origin']=='new_v2')==s['new_status_counts']=={'accepted':60,'active_or_claimed':2,'pending':54}
complete=[p for p in pairs if p['accepted_pair']];assert len(complete)==31
for p in pairs:
 assert p['accepted_pair']==all(r['accepted']for r in p['arms'].values())
 if not p['accepted_pair']:assert p['ActionPool2_minus_Native1']is None;continue
 for k in('SUN','AUDC','mSUN'):assert close(p['ActionPool2_minus_Native1'][k],p['arms']['ActionPool2']['metrics'][k]-p['arms']['Native1']['metrics'][k])
for g in s['descriptive_statistics']:
 ps=[p for p in complete if p['budget']==g['budget']and(g['task_id']is None or p['task_id']==g['task_id'])];assert len(ps)==g['complete_pairs']
 original=next(v for v in s['observer_mean_WTL_crosscheck']if(v['budget'],v['task_id'])==(g['budget'],g['task_id']))
 for metric,v in g['metrics'].items():
  values={a:[p['arms'][a]['metrics'][metric]for p in ps]for a in('Native1','ActionPool2')};values['paired_difference']=[a-n for a,n in zip(values['ActionPool2'],values['Native1'])]
  for arm,xs in values.items():
   z=v[arm];n=len(xs);assert z['n']==n and z['variance_ddof']==1
   if not n:assert z['mean']is z['sample_variance']is z['sample_SD']is None;continue
   mean=sum(xs)/n;assert close(z['mean'],mean)
   if n<2:assert z['sample_variance']is z['sample_SD']is None
   else:
    variance=sum((x-mean)**2 for x in xs)/(n-1);assert close(z['sample_variance'],variance)and close(z['sample_SD'],math.sqrt(variance))
  if ps:
   for arm in('Native1','ActionPool2'):assert close(v[arm]['mean'],original['metrics'][metric][arm+'_mean'])
   assert close(v['paired_difference']['mean'],original['metrics'][metric]['mean_paired_difference'])
  ds=values['paired_difference'];assert v['win_tie_loss']=={'wins':sum(x>1e-12 for x in ds),'ties':sum(abs(x)<=1e-12 for x in ds),'losses':sum(x<-1e-12 for x in ds)}
by={g['budget']:g for g in s['descriptive_statistics']if g['task_id']is None}
assert [by[b]['complete_pairs']for b in(10,20,30)]==[20,11,0]
assert by[20]['chemistry_counts']=={'Au-K-Tb':6,'Mg-Sn-Sr':5}and by[20]['metrics']['SUN']['win_tie_loss']=={'wins':4,'ties':2,'losses':5}
assert close(by[10]['metrics']['AUDC']['paired_difference']['mean'],-.005)
assert s['prior11_complete_pair_refs_and_metrics_identical']and s['SD_is_not_SE_or_confidence_interval']and s['only_complete_pairs_in_statistics']
assert not s['statistical_significance_claimed']and not s['uncertainty_net_increment_established']and not s['incomplete_pair_metrics_imputed']and not s['later_results_merged']
assert m['new_scientific_calls']==s['new_scientific_calls_for_publication']==0
print(json.dumps({'status':'passed','accepted':64,'complete_pairs':31,'B10_pairs':20,'B20_pairs':11,'variance_ddof':1,'sample_variances_SD_and_paired_SD_recomputed':True,'new_scientific_calls':0},sort_keys=True))
