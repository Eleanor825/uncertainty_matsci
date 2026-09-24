"""Recompute complete matching, sample dispersion and paired seed-block CIs."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,random
P=Path(__file__).resolve().parent
def read(name):return json.loads((P/name).read_bytes())
def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
manifest=read('manifest.json');assert set(manifest['files'])=={'report.md','snapshot.json','provenance.json','validate.py'}
for name,pin in manifest['files'].items():
 raw=(P/name).read_bytes();assert len(raw)==pin['bytes']and hashlib.sha256(raw).hexdigest()==pin['sha256']
s=read('snapshot.json');pairs=s['pairs'];arms=('Native','ActionPool2');metrics=('execution_success_rate','final_score_normalized')
assert len(pairs)==s['complete_pairs']==60 and s['expected_cells']==120 and s['all_status_counts']=={'accepted':120}
assert {(p['budget'],p['group'],p['seed'])for p in pairs}=={(b,w,k)for b in(10,20,30)for w in(3,4)for k in range(501,511)}
rows=[r for p in pairs for r in p['arms'].values()];assert len(rows)==len({r['job_id']for r in rows})==120
assert all(r['accepted']and r['state']=='accepted'and not r['metrics']['completed_successfully']and r['costs']['unknown_action_outcomes']==0 for r in rows)
assert s['task_success_counts']=={'Native':0,'ActionPool2':0}and s['task_success_denominators']=={'Native':60,'ActionPool2':60}
assert all(p['accepted_pair']and set(p['arms'])==set(arms)for p in pairs)
ci=s['confidence_interval_method'];assert ci['replicates']==20000 and ci['random_seed']==20260924 and ci['seed_blocks']==10 and ci['level']==.95
rng=random.Random(20260924);draws=[[rng.randrange(10)for _ in range(10)]for _ in range(20000)]
def percentile(xs,prob):
 xs.sort();n=(len(xs)-1)*prob;i=math.floor(n);f=n-i;return (1-f)*xs[i]+f*xs[min(i+1,len(xs)-1)]
for g in s['descriptive_statistics']:
 ps=[p for p in pairs if p['budget']==g['budget']and(g['group']is None or p['group']==g['group'])]
 assert len(ps)==g['complete_pairs']==(20 if g['group']is None else 10)
 for metric in metrics:
  values={a:[p['arms'][a]['metrics'][metric]for p in ps]for a in arms};values['paired_difference']=[b-a for a,b in zip(values['Native'],values['ActionPool2'])]
  v=g['metrics'][metric]
  for arm,xs in values.items():
   mean=sum(xs)/len(xs);variance=sum((x-mean)**2 for x in xs)/(len(xs)-1);z=v[arm]
   assert z['n']==len(xs)and close(z['mean'],mean)and close(z['sample_variance'],variance)and close(z['sample_SD'],math.sqrt(variance))
   blocks=[]
   for seed in range(501,511):
    within=[p for p in ps if p['seed']==seed]
    xs2=[p['arms'][arm]['metrics'][metric]if arm in arms else p['arms']['ActionPool2']['metrics'][metric]-p['arms']['Native']['metrics'][metric]for p in within]
    blocks.append(sum(xs2)/len(xs2))
   boot=[sum(blocks[i]for i in draw)/10 for draw in draws]
   expected=[percentile(boot.copy(),.025),percentile(boot.copy(),.975)]
   assert all(close(a,b)for a,b in zip(expected,z['conditional_seed_bootstrap_CI95']))
  ds=values['paired_difference'];assert v['win_tie_loss']=={'wins':sum(x>1e-12 for x in ds),'ties':sum(abs(x)<=1e-12 for x in ds),'losses':sum(x< -1e-12 for x in ds)}
pooled={g['budget']:g for g in s['descriptive_statistics']if g['group']is None}
for b,wants in {10:(.36,.535,.03125,.06875),20:(.3825,.595,.03125,.06875),30:(.3933333333333333,.6266666666666667,.0875,.075)}.items():
 actual=[pooled[b]['metrics'][m][a]['mean']for m in metrics for a in arms];assert all(close(a,b)for a,b in zip(actual,wants))
assert s['previous42_accepted_refs_and_metrics_identical']and s['all_registered_trajectories_complete']and not s['completion_subset_may_be_biased']
assert not s['statistical_significance_claimed']and not s['uncertainty_net_increment_established']and not s['incomplete_pair_metrics_imputed']and not s['later_results_merged']
assert s['variance_ddof']==1 and s['new_scientific_calls_for_publication']==manifest['new_scientific_calls']==0
print(json.dumps({'status':'passed','accepted':120,'complete_pairs':60,'pairs_each_budget':20,'sample_variance_SD_paired_WTL_recomputed':True,'bootstrap_CI_recomputed':True,'bootstrap_seed_blocks':10,'bootstrap_replicates':20000,'new_scientific_calls':0},sort_keys=True))
