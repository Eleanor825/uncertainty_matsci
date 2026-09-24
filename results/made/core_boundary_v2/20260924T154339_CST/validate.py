"""Independent CPU arithmetic and manifest check for this fixed public snapshot."""
from pathlib import Path
from collections import Counter
import hashlib,json,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_bytes())
def close(x,y):return math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12)
m=read('manifest.json');assert set(m['files'])=={'report.md','snapshot.json','provenance.json','validate.py'}
for name,pin in m['files'].items():
 b=(P/name).read_bytes();assert len(b)==pin['bytes']and hashlib.sha256(b).hexdigest()==pin['sha256']
s=read('snapshot.json');pairs=s['pairs'];rows=[r for p in pairs for r in p['arms'].values()]
assert len(pairs)==len({(p['budget'],p['group'],p['seed'])for p in pairs})==60
assert len(rows)==len({r['job_id']for r in rows})==120
assert Counter(r['state']for r in rows)==s['all_status_counts']
for p in pairs:
 assert set(p['arms'])==set(s['arms'])and p['budget']in(10,20,30)and 501<=p['seed']<=510
 assert p['accepted_pair']==all(r['accepted']for r in p['arms'].values())
 for r in p['arms'].values():
  assert r['accepted']==(r['state']=='accepted')
  if r['accepted']:assert r['metrics']is not None and len(r['completion_ref']['sha256'])==64
complete=[p for p in pairs if p['accepted_pair']];assert len(complete)==s['complete_pairs']
for g in s['descriptive_statistics']:
 ps=[p for p in complete if p['budget']==g['budget']and(g['group']is None or p['group']==g['group'])]
 assert len(ps)==g['complete_pairs']and sum(g['group_counts'].values())==len(ps)
 for metric,v in g['metrics'].items():
  values={arm:[p['arms'][arm]['metrics'][metric]for p in ps]for arm in s['arms']}
  values['paired_difference']=[b-a for a,b in zip(values[s['arms'][0]],values[s['arms'][1]])]
  for arm,xs in values.items():
   z=v[arm];assert z['n']==len(xs)
   if not xs:assert z['mean']is z['sample_variance']is z['sample_SD']is None;continue
   mean=sum(xs)/len(xs);assert close(z['mean'],mean)
   if len(xs)>1:
    variance=sum((x-mean)**2 for x in xs)/(len(xs)-1);assert close(z['sample_variance'],variance)and close(z['sample_SD'],math.sqrt(variance))
   else:assert z['sample_variance']is z['sample_SD']is None
  ds=values['paired_difference'];assert v['win_tie_loss']=={'wins':sum(x>1e-12 for x in ds),'ties':sum(abs(x)<=1e-12 for x in ds),'losses':sum(x< -1e-12 for x in ds)}
budgets={g['budget']:g for g in s['descriptive_statistics']if g['group']is None}
if s['benchmark']=='MADE':
 assert s['all_status_counts']=={'accepted':81,'active_or_claimed':1,'pending':38}
 assert [budgets[b]['complete_pairs']for b in(10,20,30)]==[20,20,0]
 assert Counter(r['state']for r in rows if r['origin']=='new_v2')==s['new_status_counts']=={'accepted':77,'active_or_claimed':1,'pending':38}
 assert s['fixed_reuse_accepted']==4 and s['previous38_complete_refs_and_metrics_identical']
 assert budgets[20]['group_counts']=={'Au-K-Tb':10,'Mg-Sn-Sr':10}
 for g in s['descriptive_statistics']:
  original=next(x for x in s['observer_mean_WTL_crosscheck']if(x['budget'],x['task_id'])==(g['budget'],g['group']))
  for metric,v in g['metrics'].items():
   for arm in s['arms']:
    if g['complete_pairs']:assert close(v[arm]['mean'],original['metrics'][metric][arm+'_mean'])
   assert v['win_tie_loss']=={k:original['metrics'][metric][k]for k in('wins','ties','losses')}
 assert budgets[20]['metrics']['SUN']['win_tie_loss']=={'wins':6,'ties':5,'losses':9}
else:
 assert s['benchmark']=='DW'and s['all_status_counts']=={'accepted':42,'claimed':9,'pending':69}
 assert [budgets[b]['complete_pairs']for b in(10,20,30)]==[19,0,0]
 assert budgets[10]['group_counts']=={'3':10,'4':9}and s['job_id_prefix_not_used_for_matching']
 assert all(not r['metrics']['completed_successfully']for r in rows if r['accepted'])
 assert budgets[10]['metrics']['execution_success_rate']['win_tie_loss']=={'wins':10,'ties':7,'losses':2}
 assert budgets[10]['metrics']['final_score_normalized']['win_tie_loss']=={'wins':5,'ties':13,'losses':1}
 expected={'A':(.9566666666666668,.07675203260533903,.2459582470651367), 'B':(.9433333333333334,.07533991237698298,.25787525236760656), 'C':(.9833333333333334,.08440221836089902,.32522925329716157), 'P':(.9666666666666667,.062042760021031866,.20336319621990662), 'P+B-hidden':(.9533333333333334,.07175848559667802,.24272086510018728), 'P+C-hidden':(.9866666666666667,.07724912376604036,.2944723679300659)}
 for arm,values in expected.items():
  v=s['offline']['models'][arm]['dev_metrics'];assert v['rows']==40
  for key,want in zip(('pooled_AUROC','micro_Brier','micro_NLL'),values):assert close(v[key],want)
# Independently reconstruct paired seed-block intervals from the published rows.
import random
rng=random.Random(20260924);draws=[[rng.randrange(10)for _ in range(10)]for _ in range(20000)]
for g in s['descriptive_statistics']:
 ps=[p for p in complete if p['budget']==g['budget']and(g['group']is None or p['group']==g['group'])]
 for metric,v in g['metrics'].items():
  for arm in (*s['arms'],'paired_difference'):
   z=v[arm]
   if not ps:assert z['conditional_seed_bootstrap_CI95']is None;continue
   blocks=[]
   for seed in range(501,511):
    selected=[p for p in ps if p['seed']==seed]
    values=[p['arms'][arm]['metrics'][metric]if arm in s['arms']else p['arms']['ActionPool2']['metrics'][metric]-p['arms']['Native1']['metrics'][metric]for p in selected]
    blocks.append(sum(values)/len(values))
   sample=sorted(sum(blocks[i]for i in draw)/10 for draw in draws);expected=[]
   for q in (.025,.975):
    at=(len(sample)-1)*q;lo=int(at);weight=at-lo;expected.append(sample[lo]*(1-weight)+sample[lo+1]*weight)
   assert all(close(a,b)for a,b in zip(expected,z['conditional_seed_bootstrap_CI95']))
assert s['confidence_interval_method']['seed_blocks']==10 and s['confidence_interval_method']['replicates']==20000

assert s['variance_ddof']==1 and s['only_complete_pairs_in_statistics']and s['SD_is_not_SE_or_confidence_interval']
assert not s['statistical_significance_claimed']and not s['uncertainty_net_increment_established']and not s['incomplete_pair_metrics_imputed']and not s['later_results_merged']
assert s['new_scientific_calls_for_publication']==m['new_scientific_calls']==0
print(json.dumps({'status':'passed','benchmark':s['benchmark'],'accepted':s['all_status_counts']['accepted'],'complete_pairs':len(complete),'variance_ddof':1,'sample_variance_SD_and_paired_SD_recomputed':True,'paired_seed_block_CI95_recomputed':True,'new_scientific_calls':0},sort_keys=True))
