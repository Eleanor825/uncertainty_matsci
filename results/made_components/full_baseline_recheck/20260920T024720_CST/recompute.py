"""Recompute a fixed interim subset and the separate complete seed1 B50 archive."""
from pathlib import Path
import csv,hashlib,io,json,math,statistics
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text())
def stats(v):return {'n':len(v),'mean':statistics.mean(v),'sample_variance_ddof1':statistics.variance(v),'SD':statistics.stdev(v)}
def sign(v):return 'positive' if v>1e-12 else 'negative' if v< -1e-12 else 'tie'
def csvbytes(rows):
 s=io.StringIO(newline='');w=csv.DictWriter(s,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()
def build():
 x=read('inputs.json');current=x['current'];old=x['old_B50'];arms=('baseline_reference','full_support_aware')
 accepted={(r['arm'],r['task_id'],r['budget'],r['seed']):r for r in current['rows'] if r['status']=='accepted'}
 systems=[t for t in current['task_ids'] if all((a,t,10,z) in accepted for a in arms for z in (2,3,4))]
 table=[];example={};seedrows=[]
 for t in systems:
  for metric in ('SUN','AUDC'):
   vals=[[accepted[a,t,10,z][metric] for z in (2,3,4)] for a in arms]
   for a in arms:
    for z in (2,3,4):
     r=accepted[a,t,10,z];cv=r['curve'];assert [q for q,v in cv]==list(range(11))
     area=sum((q1-q0)*(v1+v0) for (q0,v0),(q1,v1) in zip(cv,cv[1:]))/100
     assert r['SUN']==cv[-1][1] and math.isclose(area,r['AUDC'],abs_tol=1e-12)
   b,f=map(stats,vals);delta=f['mean']-b['mean']
   row={'task_id':t,'metric':metric,'baseline_mean':b['mean'],'full_mean':f['mean'],
        'baseline_sample_variance_ddof1':b['sample_variance_ddof1'],'full_sample_variance_ddof1':f['sample_variance_ddof1'],
        'baseline_SD':b['SD'],'full_SD':f['SD'],'mean_delta':delta,'mean_sign':sign(delta),
        'seed_wins':sum(y>x+1e-12 for x,y in zip(*vals)),'seed_ties':sum(abs(y-x)<=1e-12 for x,y in zip(*vals)),
        'seed_losses':sum(y<x-1e-12 for x,y in zip(*vals))}
   table.append(row)
   for z,bv,fv in zip((2,3,4),*vals):seedrows.append({'task_id':t,'seed':z,'metric':metric,'baseline':bv,'full':fv,'delta':fv-bv})
   if t=='Ag-Nd-Pd-Pt-Tb':example[metric]={'baseline':vals[0],'full':vals[1],**row}
 oldmap={(r['task_id'],r['method']):r for r in old};assert len(old)==60 and len({r['task_id'] for r in old})==30
 oldsummary={}
 for r in old:
  assert r['seed']==1 and r['budget']==50
  cv=r['curve'];assert [q for q,v in cv]==list(range(51))
  assert r['SUN']==cv[-1][1] and math.isclose(sum((q1-q0)*(v1+v0) for (q0,v0),(q1,v1) in zip(cv,cv[1:]))/2500,r['AUDC'],abs_tol=1e-12)
 for metric in ('SUN','AUDC'):
  by={a:[r[metric] for r in old if r['method']==a] for a in ('baseline','esopt_graph_risk')}
  assert all(len(v)==30 for v in by.values())
  oldsummary[metric]={'baseline_sum':sum(by['baseline']),'full_sum':sum(by['esopt_graph_risk']),
   'baseline_mean':statistics.mean(by['baseline']),'full_mean':statistics.mean(by['esopt_graph_risk'])}
 assert oldsummary['SUN']['baseline_sum']==233 and oldsummary['SUN']['full_sum']==147
 summary={'observed_CST':current['observed_CST'],'current_main_accepted':395,'current_planned':1080,
  'B10_complete_three_seed_baseline_full_systems':len(systems),'systems':systems,
  'missing_complete_three_seed_baseline_full_systems':[t for t in current['task_ids'] if t not in systems],
  'sign_counts':{m:{s:sum(r['metric']==m and r['mean_sign']==s for r in table) for s in ('positive','tie','negative')} for m in ('SUN','AUDC')},
  'current_B50_baseline_full_matched_task_seeds':sum(all((a,t,50,z) in accepted for a in arms) for t in current['task_ids'] for z in (2,3,4)),
  'positive_example':example,'old_seed1_B50':oldsummary,'new_scientific_calls':0}
 report=['# Full-method versus baseline: fixed interim evidence','',
  f"Current snapshot: **{current['observed_CST']}**, **395/1080** completed evaluations. Baseline and full both have all three B10 seeds for **{len(systems)}/30 systems**. This is an interim completion-defined subset, not a preselected all-system estimate.",'',
  '| Metric | Positive system means | Tied system means | Negative system means |','|---|---:|---:|---:|']
 for m,c in summary['sign_counts'].items():report.append(f"| {m} | {c['positive']} | {c['tie']} | {c['negative']} |")
 report+=['','The unit above is a system mean over evaluation seeds2/3/4. `system_statistics.csv` includes every eligible positive, zero and negative system; `paired_seeds.csv` retains each seed. Sample variance/SD are within-system evaluation-seed statistics, not across-system or training-run variance.','',
  'Ag–Nd–Pd–Pt–Tb is a positive descriptive example: SUN [1,5,3] → [4,6,3], mean3 →4.333333, sample variance4 →2.333333; AUDC [.19,.43,.51] → [.46,.60,.51], mean.3766667 →.5233333, sample variance.0277333 →.0050333. Both metrics have two winning seeds and one tie. This example was identified from completed results and does not demonstrate general efficacy or causal NN benefit.','',
  'The **separate original seed1 B50 study** is complete for30 systems/60 trajectories: baseline totalSUN233 versus original fixed-G2 method147; meanAUDC.15884 versus.1127866667. That older method/seed scope must not be pooled with the new support-aware four-arm study. Its across-system dispersion is not evaluation-seed variance.','',
  f"At this current snapshot there are **{summary['current_B50_baseline_full_matched_task_seeds']} complete new B50 baseline/full task-seed pairs**. The old B50 negative aggregate cannot be presented as a current-study B50 result, and newly running B50 jobs have no publishable paired effect yet.",'',
  'Paper framing supported by these records: performance is heterogeneous, with positive examples across evaluation seeds and substantial negative cases. The evidence does not justify a generic improvement claim; component/controller and NN contributions must remain separate. No new fitting, model call or oracle invocation was used for this recheck.','',
  'Run `python3 -B recompute.py`. Inputs retain source snapshot/result hashes and complete discovery curves.','']
 return {'summary.json':(json.dumps(summary,indent=2)+'\n').encode(),'system_statistics.csv':csvbytes(table),'paired_seeds.csv':csvbytes(seedrows),'REPORT.md':'\n'.join(report).encode()}
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
 for name,data in build().items():
  if a.write:(ROOT/name).write_bytes(data)
  else:assert (ROOT/name).read_bytes()==data,name
 print(json.dumps({'passed':True,'new_scientific_calls':0}))
if __name__=='__main__':main()
