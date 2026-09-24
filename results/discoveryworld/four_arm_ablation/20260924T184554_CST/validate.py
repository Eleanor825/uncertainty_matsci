"""Recompute public fixed-snapshot counts, means, sample variance/SD and contrasts."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,re
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def equal(a,b):
 if b is None:assert a is None
 else:assert math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12),(a,b)
def stats(record,values):
 n=len(values);assert record['n']==n
 avg=sum(values)/n if n else None
 variance=sum((v-avg)**2 for v in values)/(n-1) if n>1 else None
 equal(record['mean'],avg);equal(record['variance'],variance);equal(record['SD'],math.sqrt(variance) if variance is not None else None)
def key(row):
 j=row['job'];return(j['scenario'],j['difficulty'],j['world_seed'],j['policy_seed'],j['budget'])
def main():
 m=read(HERE/'manifest.json')
 for name,pin in m['files'].items():
  p=HERE/name;assert p.stat().st_size==pin['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
 d=read(HERE/'snapshot.json');rows=d['rows'];assert len(rows)==200
 index={}
 for r in rows:
  j=r['job'];k=key(r)+(j['arm'],);assert k not in index;index[k]=r
  assert r['state']=='accepted'and j['scenario']=='Proteomics'and j['difficulty']=='Normal'and j['world_seed']in(3,4)and j['policy_seed']in range(501,511)
  assert j['generation']==0 and j['model_key']=='qwen35_4b'
  assert 0<=r['audit']['execution_success_rate']<=1 and 0<=r['audit']['final_score_normalized']<=1
  assert isinstance(r['audit']['completed_successfully'],bool)
  assert re.fullmatch('[a-f0-9]{64}',r['source']['sha256'])
  assert r['audit']['agent_attempts']<=j['budget']
 arms=('Native','Common2','Public2','ActionPool2');metrics=('execution_success_rate','final_score_normalized','completed_successfully')
 for budget,block in d['budgets'].items():
  b=int(budget);sub=[r for r in rows if r['job']['budget']==b]
  for a in arms:assert block['accepted_by_arm'][a]==sum(r['job']['arm']==a for r in sub)
  keys=sorted({key(r)for r in sub});full=[k for k in keys if all(k+(a,)in index for a in arms)]
  assert block['four_arm_keys']==[list(k)for k in full]and block['complete_four_arm_cases']==len(full)
  if b in(20,30):assert len(full)==20
  else:assert len(full)==0
  for metric in metrics:
   for a in arms:stats(block['four_arm_stats'][metric][a],[index[k+(a,)]['audit'][metric]for k in full])
  for contrast,g in block['contrasts'].items():
   a,reference=contrast.split('-minus-');matched=[k for k in keys if k+(a,)in index and k+(reference,)in index];assert g['n_pairs']==len(matched)
   for metric,item in g['metrics'].items():
    method=[float(index[k+(a,)]['audit'][metric])for k in matched];ref=[float(index[k+(reference,)]['audit'][metric])for k in matched];delta=[x-y for x,y in zip(method,ref)]
    stats(item['method'],method);stats(item['reference'],ref);stats(item['difference'],delta)
    assert item['win_tie_loss']==[sum(x>1e-12 for x in delta),sum(abs(x)<=1e-12 for x in delta),sum(x<-1e-12 for x in delta)]
 assert all(not r['audit']['completed_successfully']for r in rows)
 assert d['queues']['dw_ablation_common_public_b20_b30_v1']['states']=={'accepted':80}
 assert d['queues']['dw_ablation_common_public_b10_v1']['states']=={'pending':40}
 assert d['queues']['dw_longbudget_b100_fourarms_v1']['states']=={'claimed':4,'pending':20}
 csv_rows=list(csv.DictReader((HERE/'per_trajectory.csv').open()));assert len(csv_rows)==200
 csvkeys=set()
 for r in csv_rows:
  k=('Proteomics','Normal',int(r['world_seed']),int(r['policy_seed']),int(r['budget']),r['arm']);assert k not in csvkeys;csvkeys.add(k);v=index[k]
  equal(float(r['action_success_rate']),v['audit']['execution_success_rate']);equal(float(r['task_progress']),v['audit']['final_score_normalized'])
  assert(r['task_success']=='True')==v['audit']['completed_successfully']and r['completion_sha256']==v['source']['sha256']
 forbidden=re.compile(b'/'+rb'(?:mnt|Users|root)/|GPU-[0-9a-fA-F]{8}-|Bearer\s+\S+|BEGIN[^\n]*PRIVATE KEY|"(?:hostname|password|cookie|access_token|raw_prompt|token_ids|transport)"\s*:')
 for name in m['files']:
  if name.endswith(('.json','.md','.csv')):assert not forbidden.search((HERE/name).read_bytes()),name
 print(json.dumps({'status':'passed','accepted_trajectories':200,'new_ablation_accepted':80,'B20_complete_four_arm_cases':20,'B30_complete_four_arm_cases':20,'all_means_sample_variances_SDs_contrasts_and_CSV_recomputed':True,'incomplete_queues_not_counted_as_complete':True,'sensitive_machine_paths_omitted':True,'new_scientific_calls':0},sort_keys=True))
if __name__=='__main__':main()
