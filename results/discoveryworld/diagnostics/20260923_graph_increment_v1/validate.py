"""Recompute all published CSV metrics and label joins without importing diagnostic code."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math
HERE=Path(__file__).resolve().parent
FAMILIES=('Internal','NoGraph','GraphOnly')
def need(ok,message):
 if not ok:raise ValueError(message)
def metric(ys,ps):
 if not ys:return {'n':0,'positive':0,'negative':0,'brier':None,'logloss':None,'auroc':None}
 n=len(ys);need(n==len(ps)and all(y in(0,1)for y in ys)and all(math.isfinite(p)and 0<=p<=1 for p in ps),'Invalid metric input')
 ranked=sorted(zip(ps,ys));rank_sum=0.;i=0
 while i<n:
  j=i+1
  while j<n and ranked[j][0]==ranked[i][0]:j+=1
  rank_sum+=(i+1+j)/2*sum(y for _,y in ranked[i:j]);i=j
 positive=sum(ys);negative=n-positive
 return {'n':n,'positive':positive,'negative':negative,'brier':math.fsum((p-y)**2 for y,p in zip(ys,ps))/n,
  'logloss':-math.fsum(math.log(max(1e-7,min(1-1e-7,p if y else 1-p)))for y,p in zip(ys,ps))/n,
  'auroc':(rank_sum-positive*(positive+1)/2)/(positive*negative)if positive and negative else None}
def equal(a,b):return a is None and b is None or a is not None and b is not None and abs(a-b)<=1e-12

def run():
 manifest=json.loads((HERE/'manifest.json').read_text())
 for name,pin in manifest['files'].items():
  b=(HERE/name).read_bytes();need(hashlib.sha256(b).hexdigest()==pin['sha256']and len(b)==pin['bytes'],'Changed publication file '+name)
 with(HERE/'candidate_risks_and_executed_labels.csv').open(newline='')as f:rows=list(csv.DictReader(f))
 with(HERE/'same_candidate_decisions.csv').open(newline='')as f:decisions=list(csv.DictReader(f))
 results=json.loads((HERE/'results.json').read_text());source=json.loads((HERE/'source_audit.json').read_text())
 need(source['status']=='independent_source_and_label_audit_passed','Missing independent source-label verification')
 need(len(rows)==110 and len(decisions)==90,'Raw row coverage differs')
 labels={(r['condition'],r['attempt']):r for r in source['labels']};need(len(labels)==90,'Duplicate or missing source labels')
 seen=set();groups=defaultdict(list);checks=0;max_error=0.
 for row in rows:
  key=(row['condition'],int(row['attempt']),int(row['proposal']));need(key not in seen,'Duplicate candidate');seen.add(key);groups[row['condition']].append(row)
  need(row['executed']in('True','False')and row['common_supported']in('True','False'),'Invalid Boolean cell')
  if row['executed']=='False':need(row['official_failure']=='','Unexecuted candidate label was fabricated')
  if row['common_supported']=='False':need(all(row['risk_'+f]==''for f in FAMILIES),'Unsupported risk was imputed')
 for cohort in results['cohorts']:
  name=cohort['condition'];items=groups[name];executed=[r for r in items if r['executed']=='True'];supported=[r for r in executed if r['common_supported']=='True']
  need(len(executed)==30 and len({r['attempt']for r in executed})==30,'Executed action denominator differs')
  need(len({r['policy_state_id']for r in items})==1 and {r['parameter_stratum']for r in items}=={cohort['parameter_stratum']},'Policy strata were mixed')
  for row in executed:
   original=labels[name,int(row['attempt'])];need(int(row['proposal'])==original['selected_proposal']and int(row['official_failure'])==original['official_failure']and not original['repaired'],'CSV/source action label differs')
  cov=cohort['coverage'];need(cov['all_agent_requests']==cov['official_action_labels']==30 and cov['common_supported_executed']==len(supported)and cov['unsupported_executed']==30-len(supported),'Coverage differs')
  models={f:metric([int(r['official_failure'])for r in supported],[float(r['risk_'+f])for r in supported])for f in FAMILIES}
  for family,measured in models.items():
   for key,x in measured.items():
    expected=cohort['metrics'][family][key];need(equal(x,expected),'Metric mismatch');checks+=1
    if x is not None:max_error=max(max_error,abs(x-expected))
  for a,b in (('Internal','NoGraph'),('GraphOnly','NoGraph'),('Internal','GraphOnly')):
   for key in ('brier','logloss','auroc'):
    x=models[a][key];y=models[b][key];delta=None if x is None or y is None else x-y
    need(equal(delta,cohort['paired_differences'][a+'_minus_'+b][key]),'Paired difference mismatch');checks+=1
  attempts=defaultdict(list)
  for row in items:attempts[int(row['attempt'])].append(row)
  choice_rows=[];pair_count=0
  for attempt,candidates in attempts.items():
   candidates.sort(key=lambda r:int(r['proposal']));d=next(x for x in decisions if x['condition']==name and int(x['attempt'])==attempt)
   parsed=[r for r in candidates if r['parsed_success']=='True'];pair=len(parsed)==2 and all(r['common_supported']=='True'for r in parsed);pair_count+=int(pair)
   actual=next(r for r in candidates if r['executed']=='True');need(int(d['recorded_selected'])==int(actual['proposal']),'Recorded selection differs')
   choices={}
   for family in FAMILIES:
    choice=min(parsed,key=lambda r:(float(r['risk_'+family]),int(r['proposal'])))if pair else parsed[0]if parsed else candidates[0]
    choices[family]=int(choice['proposal']);need(int(d[family])==choices[family],'Shadow choice mismatch')
   need((d['supported_two']=='True')==pair,'Supported ranking denominator differs');choice_rows.append(choices)
  recorded=cohort['same_candidate_selection'];need(pair_count==recorded['common_supported_two_candidate_attempts'],'Pair count mismatch')
  for a,b in (('Internal','NoGraph'),('GraphOnly','NoGraph')):need(sum(r[a]!=r[b]for r in choice_rows)==recorded[a+'_'+b+'_disagreements'],'Disagreement count mismatch')
 need(sum(r['executed']=='True'for r in rows)==90 and sum(r['executed']=='False'for r in rows)==20,'Outcome accounting differs')
 return {'status':'passed','candidate_rows':110,'executed_labels_checked_against_source':90,'unexecuted_candidates_without_labels':20,'metrics_and_paired_differences':checks,'maximum_absolute_metric_error':max_error,'source_files_verified_remotely':source['verified_source_files'],'new_model_or_environment_calls':0}
if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
