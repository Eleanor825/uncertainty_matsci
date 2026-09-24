"""Validate public aggregates and exact case joins; never load private runtime data."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,math,re
HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/name).read_bytes())
def csvrows(name):return list(csv.DictReader((HERE/name).open()))
def number(v):return None if v in(None,'')else float(v)
def equal(a,b):
 if b is None:assert a is None,(a,b)
 else:assert a is not None and math.isclose(float(a),float(b),abs_tol=1e-11,rel_tol=1e-11),(a,b)
def avg(v):return sum(v)/len(v)if v else None
def auc(y,p):
 positive=[x for x,z in zip(p,y)if z==1];negative=[x for x,z in zip(p,y)if z==0]
 if not positive or not negative:return None
 return sum(1 if a>b else .5 if a==b else 0 for a in positive for b in negative)/(len(positive)*len(negative))
def metrics(rs):
 y=[int(r['label'])for r in rs];p=[float(r['probability'])for r in rs]
 return {'n':len(rs),'failures':sum(y),'failure_rate':avg(y),'Brier':avg([(a-b)**2 for a,b in zip(y,p)]),'AUROC':auc(y,p),'NLL':avg([-a*math.log(max(min(b,1-1e-12),1e-12))-(1-a)*math.log(max(min(1-b,1-1e-12),1e-12))for a,b in zip(y,p)])}
def percentile(xs,q):
 xs=sorted(xs);x=(len(xs)-1)*q;i=int(x);j=min(i+1,len(xs)-1);return xs[i]+(xs[j]-xs[i])*(x-i)
def main():
 manifest=read('manifest.json');assert manifest['schema']=='fixed_publication_manifest_v1'
 for name,pin in manifest['files'].items():
  p=HERE/name;assert p.parent==HERE and p.is_file();assert p.stat().st_size==pin['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
 forbidden=re.compile(b'/'+rb'(?:mnt|Users|root)/|GPU-[0-9a-fA-F]{8}-|Bearer\s+\S+|BEGIN[^\n]*PRIVATE KEY|mioffice\.cn|"(?:hostname|password|cookie|access_token|refresh_token|authorization|raw_prompt|token_ids|input_ids|transport|gpu_uuid|node_id)"\s*:')
 for name in manifest['files']:
  if name.endswith(('.json','.md','.csv')):assert not forbidden.search((HERE/name).read_bytes()),name
 p1=read('p1_snapshot.json');trajectory=csvrows('p1_per_trajectory.csv');assert len(trajectory)==584;assert sum(r['status']=='accepted'for r in trajectory)==468
 assert Counter(r['cohort']for r in trajectory)=={'MADE':240,'DW_Proteomics':264,'DW_Easy':80}
 index={}
 for r in trajectory:
  k=(r['cohort'],r['task'],r['difficulty'],r['world'],int(r['budget']),int(r['policy_seed']));assert(k,r['arm'])not in index;index[(k,r['arm'])]=r
  if r['status']=='accepted':assert re.fullmatch('[a-f0-9]{64}',r['completion_sha256'])
 for s in p1['status']:
  rs=[r for r in trajectory if(r['cohort'],int(r['budget']),r['arm'])==(s['cohort'],s['budget'],s['arm'])];assert dict(Counter(r['status']for r in rs))==s['states']
 for c in p1['comparisons']:
  keys=[k for k,a in index if a==c['method']and(k[0],k[4])==(c['cohort'],c['budget'])];pairs=[]
  for k in keys:
   a=index[(k,c['method'])];b=index.get((k,c['reference']))
   if a['status']!='accepted'or not b or b['status']!='accepted':continue
   av,bv=number(a.get(c['metric'])),number(b.get(c['metric']))
   if av is not None and bv is not None:pairs.append((av,bv))
  av=[a for a,b in pairs];bv=[b for a,b in pairs];d=[a-b for a,b in pairs];assert len(pairs)==c['n_pairs'];equal(c['mean_method'],avg(av));equal(c['mean_reference'],avg(bv));equal(c['difference'],avg(d));assert(c['wins'],c['ties'],c['losses'])==(sum(x>1e-12 for x in d),sum(abs(x)<=1e-12 for x in d),sum(x< -1e-12 for x in d))
  if c['interval95']is not None:assert len(c['interval95'])==2 and c['interval95'][0]<=c['interval95'][1]
 arms=('Native','Common2','Public2','ActionPool2')
 for c in p1['complete_four_arm_comparisons']:
  keys={k for k,a in index if(k[0],k[4])==(c['cohort'],c['budget'])};full=[k for k in keys if all((k,a)in index and index[k,a]['status']=='accepted'for a in arms)];assert len(full)==c['complete_four_arm_cases']
  for metric,value in c.items():
   if metric in('cohort','budget','arm','complete_four_arm_cases'):continue
   vs=[number(index[k,c['arm']].get(metric))for k in full];vs=[x for x in vs if x is not None];equal(value,avg(vs))
 for r in p1['mechanism']:
  for numerator,denominator,metric in(('supported_candidates','support_assessed_candidates','support_fraction'),('duplicate_two_valid','two_valid_decisions','duplicate_fraction_among_two_valid'),('actual_packet_changes','actual_packet_change_denominator','actual_packet_change_fraction'),('risk_used_decisions','decisions','risk_used_fraction')):
   equal(r[metric],r[numerator]/r[denominator]if r[denominator]else None)
 predictions=csvrows('p1_representation_predictions.csv');assert len(predictions)==280
 perms=csvrows('p1_representation_permutations.csv');assert len(perms)==200
 for bench,r in p1['representations'].items():
  for arm,summary in r['baselines_saved_predictions'].items():
   rs=[x for x in predictions if(x['benchmark'],x['arm'])==(bench,arm)];m=metrics(rs)
   for k,value in m.items():equal(summary['overall'][k],value)
   for group,gm in summary['by_group'].items():
    actual=metrics([x for x in rs if x['stratum']==group])
    for k,value in actual.items():equal(gm[k],value)
  for metric,value in r['PC_minus_P']['method_minus_reference'].items():equal(value,r['baselines_saved_predictions']['P+C-hidden']['overall'][metric]-r['baselines_saved_predictions']['P']['overall'][metric])
  ps=[x for x in perms if x['benchmark']==bench];assert len(ps)==100 and {int(x['permutation'])for x in ps}==set(range(100));bs=[float(x['Brier'])for x in ps];equal(r['shuffle']['Brier_mean'],avg(bs));equal(r['shuffle']['Brier_range95_permutation_distribution'][0],percentile(bs,.025));equal(r['shuffle']['Brier_range95_permutation_distribution'][1],percentile(bs,.975))
 p2=read('p2_snapshot.json');cs=p2['cases'];assert len(cs)==8 and p2['fixed_control_cells']==8 and p2['fixed_comparator_cells']==32
 assert {(c['task_id'],c['budget'],c['policy_seed'])for c in cs}=={(t,b,s)for t in('Au-K-Tb','Mg-Sn-Sr')for b in(10,20)for s in(501,502)}
 for arm,counts in p2['status_by_arm'].items():assert dict(Counter(c['arms'][arm]['state']for c in cs))==counts
 accepted=[c for c in cs if c['arms']['RiskShuffle2']['accepted']];assert len(accepted)==2
 first=accepted[0];assert(first['task_id'],first['budget'],first['policy_seed'])==('Au-K-Tb',10,501)
 expected={'Native1':(0,0),'Common2':(3,.39),'Public2':(1,.17),'ActionPool2':(1,.17),'RiskShuffle2':(1,.09)}
 for arm,(sun,audc)in expected.items():equal(first['arms'][arm]['metrics']['SUN'],sun);equal(first['arms'][arm]['metrics']['AUDC'],audc)
 exp=first['arms']['RiskShuffle2']['exposure'];assert(exp['decisions'],exp['eligible_two_complete_risks'],exp['nonidentity_assignments'],exp['actual_packet_changed_by_permutation'],exp['risk_ties'],exp['same_action_packets'])==(15,9,6,6,0,0)
 second=next(c for c in accepted if c['task_id']=='Mg-Sn-Sr');assert second['budget']==10 and second['policy_seed']==501
 for arm,(sun,audc)in {'Native1':(8,.86),'Common2':(9,.93),'ActionPool2':(9,.95),'RiskShuffle2':(10,1.)}.items():equal(second['arms'][arm]['metrics']['SUN'],sun);equal(second['arms'][arm]['metrics']['AUDC'],audc)
 assert second['arms']['Public2']['metrics']is None and second['arms']['Public2']['state']=='failed_preserved'
 exp2=second['arms']['RiskShuffle2']['exposure'];assert(exp2['decisions'],exp2['eligible_two_complete_risks'],exp2['nonidentity_assignments'],exp2['actual_packet_changed_by_permutation'],exp2['risk_ties'],exp2['same_action_packets'])==(23,10,5,3,0,6)
 assert p2['status_by_arm']['RiskShuffle2']=={'accepted':2,'claimed_unknown_until_reconciled':1,'unclaimed':5}
 for c in p2['comparisons']+p2['ActionPool2_minus_Public2_fixed_case_baseline']:
  matched=[x for x in cs if x['budget']==c['budget']and all(x['arms'][arm]['accepted']for arm in(c['method'],c['reference']))];assert len(matched)==c['complete_pairs']and c['expected_fixed_cells']==4
  for metric,r in c['metrics'].items():
   av=[x['arms'][c['method']]['metrics'][metric]for x in matched];bv=[x['arms'][c['reference']]['metrics'][metric]for x in matched];d=[a-b for a,b in zip(av,bv)];equal(r['method_mean'],avg(av));equal(r['reference_mean'],avg(bv));equal(r['mean_paired_difference'],avg(d));assert(r['wins'],r['ties'],r['losses'])==(sum(x>1e-12 for x in d),sum(abs(x)<=1e-12 for x in d),sum(x< -1e-12 for x in d))
  for metric,r in c['costs'].items():
   ps=[(x['arms'][c['method']]['costs'].get(metric),x['arms'][c['reference']]['costs'].get(metric))for x in matched];ps=[(a,b)for a,b in ps if a is not None and b is not None];av=[a for a,b in ps];bv=[b for a,b in ps];assert len(ps)==r['paired_cost_rows'];equal(r['method_mean'],avg(av));equal(r['reference_mean'],avg(bv));equal(r['mean_paired_difference'],avg([a-b for a,b in ps]));equal(r['observed_mean_ratio'],avg(av)/avg(bv)if avg(bv)else None)
 assert len(csvrows('p2_every_case.csv'))==40
 assert p1['limitations']['controlled_identical_hardware_load_and_timing']is False
 assert p2['interpretation']['unknown_outcomes_imputed']is False
 print(json.dumps({'status':'passed','P1_registered_cells':584,'P1_accepted_trajectories':468,'P2_fixed_controls':8,'P2_accepted_controls':2,'P2_case_rows':40,'task_and_cost_means_contrasts_counts_recomputed':True,'original_offline_prediction_metrics_recomputed':True,'bootstrap_intervals_preserved_source_bound':True,'permutation_summary_recomputed_from_published_iteration_rows':True,'private_paths_hosts_auth_and_raw_tokens_omitted':True,'aggregate_token_counts_retained':True,'new_scientific_calls':0},sort_keys=True))
if __name__=='__main__':main()
