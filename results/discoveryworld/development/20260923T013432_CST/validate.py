"""Stdlib-only checks of four closed trajectories and separate fixed-TRAIN diagnostics."""
from pathlib import Path
import csv,hashlib,json,math,re,statistics
P=Path(__file__).resolve().parent

def read(name):return json.loads((P/name).read_text())
def require(ok,msg):
 if not ok:raise ValueError(msg)
def near(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
def main():
 c=read('cohort.json');require((c['world_seed'],c['policy_seed'],c['budget'])==(2,307,30),'Wrong cohort')
 require(c['Native1_present']is False and c['preselected_internal_variant']=='combined'and c['base84_promoted_after_results']is False,'Wrong comparator/selection claim')
 expected={'Common2':(0,7,0,0),'Internal_base84':(.125,5,14,10),'Internal_combined':(.125,21,13,3),'OutputAction_combined':(0,26,16,3)}
 require({a['condition']for a in c['arms']}==set(expected),'Missing/extra arm')
 with(P/'metrics.csv').open(newline='')as f:metrics={r['condition']:r for r in csv.DictReader(f)}
 with(P/'actions.csv').open(newline='')as f:actions=list(csv.DictReader(f))
 names=['base84/Internal','combined/Internal','combined/OutputAction'];computed_shadow=[];identical={n:[]for n in names};positive={};graph_ok=graph_missing=0
 for a in c['arms']:
  name=a['condition'];rows=a['selection_rows'];closed=a['closed_metrics'];progress=closed['runner_progress_audit']['observed_changes'];want=expected[name]
  require(a['status']=='closed_accepted_episode'and len(rows)==len(progress)==30 and closed['closed_RPC_verified'],'Unclosed/incomplete episode')
  require(closed['task_success']is False and near(closed['official_score'],want[0])and closed['failure_count']==want[1],'Wrong terminal outcome')
  require(sum(x['official_or_parser_failure']for x in rows)==want[1],'Failure count differs from actual actions')
  require(sum(x['NN_rankchange']for x in rows)==want[2]and sum(x['NN_rankchange_changed_action']for x in rows)==want[3],'Rank/packet changes differ')
  require(a['selection_counts']['NN_rankchange']==want[2]and a['selection_counts']['NN_rankchange_changed_action']==want[3],'Summary differs')
  require(float(metrics[name]['score'])==want[0]and int(metrics[name]['failures'])==want[1],'CSV terminal mismatch')
  prefix=closed['runner_B10_B30_prefix_audit'];require(prefix['B10_adaptive_stop_or_selection']is False,'Adaptive B10 selection claim')
  graph_ok+=a['graph_status_counts']['succeeded'];graph_missing+=a['graph_status_counts']['unavailable'];positive[name]=[]
  for i,(row,p)in enumerate(zip(rows,progress),1):
   require(row['attempt_index']==p['attempt_index']==i and row['second_proposal_requested']is True,'Wrong two-proposal/step grid')
   require(row['official_executed_outcome_available']and row['no_counterfactual_failure_labels'],'Outcome/label issue')
   require(row['selected_action']==row['candidate_actions'][row['selected_index']],'Selected action identity differs')
   packet=row['selected_action']!=row['candidate_actions'][row['common_hash_index']]
   require(row['NN_rankchange_changed_action']==bool(row['NN_rankchange']and packet),'Index change confused with packet change')
   require(near(p['score_after']-p['score_before'],p['delta_since_previous_observation']),'Bad score delta')
   if i>1:require(near(p['score_before'],progress[i-2]['score_after']),'Broken observed score chain')
   if p['delta_since_previous_observation']>0:positive[name].append(i)
   csvrow=next(x for x in actions if x['condition']==name and int(x['attempt'])==i)
   require(json.loads(csvrow['selected_action'])==row['selected_action']and near(float(csvrow['score_delta']),p['delta_since_previous_observation']),'Action CSV mismatch')
   vals=[row['recorded_shadow_risks'].get(n,[None,None])[row['selected_index']]for n in names]
   if all(type(v)in(int,float)and math.isfinite(v)for v in vals):computed_shadow.append((name,i,int(row['official_or_parser_failure']),vals))
   if row['candidate_actions'][0]is not None and row['candidate_actions'][0]==row['candidate_actions'][1]:
    for n in names:
     pair=row['recorded_shadow_risks'].get(n)
     if pair and all(type(v)in(int,float)and math.isfinite(v)for v in pair):identical[n].append(abs(pair[0]-pair[1]))
  require(near(progress[-1]['score_after'],closed['official_score']),'Final score mismatch')
 require(positive=={'Common2':[],'Internal_base84':[3],'Internal_combined':[25],'OutputAction_combined':[]},'Unexpected progress steps')
 require(graph_ok==229 and graph_missing==11,'Candidate graph inventory differs')
 arms={a['condition']:a for a in c['arms']};b=arms['Internal_base84']['selection_rows'];z=arms['Internal_combined']['selection_rows']
 require(b[0]['NN_rankchange_changed_action']and b[0]['official_or_parser_failure'],'Initial base84 divergence changed')
 require(b[2]['candidate_actions'][0]==b[2]['candidate_actions'][1]and not b[2]['NN_rankchange_changed_action'],'Base84 scoring packet changed')
 require(z[24]['fallback_to_common_hash']and not z[24]['NN_rankchange_changed_action']and z[24]['candidate_graph_available']==[True,False],'Combined scoring fallback changed')
 objects=read('public_object_identity.json')['arms'];require(len({x['initial_public_observation_sha256']for x in objects})==1,'Initial public observations differ')
 for x in objects:
  require(all(p['object']['name']=='proteomics meter'and p['object']['uuid']==33276 for p in x['object_33276_public_identity']),'Wrong item identity')
  if x['condition']in ('Internal_base84','Internal_combined'):
   step=3 if x['condition']=='Internal_base84' else 25
   require(any(p['attempt']==step and 'on agent' in p['object']['description']for p in x['object_33276_public_identity']),'Missing meter-in-inventory proof')
 shadow=read('shadow_diagnostic.json');require(len(computed_shadow)==113 and sum(x[2]for x in computed_shadow)==57,'Wrong complete-case rows')
 with(P/'selected_shadow_rows.csv').open(newline='')as f:srows=list(csv.DictReader(f))
 require([(x['condition'],int(x['attempt']))for x in srows]==[(x[0],x[1])for x in computed_shadow],'Shadow row identities differ')
 for j,n in enumerate(names):
  pos=[x[3][j]for x in computed_shadow if x[2]];neg=[x[3][j]for x in computed_shadow if not x[2]]
  auc=sum((p>q)+.5*(p==q)for p in pos for q in neg)/(len(pos)*len(neg));brier=statistics.mean((x[3][j]-x[2])**2 for x in computed_shadow);m=shadow['matched_selected_action_metrics'][n]
  require(near(auc,m['AUROC_descriptive'])and near(brier,m['Brier']),'Shadow recomputation differs')
  require(len(identical[n])==58 and near(statistics.mean(identical[n]),shadow['same_parsed_packet_risk_variation'][n]['mean_absolute_gap']),'Identical packet gap mismatch')
 num=read('numerical_path_diagnostic.json');v=num['completion'];require(v['status']=='complete_numerical_measurements'and v['qualified']is False and v['no_main_admission']is True and num['resource_closed'],'Wrong numerical outcome')
 require(v['counts']['model_forward_returns']==15 and v['counts']['backward_returns']==2 and v['counts']['optimizer_updates']==v['counts']['environment_calls']==v['counts']['generations']==0,'Numerical cost mismatch')
 require(num['weight_integrity']['unchanged']and num['gradient_comparison']['passed']and num['gradient_comparison']['global_error_L2']==0,'Gradient/weight parity mismatch')
 cache=v['teacher_forcing_vs_forced_cache'];require(cache['argmax_equal_count']==cache['token_count']==13 and cache['raw_logits']['passed']is False and cache['target_logprobs']['passed']is False,'Strict cache mismatch lost')
 diag=read('action_influence_diagnostic.json');require(diag['completion']['successful_graphs']==3 and diag['qualification']['prefix_results_passed']==8,'Wrong selector graph/qualification count')
 require(diag['completion']['NN_calls']==diag['completion']['new_environment_calls']==diag['completion']['new_policy_generations']==0,'Unexpected selector science scope')
 require(diag['selector_costs']['actual_extra_backward_returns']==11,'Extra selection cost missing')
 for row in diag['three_old_new_graphs']:
  require(row['original_source_prefix_model_bank_match']is True and row['new']['selected']['count']==32 and row['new']['total_backward_calls_including_selector']==34,'Wrong selector budget or source identity')
  require(row['old']['TC_devices']==row['new']['TC_devices']==['cpu']and row['new']['constant_storage_policy']=='cpu','Unregistered residency change')
 manifest=read('manifest.json')
 for name,item in manifest['files'].items():
  p=P/name;require(p.is_file()and not p.is_symlink()and p.stat().st_size==item['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Manifest mismatch: '+name)
 for p in P.rglob('*'):
  if p.is_file()and p.suffix in ('.json','.csv','.py','.md'):
   text=p.read_text();bad=['/'+ 'Users'+'/', '/'+ 'mnt'+'/',r'tj-\d+-t-\d+',r'GPU-[0-9a-f]{8}-[0-9a-f-]{20,}',r'Bearer\s+[A-Za-z0-9._-]{12,}']
   require(not any(re.search(s,text)for s in bad),'Private material in '+p.name)
 print(json.dumps({'passed':True,'closed_arms':4,'executed_actions':120,'task_successes':0,'meter_pickup_score_transitions':2,'graphs_succeeded':229,'graphs_unavailable_preserved':11,'same_selected_shadow_rows':113,'identical_packet_pairs':58,'new_scientific_calls_for_export':0,'new_Native_or_heldout_comparator':False,'combined_remains_preselected':True,'files_hash_checked':len(manifest['files'])},sort_keys=True))
if __name__=='__main__':main()
