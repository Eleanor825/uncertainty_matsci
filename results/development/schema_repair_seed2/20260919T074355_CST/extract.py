from pathlib import Path
from collections import Counter
import json,hashlib,time,math,importlib.util
D=Path('/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607/benchmark_extensions/made_controller_repair_20260919/development_validation')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def read(p):return json.loads(p.read_text())
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def action(a):
 if not isinstance(a,dict):return a
 args=a.get('arguments',{});safe={k:v for k,v in args.items() if k not in ['structure','frac_coords','species']};return {'tool':a.get('tool'),'arguments':safe,'large_geometry_fields_omitted':sorted(set(args)-set(safe))}
reg=read(D/'registration.json');mods={};sources={}
for arm in ['original_controller','schema_repaired_controller']:
 p=D/arm/'src/matdiscovery/failure_controller.py';r=next(x for x in reg['source_files'][arm] if x['path']==str(p));assert sha(p)==r['sha256'];sources[arm]=r;sp=importlib.util.spec_from_file_location('controller_'+arm,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);mods[arm]=m
out={'read_at':time.time(),'scope':'Only completed Al-Pd-Sm seed2 development pair','registration':ref(D/'registration.json'),'controller_sources':sources,'execution_profile':{k:v for k,v in reg['execution_profile'].items() if k not in ['artifacts']},'arms':[]}
for arm in mods:
 job=D/'jobs'/('dev-schema-Al-Pd-Sm-B50-seed2-'+arm);result=read(job/'result.json');claim=read(job/'claim.json');A=job/claim['attempt_id'];inventory={x['path']:x['sha256'] for x in result['artifacts']};rows=[];originals=[]
 for line_no,line in enumerate((A/'decisions.jsonl').open(),1):
  r=json.loads(line);originals.append(r);g=r['generation'];fc=r.get('failure_controller') or {};selection=r.get('failure_candidate_selection') or {};parts=r['local_decision_id'].split('-c');seq=int(parts[0][1:]);candidate=int(parts[1]);a=g.get('parsed_action');assert g['seed']==(2*10000019+seq*97+candidate)%2**63
  values={k:r.get(k) for k in ['decision_id','local_decision_id','prefix_hash','disposition','label_immediate_error','label_future_failure','graph_status','graph_error','generation_seconds','graph_seconds','predicted_failure_probability','failure_type_probabilities','observed_failure_types','failure_type_observation_rpc_ids','failure_types_used_for_control','tool_error']}
  values.update(line=line_no,sequence=seq,candidate_index=candidate,generation_success=g['success'],generation_failure_code=g['failure_code'],seed=g['seed'],action=action(a),action_sha256=fp(a),completion_tokens=g['completion_count'],prompt_tokens=g['prompt_token_count'],policy_stamp=r['policy_stamp'],graph_missing=(r.get('features') or {}).get('graph.graph_missing'),controller={k:fc.get(k) for k in ['threshold','trigger','trigger_source','request_retry','controller_action','local_schema_issues','triggered_heads','feedback_for_retry','reason']},feedback_received=r.get('feedback_received_for_this_candidate'),selection={'selected_index':selection.get('selected_index'),'ranking_rule':selection.get('ranking_rule'),'common_comparison_heads':selection.get('common_comparison_heads'),'type_signals_used':selection.get('type_signals_used'),'ranking':[{'candidate_index':x['candidate_index'],'valid':x['valid'],'rank_key':x['rank_key'],'compared_head_probabilities':x['compared_head_probabilities']} for x in selection.get('ranking',[])]},schema_replay={name:m._action_issues(a) for name,m in mods.items()})
  rows.append(values)
 assert sha(A/'decisions.jsonl')==inventory[str(A/'decisions.jsonl')]
 pending={};steps=[];tools=[];init=None;close=None;observes=0
 for line_no,line in enumerate((A/'rpc/rpc.jsonl').open(),1):
  e=json.loads(line);v=e.get('payload',{});i=v.get('id')
  if e.get('direction')=='request':pending[i]=(v,line_no)
  elif e.get('direction')=='response':
   request,rl=pending.pop(i);op=request['op'];res=v.get('result',{})
   if op=='init':init={'request':{k:x for k,x in request.get('args',{}).items() if k in ['elements','seed','budget','stability_tolerance','orb_device','mace_num_workers']},'metadata':res.get('metadata'),'counts':res.get('observation',{}).get('counts')}
   if op=='tool':
    args=request['args'];output=res.get('output');summary={}
    if isinstance(output,dict):
     summary={k:x for k,x in output.items() if k not in ['records','entries','structure','structures','results'] and isinstance(x,(str,int,float,bool,type(None)))}
     if isinstance(output.get('records'),list):summary['records']=[{k:x for k,x in rec.items() if k in ('accepted','reason','hash','composition','structure_index')} for rec in output['records']]
    elif isinstance(output,str):summary={'text':output[:500]}
    elif isinstance(output,list):summary={'list_length':len(output)}
    tools.append({'rpc_id':i,'request_line':rl,'response_line':line_no,'time':e.get('time'),'ok':v['ok'],'error':v.get('error'),'action':action({'tool':args['name'],'arguments':args.get('arguments',{})}),'output_summary':summary,'counts':res.get('status',{}).get('counts')})
   if op=='step':
    o=res.get('official_observation',{});metrics=res.get('official_metrics',{});steps.append({'step':len(steps)+1,'rpc_id':i,'response_line':line_no,'time':e.get('time'),'ok':v['ok'],'error':v.get('error'),'candidate_hash':res.get('event',{}).get('candidate_hash'),'official':{k:x for k,x in o.items() if not isinstance(x,(dict,list))},'metrics':{k:x for k,x in metrics.items() if k in ['num_newly_discovered_structures','num_newly_discovered_stable','queries_used'] or k.startswith('novelty_') or k in ['diversity_all_structure_unique_structure_count','diversity_all_composition_unique_composition_count']}})
   if op=='observe':observes+=1
   if op=='close':close={'ok':v['ok'],'line':line_no,'rpc_id':i,'time':e.get('time')}
 assert not pending and len(steps)==50 and close['ok'] and sha(A/'rpc/rpc.jsonl')==inventory[str(A/'rpc/rpc.jsonl')]
 events=[dict(json.loads(l),line=i) for i,l in enumerate((A/'decision_events.jsonl').open(),1)];fixed=[e for e in events if e['event']=='fixed_failure_recovery']
 for e in fixed:e['action']=action(e['action'])
 journal=A/'environment/oracle_attempts.jsonl';jev=[json.loads(l) for l in journal.open()];jcounts=Counter(e.get('event') for e in jev)
 small_examples=[{k:v for k,v in e.items() if not isinstance(v,(dict,list))} for e in jev[:6]]
 ep=result['episodes'][0];out['arms'].append({'arm':arm,'result':ref(job/'result.json'),'job':result['job'],'claim':ref(job/'claim.json'),'data_sources':{p.name:ref(p) for p in [A/'decisions.jsonl',A/'decision_events.jsonl',A/'rpc/rpc.jsonl',journal]},'initialization':init,'costs':result['scientific_evidence']['costs'],'episode':ep,'rows':rows,'tools':tools,'steps':steps,'fixed_recovery':fixed,'close':close,'observe_rpc_count':observes,'journal_event_counts':dict(jcounts),'journal_first_schema_examples':small_examples})
print(json.dumps(out,allow_nan=False),flush=True)
