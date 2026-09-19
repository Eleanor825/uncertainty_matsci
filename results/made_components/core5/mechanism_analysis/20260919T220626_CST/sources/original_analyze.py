"""All60 saved traces only. No ML/model/graph/oracle import or counterfactual outcome imputation."""
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone,timedelta
import csv,gzip,hashlib,io,json,math,statistics

HERE=Path(__file__).resolve().parent;L=HERE.parents[1]
INPUT=L/'technical_not_main/made_core60_mechanism_actual_20260919_v1'
DATA_SHA='6430dde5138feb880868d63744522e475bc8f18f3d7e59cce118e4c830519a05'
SUMMARY_SHA='7e46c8a22911f38cab7670d48440e71874b3f95bb86b52ca9e5906cf7beb1ae7'
ARMS=('baseline_reference','es_only_independent','uq_only_support_aware','full_support_aware')
SYSTEMS=('Al-Li-V','Al-V-Zn','Au-K-Tb','Co-Dy-W','Co-Mg-Na')
TOOLS=('generate_structures','score_buffer','query_structures','select_for_evaluation','get_buffer_stats','list_compositions','create_structure')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name,value):
    raw=value.encode() if isinstance(value,str) else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();p=HERE/name
    if p.exists():assert p.read_bytes()==raw,name
    else:
        with p.open('xb') as f:f.write(raw)

def csvsave(name,rows):
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    save(name,out.getvalue())

def mean(v):return math.fsum(v)/len(v) if v else None

def error(value):return (value or {}).get('message')

def compact_action(d):
    a=d.get('action') or {};args=a.get('arguments') or {}
    return {'tool':a.get('tool'),'arguments':{k:v for k,v in args.items() if k not in ('structure','frac_coords','species')},'action_sha256':d.get('action_sha256')}


def aggregate(rows,**key):
    out=dict(key,trajectories=len(rows),SUN_sum=sum(r['SUN'] for r in rows),SUN_mean=mean([r['SUN'] for r in rows]),
        AUDC_mean=mean([r['AUDC'] for r in rows]),SUN_sample_variance=statistics.variance([r['SUN'] for r in rows]) if len(rows)>1 else None,
        AUDC_sample_variance=statistics.variance([r['AUDC'] for r in rows]) if len(rows)>1 else None)
    additive=['proposals','executed_tools','physical_steps','initialization_ORB','MACE_events','MACE_count_field_missing','failed_selects','tool_failures',
        'graph_succeeded','graph_unavailable','invalid_generation','controller_schema_retry_requests','controller_neural_retry_requests','actual_second_proposals',
        'actual_second_schema','actual_second_neural','selected_second','selected_second_after_schema','multiple_valid_risk_selection_changed',
        'supported_scalar_ge_threshold','unexecuted_neural_second','neural_second_generation_seconds','neural_second_graph_seconds','neural_second_prompt_tokens','neural_second_completion_tokens']+[t+'_executed' for t in TOOLS]
    for k in additive:out[k]=math.fsum(r[k] for r in rows) if 'seconds' in k else sum(r[k] for r in rows)
    out['scalar_NN_n']=sum(r['scalar_NN_n'] for r in rows)
    out['scalar_NN_min']=min((r['scalar_NN_min'] for r in rows if r['scalar_NN_min'] is not None),default=None)
    out['scalar_NN_max']=max((r['scalar_NN_max'] for r in rows if r['scalar_NN_max'] is not None),default=None)
    return out


def main():
    p=INPUT/'trajectories.jsonl.gz';assert sha(p)==DATA_SHA and sha(INPUT/'summary.json')==SUMMARY_SHA
    source=json.loads((INPUT/'summary.json').read_text());assert source['complete'] and source['gaps']==[]
    runs=[json.loads(line) for line in gzip.decompress(p.read_bytes()).splitlines()]
    assert {(x['job']['arm'],x['job']['task_id'],x['job']['seed']) for x in runs}=={(a,t,s) for a in ARMS for t in SYSTEMS for s in (2,3,4)}
    trajectories=[];seqrows=[];decisions=[];outcomes=[];failures=[];neural_extra=[];retry_feedback=[];typed_requests=[];provenance=[]
    for x in runs:
        j=x['job'];jid=j['job_id'];arm=j['arm'];ds=x['decisions'];steps=x['steps'];groups=defaultdict(list);threshold=x['rollout_settings']['risk_threshold']
        assert j['budget']==len(steps)==10 and x['new_model_or_oracle_calls']==0 and x['unexecuted_outcomes_imputed'] is False
        assert abs((2*sum(s['recorded_curve_SUN_after'] for s in steps)-steps[-1]['recorded_curve_SUN_after'])/100-j['AUDC'])<1e-14
        assert steps[-1]['recorded_curve_SUN_after']==j['SUN']
        byid={d['decision_id']:d for d in ds};assert len(byid)==len(ds)
        aline={a['decision_id']:a for a in x['execution_alignment'] if not a['fixed_recovery']}
        assert len(aline)==x['counts']['executed_policy_tools'] and x['counts']['fixed_recovery_tools']==0
        for d in ds:groups[d['sequence']].append(d)
        assert len(groups)==x['counts']['execution_sequences']
        record=dict(j,proposals=len(ds),executed_tools=len(aline),physical_steps=len(steps),
            initialization_ORB=x['costs']['initialization_oracle_attempts'],MACE_count_field=x['costs'].get('surrogate_oracle_attempts'),
            MACE_count_field_missing=int('surrogate_oracle_attempts' not in x['costs']),
            MACE_events=sum(e.get('role')=='mace' and e['kind']=='oracle_evaluation' for e in x['environment_events']),
            wall_seconds=x['costs']['wall_seconds'],graph_seconds=x['costs']['graph_seconds'],
            llm_calls=x['costs']['llm_calls'],prompt_tokens=x['costs']['prompt_tokens'],completion_tokens=x['costs']['completion_tokens'],
            failed_selects=0,tool_failures=0,graph_succeeded=sum(d['graph_status']=='succeeded' for d in ds),
            graph_unavailable=sum(d['graph_status']=='unavailable' for d in ds),invalid_generation=sum(not d['generation_success'] for d in ds),
            controller_schema_retry_requests=0,controller_neural_retry_requests=0,actual_second_proposals=0,actual_second_schema=0,actual_second_neural=0,
            selected_second=0,selected_second_after_schema=0,multiple_valid_risk_selection_changed=0,supported_scalar_ge_threshold=0,
            unexecuted_neural_second=0,neural_second_generation_seconds=0.,neural_second_graph_seconds=0.,neural_second_prompt_tokens=0,neural_second_completion_tokens=0)
        for t in TOOLS:record[t+'_executed']=sum(a['action']['tool']==t for a in aline.values())
        if record['MACE_count_field'] is not None:assert record['MACE_count_field']==record['MACE_events']
        vals=[d['raw_scalar_NN_if_recorded'] for d in ds if d['generation_success'] and d['raw_scalar_NN_if_recorded'] is not None]
        record.update(scalar_NN_n=len(vals),scalar_NN_min=min(vals,default=None),scalar_NN_max=max(vals,default=None),
            model_state_hash=x['profile']['actual_model_state_hash'],selected_generation=x['profile'].get('selected_generation',0),
            direct_stable_new_events=sum(bool(s['event_stable_and_new']) for s in steps),
            repeated_step_candidate_hashes=len(steps)-len({s['candidate_hash'] for s in steps}),
            SUN_curve=[s['recorded_curve_SUN_after'] for s in steps])
        for d in ds:
            plan=d['failure_controller'] or {};support=d['uncertainty_support'] or {};controlled=arm in ('full_support_aware','uq_only_support_aware')
            nn=d['raw_scalar_NN_if_recorded'] if d['generation_success'] else None
            decision=dict(job_id=jid,arm=arm,task_id=j['task_id'],seed=j['seed'],decision_id=d['decision_id'],sequence=d['sequence'],candidate_index=d['candidate_index'],
                tool=(d['action'] or {}).get('tool'),disposition=d['disposition'],generation_success=d['generation_success'],controller_schema_valid=d['controller_schema_valid'],
                graph_status=d['graph_status'],graph_error=d['graph_error'],support=support.get('supported'),scalar_NN=nn,scalar_origin=d['scalar_origin'],
                decision_scalar=d['recorded_decision_scalar'],threshold=threshold,controller_enabled=controlled,
                scalar_above_threshold=nn>=threshold if nn is not None else None,
                supported_scalar_above_threshold=bool(controlled and d['generation_success'] and support.get('supported') and d['recorded_decision_scalar'] is not None and d['recorded_decision_scalar']>=threshold),
                request_retry=plan.get('request_retry',False),trigger=plan.get('trigger'),trigger_source=plan.get('trigger_source'),triggered_heads=plan.get('triggered_heads',[]),
                local_schema_issues=plan.get('local_schema_issues',[]),has_actual_next_candidate=d['has_next_recorded_candidate'],
                next_scientific_rpc_id=d['next_scientific_rpc_id'],direct_material_step_rpc_id=d['direct_material_step_rpc_id'],
                label_future_failure=d['label_future_failure'],tool_error=error(d['tool_error']),received_retry_feedback=d['received_retry_feedback'])
            decisions.append(decision);record['supported_scalar_ge_threshold']+=decision['supported_scalar_above_threshold']
            if plan.get('request_retry'):
                if plan.get('trigger_source')=='local_schema_validation':record['controller_schema_retry_requests']+=1
                elif plan.get('trigger_source')=='predicted_failure_type':record['controller_neural_retry_requests']+=1
                typed_requests.append({k:decision[k] for k in ('job_id','arm','task_id','seed','decision_id','candidate_index','trigger','trigger_source','triggered_heads','has_actual_next_candidate','local_schema_issues')})
            if d['disposition']=='executed':
                a=aline[d['decision_id']];assert a['tool_rpc_id']==d['tool_rpc_id']
                if not a['tool_ok']:
                    record['tool_failures']+=1;record['failed_selects']+=a['action']['tool']=='select_for_evaluation'
                    failures.append({'job_id':jid,'arm':arm,'task_id':j['task_id'],'seed':j['seed'],'decision_id':d['decision_id'],
                        'tool_rpc_id':a['tool_rpc_id'],'tool':a['action']['tool'],'error':error(a['tool_error']),
                        'next_scientific_rpc_id':d['next_scientific_rpc_id'],'action':compact_action(d)})
            else:assert d['executed_tool_outcome'] is None and d['tool_rpc_id'] is None and d['direct_material_step_rpc_id'] is None
        for seq,gg in sorted(groups.items()):
            gg.sort(key=lambda d:d['candidate_index']);assert [d['candidate_index'] for d in gg]==list(range(len(gg))) and len(gg) in (1,2)
            assert all(d['has_next_recorded_candidate']==(i+1<len(gg)) for i,d in enumerate(gg))
            chosen=[d for d in gg if d['disposition']=='executed'];assert len(chosen)<=1
            sel=gg[0]['candidate_selection'] or {};assert all((d['candidate_selection'] or {})==sel for d in gg)
            ranking=sel.get('ranking',[]);valid=[r['candidate_index'] for r in ranking if r['valid']]
            chosen_index=chosen[0]['candidate_index'] if chosen else None
            if sel:assert chosen_index==sel['selected_index']
            first=gg[0];plan=first['failure_controller'] or {};source_kind=plan.get('trigger_source')
            row={'job_id':jid,'arm':arm,'task_id':j['task_id'],'seed':j['seed'],'sequence':seq,
                'decision_ids':[d['decision_id'] for d in gg],'proposal_count':len(gg),'chosen_index':chosen_index,
                'first_trigger':plan.get('trigger'),'first_trigger_source':source_kind,
                'generation_success':[d['generation_success'] for d in gg],'controller_schema_valid':[d['controller_schema_valid'] for d in gg],
                'graph_statuses':[d['graph_status'] for d in gg],'candidate_tools':[(d['action'] or {}).get('tool') for d in gg],
                'controller_valid_indices':sorted(valid),'type_signals_used':sel.get('type_signals_used'),
                'ranking_rule':sel.get('ranking_rule'),'common_heads':sel.get('common_comparison_heads'),
                'selection_changed_from_first_controller_valid':bool(len(valid)>=2 and chosen_index!=min(valid)),
                'chosen_decision_id':chosen[0]['decision_id'] if chosen else None,
                'actual_next_candidate':len(gg)==2,'second_feedback':gg[1]['received_retry_feedback'] if len(gg)==2 else None}
            seqrows.append(row);record['multiple_valid_risk_selection_changed']+=row['selection_changed_from_first_controller_valid']
            if len(gg)==2:
                record['actual_second_proposals']+=1;record['actual_second_schema']+=source_kind=='local_schema_validation';record['actual_second_neural']+=source_kind=='predicted_failure_type'
                if controlled:=arm in ('full_support_aware','uq_only_support_aware'):
                    expected={'predicted_failure_type':plan['trigger'],'instruction':plan['feedback_for_retry'],
                        'suggested_legal_tool':plan['optional_tool_preference'],'source':'current_candidate_predictions_or_visible_schema_issue','not_a_measured_scientific_result':True}
                    assert gg[1]['received_retry_feedback']==expected
                    retry_feedback.append(dict(job_id=jid,arm=arm,task_id=j['task_id'],seed=j['seed'],first_decision_id=first['decision_id'],second_decision_id=gg[1]['decision_id'],source=source_kind,feedback=expected))
                record['selected_second']+=chosen_index==1;record['selected_second_after_schema']+=chosen_index==1 and source_kind=='local_schema_validation'
                if source_kind=='predicted_failure_type':
                    extra=gg[1];unused=extra['disposition']!='executed';record['unexecuted_neural_second']+=unused
                    record['neural_second_generation_seconds']+=extra['generation_seconds'];record['neural_second_graph_seconds']+=extra['graph_seconds'] or 0
                    record['neural_second_prompt_tokens']+=extra['generation']['prompt_token_count'];record['neural_second_completion_tokens']+=extra['generation']['completion_count']
                    neural_extra.append(dict(row,second_graph_error=extra['graph_error'],second_generation_seconds=extra['generation_seconds'],
                        second_graph_seconds=extra['graph_seconds'],second_prompt_tokens=extra['generation']['prompt_token_count'],
                        second_completion_tokens=extra['generation']['completion_count'],second_unused=unused))
        for step in steps:
            d=byid[step['selection_decision_id']];assert d['disposition']=='executed' and (d['action'] or {}).get('tool')=='select_for_evaluation'
            assert d['direct_material_step_rpc_id']==step['rpc_id']
            obs=step['official_observation'] or {};lab=int(not step['ok'] or not step['event_stable_and_new']);assert d['label_future_failure']==lab
            outcomes.append({'job_id':jid,'arm':arm,'task_id':j['task_id'],'seed':j['seed'],'step_index':step['step_index'],'rpc_id':step['rpc_id'],
                'decision_id':d['decision_id'],'candidate_hash':step['candidate_hash'],'reduced_formula':obs.get('reduced_formula'),
                'is_stable':obs.get('is_stable'),'is_newly_discovered':obs.get('is_newly_discovered'),'event_stable_and_new':step['event_stable_and_new'],
                'SUN_before':step['recorded_curve_SUN_before'],'SUN_after':step['recorded_curve_SUN_after'],'SUN_delta':step['recorded_curve_SUN_delta'],
                'e_above_hull':obs.get('e_above_hull'),'direct_selection_scalar':d['recorded_decision_scalar'],'graph_status':d['graph_status'],
                'linked_preceding_executed_actions':sum(z['next_scientific_rpc_id']==step['rpc_id'] and z['disposition']=='executed' for z in ds)})
        assert len(aline)==sum(record[t+'_executed'] for t in TOOLS)
        trajectories.append(record);provenance.append({'job':j,'source_files':x['source_files'],'profile':x['profile']})
    assert len(decisions)==1237 and len(seqrows)==1176 and len(outcomes)==600 and sum(r['executed_tools'] for r in trajectories)==1174
    byarm=[aggregate([r for r in trajectories if r['arm']==a],arm=a) for a in ARMS]
    bysystem=[aggregate([r for r in trajectories if r['arm']==a and r['task_id']==t],arm=a,task_id=t) for a in ARMS for t in SYSTEMS]
    byseed=[aggregate([r for r in trajectories if r['arm']==a and r['seed']==s],arm=a,seed=s) for a in ARMS for s in (2,3,4)]
    pairs=[]
    lookup={(r['arm'],r['task_id'],r['seed']):r for r in trajectories}
    for t in SYSTEMS:
        for s in (2,3,4):
            values={a:lookup[(a,t,s)] for a in ARMS}
            pairs.append({'task_id':t,'seed':s,**{a+'_SUN':v['SUN'] for a,v in values.items()},**{a+'_AUDC':v['AUDC'] for a,v in values.items()},
                'Full_minus_UQ_SUN':values['full_support_aware']['SUN']-values['uq_only_support_aware']['SUN'],
                'Full_minus_UQ_AUDC':values['full_support_aware']['AUDC']-values['uq_only_support_aware']['AUDC'],
                'UQ_minus_base_SUN':values['uq_only_support_aware']['SUN']-values['baseline_reference']['SUN'],
                'UQ_minus_base_AUDC':values['uq_only_support_aware']['AUDC']-values['baseline_reference']['AUDC']})
    case_ids={r['job_id'] for r in trajectories if r['task_id']=='Al-V-Zn' or r['seed']==4}
    cases=[{'trajectory':r,'direct_outcomes':[o for o in outcomes if o['job_id']==r['job_id']],
        'failed_tools':[f for f in failures if f['job_id']==r['job_id']],
        'retries':[s for s in seqrows if s['job_id']==r['job_id'] and s['proposal_count']==2],
        'generated_actions':[{'decision_id':d['decision_id'],**compact_action(d)} for x in runs if x['job']['job_id']==r['job_id'] for d in x['decisions'] if (d['action'] or {}).get('tool')=='generate_structures' and d['disposition']=='executed']} for r in trajectories if r['job_id'] in case_ids]
    for name,rows in [('trajectories.csv',trajectories),('per_arm.csv',byarm),('per_system_arm.csv',bysystem),('per_seed_arm.csv',byseed),('paired_results.csv',pairs),('sequence_effects.csv',seqrows),('decision_signals.csv',decisions),('direct_material_outcomes.csv',outcomes),('tool_failures.csv',failures)]:csvsave(name,rows)
    save('neural_extra_candidates.json',neural_extra);save('retry_feedback.json',retry_feedback);save('controller_requests.json',typed_requests);save('key_cases.json',cases);save('source_provenance.json',provenance)
    summary={'schema':'MADE_all60_core_trace_mechanism_analysis_v1','input':{'path':str(p),'sha256':DATA_SHA},'original_summary_sha256':SUMMARY_SHA,
        'actual_observed_CST':datetime.fromtimestamp(source['finished_epoch'],timezone(timedelta(hours=8))).isoformat(),'actual_observed_UTC':datetime.fromtimestamp(source['finished_epoch'],timezone.utc).isoformat(),
        'trajectory_count':60,'proposals':1237,'execution_sequences':1176,'executed_policy_tools':1174,'candidate_ORB_steps':600,
        'per_arm':byarm,'per_system_arm':bysystem,'per_seed_arm':byseed,'paired_results':pairs,
        'all_neural_extra_count':len(neural_extra),'neural_extra_all_unused':all(x['second_unused'] for x in neural_extra),
        'neural_extra_graph_unavailable':sum(x['graph_statuses'][1]=='unavailable' for x in neural_extra),
        'per_arm_failure_messages':{a:dict(Counter(f['error'] for f in failures if f['arm']==a)) for a in ARMS},
        'type_signal_flag_is_not_action_change':True,'unexecuted_counterfactual_outcomes_imputed':False,
        'NN_prompt_numeric_values_in_schema_feedback':False,'no_policy_counterfactual_equivalence_claim':True,'no_test_tuning':True,
        'new_fits':0,'new_LLM_graph_oracle_calls':0,'public_push':False}
    save('summary.json',summary);print(json.dumps({'passed':True,'runs':60,'proposals':1237,'tools':1174,'physical_steps':600,'per_arm':[{k:r[k] for k in ('arm','SUN_sum','AUDC_mean','failed_selects','supported_scalar_ge_threshold','actual_second_neural','multiple_valid_risk_selection_changed')} for r in byarm]},sort_keys=True))


if __name__=='__main__':main()
