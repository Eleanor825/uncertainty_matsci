"""Read only two CLOSED p336 arms; export recorded decisions, never run a model.

Safe for hash-bound source exec: no __file__, project imports, subprocess or writes.
Candidate prompts/tokens/raw text and feature vectors are not exported. Missing
recorded alternative risks remain null; this probe does not recompute them.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,re,time

R=Path('/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607').resolve()
P=R/'experiments/discoveryworld_full_method_20260922/six_hour_subset_v1/explicit_repeat_no_graph_world4_seeded_confirmation_v2'
ARMS=('NoGraphRisk','ExplicitRepeatRisk')
READS={}

def need(value,message):
    if not value:raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(path,pin=None):
    path=Path(path).resolve();need(path.is_relative_to(P),'Read outside the fixed confirmation namespace')
    before=path.stat();need(before.st_size<=32*1024*1024,'Unexpected oversized report')
    data=path.read_bytes();after=path.stat()
    need(before.st_size==after.st_size==len(data)and before.st_mtime_ns==after.st_mtime_ns,'Report changed while reading')
    receipt={'artifact':str(path.relative_to(R)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
    if pin is not None:
        need(Path(pin['path']).resolve()==path and pin['sha256']==receipt['sha256']and('bytes'not in pin or pin['bytes']==len(data)),'Closed-result report hash differs')
    READS[str(path)]=receipt
    if path.suffix=='.jsonl':
        need(not data or data.endswith(b'\n'),'Partial closed report')
        return [json.loads(line)for line in data.splitlines()if line.strip()],receipt
    return json.loads(data),receipt

def failure_detail(row):
    error=row.get('error');detail={'error_type':row.get('error_type'),'error_sha256':digest(error)if error is not None else None}
    if not error:return detail
    # Export known diagnostic messages, never an arbitrary exception payload.
    match=re.fullmatch(r'Current-policy transcoder fidelity gate failed at ([A-Za-z0-9_.]+): FVU=([-+0-9.eE]+), defined=(True|False)\.',error)
    if match:
        detail.update(category='transcoder_fidelity_gate',error=error,layer=match[1],output_fvu=float(match[2]),fvu_defined=match[3]=='True')
    elif re.fullmatch(r'Cut model changed forward logits \(max abs [-+0-9.eE]+\)\.',error):
        detail.update(category='capture_logit_parity',error=error)
    elif error in ('Original capture32 FVU failed','Malformed action JSON has no semantic action graph',
        'A token inseparably mixes action values with field names or rationale',
        'Action value characters cannot be mapped completely to original token boundaries',
        'The exact emitted action JSON span is missing or ambiguous',
        'The emitted JSON has no nonempty action-bearing scalar values',
        'Original capture must return exactly2 full forwards','Strict3-forward NoGraph procedure changed'):
        detail.update(category='recorded_capture_or_target_gate',error=error)
    else:detail.update(category='other_recorded_error',error='[unrecognized diagnostic withheld; type and hash retained]')
    return detail

def candidate(root,index,proposal,evidence,risk,risk_source):
    stem=f'a{index:04d}_p{proposal}';entry=evidence[(index,proposal)]
    saved,cr=read(root/'candidates'/(stem+'.json'),entry['candidate_with_raw_prompt'])
    ng,nr=read(root/'no_graph'/(stem+'.json'),entry['no_graph_features']);g=saved['generation']
    need(ng['schema']=='dw_no_graph_native_capture_features_v1'and ng['graph_status']=='not_requested'and ng['full_method_features_available']is False,'Wrong NoGraph report')
    need(ng['capture_assemble_calls']==ng['candidate_backward_calls']==0,'Unexpected graph/backward report')
    need(ng['generation_seed']==g['seed']and ng['prefix_hash']==g['prefix_hash']and ng['generated_action']==g['parsed_action']and ng['raw_text_sha256']==hashlib.sha256(g['raw_text'].encode()).hexdigest(),'Candidate/NoGraph identity differs')
    need(type(g['success'])is bool and type(ng['supported'])is bool,'Invalid support flags')
    if risk is not None:need(type(risk)in(int,float)and math.isfinite(risk)and 0<=risk<=1,'Invalid recorded risk')
    support=g['success']and ng['supported']and ng['no_graph_features_available']is True
    reasons=[]
    if not g['success']:reasons.append('generation_not_successful')
    if not ng['supported']:reasons.append('NoGraph_capture_unsupported')
    if ng['no_graph_features_available']is not True:reasons.append('NoGraph_features_unavailable')
    return {'parsed_action':g['parsed_action'],'generation_success':g['success'],'action_sha256':digest(g['parsed_action']),
        'raw_text_sha256':ng['raw_text_sha256'],'recorded_risk':risk,'risk_source':risk_source,
        'NoGraph_supported':ng['supported'],'NoGraph_features_available':ng['no_graph_features_available'],
        'passes_basic_controller_support_flags':bool(support),'unsupported_flags':reasons,
        'forward_counts':ng['forward_counts'],'capture_failure':failure_detail(ng),
        'candidate_sha256':cr['sha256'],'NoGraph_sha256':nr['sha256']}

def arm(condition):
    job_id=f'dw-explicit-repeat-nograph-world4-seeded-confirm-v2-test-{condition}-w4-p336-b30-g0'
    root=P/'runs/v1/episodes/jobs'/job_id
    stored,sr=read(P/'runs/v1/groups/p336'/(condition+'_result.json'))
    job=stored['job'];result=stored['result'];closure=result['environment_process_closure']
    need(job['job_id']==job_id and job['world_seed']==4 and job['policy_seed']==336 and job['grounded_repair']==condition and job['max_agent_attempts']==30,'Wrong fixed arm')
    need(stored['accepted']is True and result['accepted']is True and result['completed']is True and result['failure']is None,'Arm not accepted/closed')
    need(closure['closed_response']and closure['returncode']==0 and not closure['child_still_running']and not closure['unknown_connection'],'Arm process not closed')
    need(not(root/'failure.json').exists(),'Arm has a failure marker')
    attempts,ar=read(root/'policy_attempts.jsonl',result['policy_attempts'])
    need(len(attempts)==result['attempts']==30,'Not the closed B30 arm')
    ce=stored['candidate_evidence'];evidence={(x['attempt_index'],x['proposal_index']):x for x in ce['all_candidates']}
    need(len(evidence)==len(ce['all_candidates'])==ce['generated_proposals'],'Repeated/missing candidate evidence')
    rows=[];counts=Counter();errors=Counter();paths=Counter();forwards=Counter()
    for index,attempt in enumerate(attempts,1):
        need(attempt['attempt_index']==index and attempt['candidate_count']in(1,2),'Attempt order/count differs')
        d=attempt['controller_decision'];s=attempt['selection'];risks=s.get('risks');n=attempt['candidate_count']
        if risks is not None:need(len(risks)==2 and n==2,'Unexpected recorded risk list')
        first=candidate(root,index,0,evidence,d.get('risk'),'controller_decision.risk')
        alt=candidate(root,index,1,evidence,risks[1]if risks is not None else None,'selection.risks[1]'if risks is not None else'not_persisted_when_comparison_unavailable')if n==2 else None
        if risks is not None:need(math.isclose(first['recorded_risk'],risks[0],abs_tol=1e-12),'Repeated first risk differs')
        parsed=[i for i,c in enumerate((first,alt)[:n])if c['generation_success']]
        expected=parsed[0]if parsed else 0
        if s.get('supported_comparison'):
            need(risks is not None,'Supported comparison lacks risks');expected=1 if risks[1]<risks[0]else 0
            path='alternative_lower_risk'if expected else'alternative_risk_tie_keeps_first'if risks[1]==risks[0]else'alternative_higher_risk_keeps_first'
        else:path='no_revision'if n==1 else'parser_fallback'if len(parsed)<2 else'incomplete_support_fallback'if s.get('reason')=='incomplete_graph_support'else'controller_exception_fallback'
        need(s['selected_index']==expected,'Saved selection contradicts frozen selector')
        selected=(first,alt)[s['selected_index']]
        need(attempt['generation']['parsed_action']==selected['parsed_action']and attempt['generation']['success']==selected['generation_success'],'Chosen generation differs')
        changed=alt is not None and alt['parsed_action']is not None and alt['parsed_action']!=first['parsed_action']
        executed=selected['parsed_action']if selected['generation_success']else None
        counts.update(attempts=1,generated_proposals=n,revisions=int(n==2),neural_revision_requests=int(d['neural_revision']),
            different_parsed_revisions=int(changed),selected_alternative=int(s['selected_index']==1),neural_rank_changes=int(bool(s.get('rankchange'))),executed_action_changed_from_first=int(executed!=first['parsed_action']),
            supported_comparisons=int(bool(s.get('supported_comparison'))),different_revision_support_failures=int(changed and not alt['passes_basic_controller_support_flags']))
        paths[path]+=1
        for proposal,c in enumerate((first,alt)[:n]):
            counts['NoGraph_supported_candidates']+=int(c['NoGraph_supported']);counts[f'proposal{proposal}_supported']+=int(c['NoGraph_supported'])
            counts[f'proposal{proposal}_parsed_success']+=int(c['generation_success']);forwards.update(c['forward_counts'])
            if not c['NoGraph_supported']:errors[(str(proposal),c['capture_failure'].get('category','no_recorded_error'),c['capture_failure'].get('error_type'),c['capture_failure'].get('layer'))]+=1
        audit=d.get('explicit_repeat_feedback_audit',{})
        rows.append({'attempt':index,'first':first,'alternative':alt,'raw_text_changed':alt['raw_text_sha256']!=first['raw_text_sha256']if alt else None,
            'parsed_action_changed':changed if alt else None,'decision':{k:d.get(k)for k in ('risk','supported','neural_revision','reason')},
            'matching_prior_failures':audit.get('matching_prior_failures'),'selection':{k:s.get(k)for k in ('selected_index','supported_comparison','neural_rank_change','rankchange','risks','reason')},
            'selection_path':path,'selected_action':executed,'selected_action_changed_from_first':executed!=first['parsed_action']})
    need(counts['generated_proposals']==len(evidence)==ce['generated_proposals']and counts['NoGraph_supported_candidates']==ce['NoGraph_supported_candidates']and dict(forwards)==ce['NoGraph_forward_counts'],'Closed candidate totals differ')
    need(counts['neural_revision_requests']==result['controller_counts']['supported_neural_revisions']and counts['neural_rank_changes']==result['controller_counts']['supported_rank_changes'],'Closed controller totals differ')
    for key in ('generated_revision_action_different','executed_action_changed_from_first_proposal'):
        actual=counts['different_parsed_revisions'if key=='generated_revision_action_different'else'executed_action_changed_from_first']
        need(actual==stored['action_change_and_repetition']['counts'][key],'Closed action-change totals differ')
    return {'condition':condition,'world_seed':4,'policy_seed':336,'closed_B30_validated':True,'sources':{'closed_result':sr,'policy_attempts':ar},
        'summary_counts':dict(counts),'recorded_forward_totals':dict(forwards),'selection_paths':dict(paths),
        'unsupported_groups':[{'proposal':int(k[0]),'category':k[1],'error_type':k[2],'layer':k[3],'count':v}for k,v in sorted(errors.items(),key=lambda x:str(x[0]))], 'steps':rows}

def main():
    arms=[arm(condition)for condition in ARMS]
    print(json.dumps({'schema':'dw_p336_two_arm_revision_failure_readonly_audit_v1','observed_unix':time.time(),'read_only':True,
        'arms':arms,'read_summary':{'files':len(READS),'bytes':sum(x['bytes']for x in READS.values()),'stable_at_read_time':True},
        'scientific_calls':{'LLM':0,'NN':0,'environment':0,'optimizer':0,'GPU_query':0},
        'private_task_answers_or_maps_read':False,'raw_prompts_tokens_or_activations_exported':False,
        'risks_are_only_previously_recorded_values':True,'unexecuted_outcome_labels_created':False,
        'limitation':'Alternative risk is not persisted by RiskController when any candidate support is missing; null is not a recomputed score.'},sort_keys=True,allow_nan=False),flush=True)

if __name__=='__main__':main()
