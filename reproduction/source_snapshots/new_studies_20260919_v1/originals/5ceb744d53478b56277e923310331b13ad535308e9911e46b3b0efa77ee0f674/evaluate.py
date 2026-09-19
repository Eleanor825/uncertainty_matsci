"""One explicitly assigned component trajectory, or a CPU-only scope aggregate.

No automatic scheduling or retries. Unknown/partial claims remain permanently
reserved. Production execution requires the assigned empty GPU's shared locks.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'frozen_src'))
import argparse,json,os,socket,statistics,time,uuid
from matdiscovery.accounting import file_sha256,fingerprint,write_json_atomic
from matdiscovery.support_scope import (ARMS,BUDGETS,EVALUATION_SEEDS,SCHEMA,artifact,read,
    read_registration,validate_launch,validate_evaluation_job,require)
from matdiscovery.immutable_hash_cache import immutable_hash_cache
from full_es_training import publish


def run_job(reg,job_id,*,policy_factory=None,rollout_factory=None,_context=None):
    validate_launch(reg)
    matches=[j for j in reg['new_jobs'] if j['job_id']==job_id];require(len(matches)==1,'Unregistered evaluation job')
    job=matches[0];validate_evaluation_job(reg,job,job['budget'])
    if policy_factory is None:
        from resource_guard import require_resource
        require_resource(reg)
    folder=Path(reg['workspace'])/'experiments/support_final/jobs'/job_id
    folder.mkdir(parents=True,exist_ok=False)
    claim={'schema':SCHEMA,'support_fingerprint':reg['fingerprint'],'job':job,'attempt_id':uuid.uuid4().hex,
           'pid':os.getpid(),'hostname':socket.gethostname(),'started_at':time.time(),'automatic_replay':False}
    with (folder/'claim.json').open('x') as f:
        json.dump(claim,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    output=folder/claim['attempt_id']
    try:
        import torch
        from matdiscovery.core_final import CoreFinalRunner
        from matdiscovery.esopt import tensor_state_hash
        from matdiscovery.final_evaluation import execution_job
        from matdiscovery.rollouts import RolloutSettings
        from matdiscovery.training_jobs import reconstruct_scientific_evidence,TrainingJobCallbacks
        from arm_runtime import SupportAwareRollout,prepare_arm,selected_generation
        context={} if _context is None else _context
        if not context:
            provider=CoreFinalRunner(reg['parent_core']['workspace'],policy_factory=policy_factory)
            torch.set_num_threads(provider.runtime['torch_cpu_threads'])
            if policy_factory is None:torch.cuda.set_per_process_memory_fraction(provider.runtime['cuda_memory_fraction'],device=provider.runtime['device'])
            policy=provider.load_policy();base={n:t.detach().cpu().clone() for n,t in policy.model.state_dict().items()}
            runtime=prepare_arm(provider,policy,job['arm_id'],base,tensor_state_hash(base),reg);del base
            context.update(provider=provider,policy=policy,runtime=runtime,arm_id=job['arm_id'])
        require(context['arm_id']==job['arm_id'],'A resident worker may not silently change its arm')
        provider,policy,runtime=(context[k] for k in ('provider','policy','runtime'))
        profile=runtime.profile
        profile_path=Path(reg['workspace'])/'profiles'/(job['arm_id']+'.json')
        publish(profile_path,{'support_fingerprint':reg['fingerprint'],'profile':profile,'profile_fingerprint':fingerprint(profile)})
        actual=execution_job(job,policy,profile);actual.update(source=job['source'],group_id=job['group_id'])
        require(actual['selected_generation']==selected_generation(profile,job['arm_id']),'Wrong live selected generation')
        settings=RolloutSettings(max_generation_retries=provider.protocol['risk_network']['maximum_candidate_generations_per_decision'],
            risk_threshold=provider.protocol['risk_network']['threshold'],history_results=provider.protocol['memory']['scientific_history_per_episode'],
            recent_tools=provider.protocol['memory']['recent_tool_responses'],failure_aware_control=runtime.failure_aware_control)
        provider.unchanged()
        (rollout_factory or SupportAwareRollout)(provider.project,policy,settings=settings,risk_model=runtime.risk,attributor=runtime.graph).run(actual,output,collection=False)
        require(policy.get_state_id()==actual['policy_state_id'] and tensor_state_hash(dict(policy.model.state_dict()))==profile['actual_model_state_hash'],'Fixed evaluation weights mutated')
        evidence=reconstruct_scientific_evidence(output,actual,tasks=provider.tasks,partition='final_test',expected_mace_num_workers=4,
            expected_made_budget=job['budget'],support_registration=reg)
        provider.unchanged()
        result={'schema':SCHEMA,'complete':True,'support_fingerprint':reg['fingerprint'],'job':job,'execution_job':actual,
                'claim':artifact(folder/'claim.json'),'profile':profile,'profile_record':artifact(profile_path),
                'scientific_evidence':evidence,'episodes':read(output/'episodes.json'),
                'artifacts':TrainingJobCallbacks._inventory(output),'main_study_complete':False,'finished_at':time.time()}
        publish(folder/'result.json',result)
        receipt={'schema':SCHEMA,'complete':True,'support_fingerprint':reg['fingerprint'],'job_id':job_id,
                 'result':artifact(folder/'result.json'),'claim':artifact(folder/'claim.json')}
        receipt['fingerprint']=fingerprint(receipt);publish(folder/'receipt.json',receipt)
        return receipt
    except BaseException as exc:
        publish(folder/'failure.json',{'schema':SCHEMA,'job_id':job_id,'error_type':type(exc).__name__,
            'physical_outcome_requires_reconciliation':True,'automatic_replay':False,'failed_at':time.time()})
        raise


def run_arm(reg,arm,actor_id):
    """One resident model per assigned arm; each job still has one permanent claim."""
    import re,gc
    from resource_guard import require_resource
    require_resource(reg);validate_launch(reg)
    require(arm in ARMS and isinstance(actor_id,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',actor_id),'Explicit safe actor/arm identity required')
    actor=Path(reg['workspace'])/'experiments/support_final/actors'/actor_id
    actor.mkdir(parents=True,exist_ok=False)
    with (actor/'invocation.json').open('x') as f:
        json.dump({'support_fingerprint':reg['fingerprint'],'arm':arm,'actor_id':actor_id,'pid':os.getpid(),
                   'hostname':socket.gethostname(),'started_at':time.time(),'automatic_replay':False},f)
        f.flush();os.fsync(f.fileno())
    context={};done=[]
    try:
        for job in (j for j in reg['new_jobs'] if j['arm_id']==arm):
            directory=Path(reg['workspace'])/'experiments/support_final/jobs'/job['job_id']
            if directory.exists():continue
            write_json_atomic(actor/'status.json',{'status':'attempting_one_registered_job','job_id':job['job_id'],'completed_job_ids':done,'time':time.time()})
            try:receipt=run_job(reg,job['job_id'],_context=context)
            except FileExistsError:
                # Only skip a lost atomic directory reservation. A worker that
                # wrote its own claim never returns FileExistsError normally.
                if not directory.exists():raise
                claim_path=directory/'claim.json'
                if claim_path.exists():
                    prior=read(claim_path)
                    if prior.get('pid')==os.getpid() and prior.get('hostname')==socket.gethostname():raise
                continue
            done.append(job['job_id'])
        value={'status':'no_unclaimed_jobs_for_assigned_arm','support_fingerprint':reg['fingerprint'],
               'arm':arm,'actor_id':actor_id,'completed_job_ids':done,'global_complete_claimed':False}
        publish(actor/'receipt.json',value);return value
    except BaseException as exc:
        publish(actor/'failure.json',{'error_type':type(exc).__name__,'automatic_replay':False,'completed_job_ids':done})
        raise
    finally:
        context.clear();gc.collect()
        import torch
        if torch.cuda.is_available():torch.cuda.empty_cache()


def verify_decision_components(job,actual,rows):
    from decision_validation import validate_support_decisions
    validate_support_decisions(rows)
    for row in rows:
        require(row['support_arm']==job['arm_id'] and row['registered_control_route']==job['control_route'],'Actual support route changed')
        ident=row['local_decision_id'];seq=int(ident.split('-')[0][1:]);candidate=int(ident.split('-c')[1])
        require(row['generation']['seed']==(job['seed']*10000019+seq*97+candidate)%2**63,'Generation seed changed')
        stamp=row['policy_stamp']
        require(stamp['state_id']==actual['policy_state_id'] and stamp['generation']==actual['selected_generation'],
                'Decision evaluated the wrong fixed checkpoint')


def audit_job(reg,job):
    from arm_runtime import validate_profile,selected_generation
    from matdiscovery.training_jobs import TrainingJobCallbacks,reconstruct_scientific_evidence
    validate_evaluation_job(reg,job,job['budget'])
    folder=Path(reg['workspace'])/'experiments/support_final/jobs'/job['job_id']
    receipt=read(folder/'receipt.json');require(receipt['fingerprint']==fingerprint({k:v for k,v in receipt.items() if k!='fingerprint'}),'Receipt seal changed')
    require(receipt['schema']==SCHEMA and receipt['complete'] is True and receipt['support_fingerprint']==reg['fingerprint'] and receipt['job_id']==job['job_id'],'Receipt scope changed')
    require(receipt['result']==artifact(folder/'result.json') and receipt['claim']==artifact(folder/'claim.json'),'Result/claim changed')
    claim=read(folder/'claim.json');result=read(folder/'result.json')
    require(claim['job']==job and claim['support_fingerprint']==reg['fingerprint'] and claim['automatic_replay'] is False,'Claim changed')
    attempt=claim.get('attempt_id')
    require(isinstance(attempt,str) and len(attempt)==32 and all(c in '0123456789abcdef' for c in attempt),'Invalid or escaping attempt identity')
    require(result['schema']==SCHEMA and result['complete'] is True and result['job']==job and result['claim']==receipt['claim'] and result['support_fingerprint']==reg['fingerprint'],'Executed another component job')
    profile=validate_profile(result['profile'],reg,arm_id=job['arm_id']);actual=result['execution_job']
    profile_path=Path(reg['workspace'])/'profiles'/(job['arm_id']+'.json')
    wrapper=read(profile_path)
    require(result['profile_record']==artifact(profile_path) and wrapper=={'support_fingerprint':reg['fingerprint'],'profile':profile,'profile_fingerprint':fingerprint(profile)},'Unsealed final component profile')
    validate_evaluation_job(reg,actual,job['budget'],execution=True)
    require(actual['execution_profile_fingerprint']==fingerprint(profile) and actual['actual_model_state_hash']==profile['actual_model_state_hash']
            and actual['selected_generation']==selected_generation(profile,job['arm_id']) and actual['initial_checkpoint_manifest_hash']==profile['initial_checkpoint_manifest_hash']
            and actual['policy_configuration_fingerprint']==profile['policy_configuration_fingerprint'],'Actual checkpoint/runtime profile changed')
    output=folder/attempt;require(output.is_dir() and not output.is_symlink(),'Missing or aliased physical attempt')
    require(result['artifacts']==TrainingJobCallbacks._inventory(output),'Raw component evidence changed')
    tasks=read(Path(reg['parent_core']['workspace'])/'configs/benchmark_tasks.json')
    evidence=reconstruct_scientific_evidence(output,actual,tasks=tasks,partition='final_test',expected_mace_num_workers=4,
        expected_made_budget=job['budget'],support_registration=reg)
    episodes=read(output/'episodes.json');require(evidence==result['scientific_evidence'] and result['episodes']==episodes and len(episodes)==1,'Reconstructed evidence differs')
    rows=[json.loads(line) for line in (output/'decisions.jsonl').read_text().splitlines() if line.strip()]
    verify_decision_components(job,actual,rows)
    ep=episodes[0];sun=ep['discovery_curve'][-1][1]
    require(ep['metrics']['mSUN']==sun/job['budget'],'mSUN denominator changed')
    return {'job_id':job['job_id'],'arm':job['arm_id'],'task_id':job['task_id'],'seed':job['seed'],'budget':job['budget'],
            'SUN':sun,'mSUN':sun/job['budget'],'AUDC':ep['metrics']['AUDC'],'costs':evidence['costs'],'receipt':artifact(folder/'receipt.json')}


def aggregate(reg):
    validate_launch(reg);rows=[];states={}
    for job in reg['new_jobs']:
        folder=Path(reg['workspace'])/'experiments/support_final/jobs'/job['job_id']
        if (folder/'receipt.json').exists():rows.append(audit_job(reg,job));state='complete'
        elif (folder/'failure.json').exists():state='failed_requires_reconciliation'
        elif (folder/'claim.json').exists():state='claimed_unknown_or_active'
        else:state='unclaimed'
        states[state]=states.get(state,0)+1
    from shared_references import audit_shared
    shared=audit_shared(reg)
    complete=len(rows)==540 and shared['complete']
    report={'schema':SCHEMA,'support_fingerprint':reg['fingerprint'],'complete':complete,
        'new_expected_jobs':540,'shared_expected_jobs':540,'new_states':states,'rows':rows,
        'shared':shared,'legacy_uq_only_schema_fixed_included':False,'seed1_results_included':False}
    if complete:
        require(sum(r['costs']['candidate_oracle_attempts'] for r in rows)==16200,'New evaluation costs differ')
        allrows=rows+shared['rows']
        require(sum(r['costs']['candidate_oracle_attempts'] for r in allrows)==32400,'Shared physical calls double counted or missing')
        stats={}
        for budget in BUDGETS:
            stats[str(budget)]={}
            for arm in (*ARMS,'baseline_reference','es_only_independent'):
                chosen=[r for r in allrows if r['budget']==budget and r['arm']==arm]
                require(len(chosen)==90,'Incomplete arm/budget repeated-seed matrix')
                metrics={}
                for metric in ('SUN','mSUN','AUDC'):
                    means=[]
                    for seed in EVALUATION_SEEDS:
                        group=[r for r in chosen if r['seed']==seed];require(len(group)==30,'Missing per-seed system')
                        means.append(statistics.mean(r[metric] for r in group))
                    metrics[metric]={'seed_ids':list(EVALUATION_SEEDS),'seed_means':means,'n_seeds':3,
                        'mean':statistics.mean(means),'sample_variance':statistics.variance(means),'sample_sd':statistics.stdev(means)}
                stats[str(budget)][arm]=metrics
        report['across_seed_statistics']=stats
        report['paired_changes_vs_shared_baseline']={}
        for budget in BUDGETS:
            report['paired_changes_vs_shared_baseline'][str(budget)]={}
            for arm in (*ARMS,'es_only_independent'):
                values={}
                for metric in ('SUN','mSUN','AUDC'):
                    baseline=stats[str(budget)]['baseline_reference'][metric]['seed_means']
                    delta=[b-a for a,b in zip(baseline,stats[str(budget)][arm][metric]['seed_means'],strict=True)]
                    values[metric]={'seed_ids':list(EVALUATION_SEEDS),'seed_deltas':delta,'n_seeds':3,
                        'mean':statistics.mean(delta),'sample_variance':statistics.variance(delta)}
                report['paired_changes_vs_shared_baseline'][str(budget)][arm]=values
        from full_es_training import verify_full_es_receipt
        report['new_full_ES']=verify_full_es_receipt(registration=reg)
        report['candidate_calls']={'new_full_ES':60,'shared_independent_ES_only':60,
            'new_UQ_and_full_evaluation':16200,'shared_baseline_and_ES_only_evaluation':16200,'combined_unique':32520}
        publish(Path(reg['workspace'])/'experiments/support_reports/report.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--job-id');p.add_argument('--arm',choices=tuple(ARMS));p.add_argument('--actor-id');p.add_argument('--execute',action='store_true');p.add_argument('--gpu-uuid');p.add_argument('--aggregate',action='store_true');a=p.parse_args()
    with immutable_hash_cache():
        reg=read_registration(a.workspace)
        if a.aggregate:result=aggregate(reg)
        elif a.execute:
            if bool(a.job_id)==bool(a.arm):p.error('--execute needs exactly one of --job-id or --arm')
            if a.arm and not a.actor_id:p.error('--arm requires a unique --actor-id')
            from resource_guard import exclusive_resource
            with exclusive_resource(reg,a.gpu_uuid):result=run_arm(reg,a.arm,a.actor_id) if a.arm else run_job(reg,a.job_id)
        else:result={'science_started':False,'jobs':reg['new_jobs'],'expected_counts':reg['expected_counts']}
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
