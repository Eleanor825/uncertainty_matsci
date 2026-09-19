"""Independent MADE ES-only training, never a rename of the old full-method G2.

Run with this extension's frozen_src on PYTHONPATH. Original ES arithmetic,
policy, rollout, physical reconstruction and reload checker are used unchanged.
Only this new registered six-trajectory scope is admitted at the shared boundary.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'frozen_src'))

import torch

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.es_training import (ESTrainingConfig, ESTrainingDriver, EvaluationRequest,
                                      EvaluationResult, TrainingCase, TrainingContractError)
from matdiscovery.esopt import AgenticESOpt, population_zscores, tensor_state_hash
from matdiscovery.component_scope import admitted_parent, read_registration, validate_launch
from matdiscovery.rollouts import DiscoveryRollout, RolloutSettings
from matdiscovery.training_jobs import (TrainingJobCallbacks, make_training_job,
    reconstruct_scientific_evidence, cpu_clean_reload_validator)

SCHEMA = 'made_independent_es_only_complete_v1'
JOB_SOURCE = 'registered_independent_MADE_ES_only_B10_training_v1'


def require(ok, message):
    if not ok: raise TrainingContractError(message)


def read(path):
    def duplicate(items):
        out={}
        for k,v in items:
            require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=duplicate,
        parse_constant=lambda v:(_ for _ in ()).throw(TrainingContractError('Nonfinite JSON')))


def artifact(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':file_sha256(path)}


def publish(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,path)
        except FileExistsError:
            require(read(path)==value,'Refuse to replace immutable ES-only evidence: '+str(path))
    finally:temporary.unlink(missing_ok=True)


def _source_bound(reg):
    require(artifact(__file__) in reg['external_sources'],'Executing ES-only source is not registered')


def _costs_match(left,right):
    """Physical counts exact; permit only floating summation-order roundoff."""
    integer_keys={'candidate_oracle_attempts','initialization_oracle_attempts','surrogate_oracle_attempts',
                  'dft_episode_attempts','step_attempts','rpc_calls','completed_episodes','llm_calls','completion_tokens','prompt_tokens'}
    for key in set(left)|set(right):
        a,b=left.get(key,0),right.get(key,0)
        if not (type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)):return False
        if key in integer_keys:
            if not (a==b and float(a).is_integer()):return False
        elif not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12):return False
    return True


def _parent_profile(reg):
    ref=reg['parent'];require(file_sha256(ref['path'])==ref['sha256'],'Parent source changed')
    return read(ref['path'])['execution_profiles']['baseline']


@dataclass(frozen=True)
class ESOnlyCondition:
    project: Path
    core: dict
    model_key: str='qwen35_4b'
    benchmark: str='made'
    method: str='esopt'
    seed: int=1
    property_name: str|None=None

    def protocol(self):
        source=self.core['derived_main_protocol']
        require(Path(source['path']).resolve()==self.project/'configs/main_protocol.json'
                and file_sha256(source['path'])==source['sha256'],'Original FAST training protocol changed')
        protocol=read(source['path']);es=protocol['esopt']
        fixed={'generations':2,'population':2,'cases_per_generation':1,'alpha':.0005,
               'sigma_start':.001,'sigma_end':.0002,'validation_every_generations':1,
               'schedule':'cosine','standardization':'population_z_score_ddof0','extra_inverse_sigma':False}
        require(all(es.get(k)==v and self.core['esopt'].get(k)==v for k,v in fixed.items()),'Independent ES must match frozen FAST G2/P2 hyperparameters')
        require(protocol['models']==['qwen35_4b'] and protocol['training_seeds']==[1],'Model/training seed changed')
        from matdiscovery.execution_contract import declared_mace_workers
        require(declared_mace_workers(protocol)==4,'ES-only needs the same MACE4/ORB1 evaluator')
        return protocol

    def config(self):
        es=self.protocol()['esopt']
        return ESTrainingConfig(benchmark='made',method='esopt',seed=1,dev_every=1,
            **{k:es[k] for k in ('generations','population','cases_per_generation','alpha','sigma_start','sigma_end')})

    def cases(self):
        expected={'train':{'id':'Al-Au-Hf','elements':['Al','Au','Hf']},
                  'dev':{'id':'Al-Pd-Sm','elements':['Al','Pd','Sm']}}
        result=[]
        for split,task in expected.items():
            require(self.core[split+'_tasks']==[task],'ES-only may use only the original fixed train/dev chemistry')
            result.append((TrainingCase(f'{split}:made:{task["id"]}',split,'made',
                payload={'task_id':task['id'],'task':task}),))
        require(not ({x['id'] for x in self.core['test_tasks']}&{x['id'] for x in expected.values()}),'Training/test chemistry overlap')
        return tuple(result)


def condition_for(reg):
    _source_bound(reg)
    core=admitted_parent(reg)
    require(core['model_key']=='qwen35_4b' and core['training_seed']==1,'Wrong admitted base core')
    return ESOnlyCondition(Path(core['workspace']).resolve(),core)


def condition_directory(reg): return Path(reg['workspace']).resolve()/'experiments/es_only'


def _builder(condition,request,entry,configuration_fingerprint,reg):
    # Pure field construction; no B50 job is executed or relabelled. The new B10
    # budget/source below is independently checked before every physical call.
    job=make_training_job(condition,request,model_entry=entry,
                          policy_configuration_fingerprint=configuration_fingerprint)
    job.update(job_id='es-only-'+request.request_id,budget=10,
               expected_counts={'episodes':1,'candidate_oracle_attempts':10,'dft_episode_attempts':0},
               source=JOB_SOURCE,ablation_fingerprint=reg['fingerprint'],
               ablation_training_scope='independent_es_only',risk_used=False,
               uncertainty_reward_weight=0.0,controller_enabled=False)
    return job


def _planner(condition,identity,directory):
    """Public driver constructor supplies its exact RNG plan without execution."""
    policy=SimpleNamespace(model_id=identity['model_id'],revision=identity['model_revision'],
        checkpoint_hash=identity['initial_checkpoint_manifest_hash'],
        configuration_fingerprint=identity['policy_configuration_fingerprint'])
    def unavailable(*a,**k): raise TrainingContractError('Read-only planner cannot execute')
    train,dev=condition.cases()
    return ESTrainingDriver(policy,condition.config(),train_cases=train,dev_cases=dev,
        job_factory=unavailable,evaluate=unavailable,result_verifier=unavailable,
        clean_reload_validator=unavailable,output_dir=directory,
        callback_fingerprint=identity['callback_fingerprint'])


def validate_training_job_scope(reg,job,expected_made_budget,core_protocol=None):
    condition=condition_for(reg)
    require(core_protocol is None or core_protocol==condition.core,'Independent training cannot replace its admitted parent')
    require(type(expected_made_budget) is int and expected_made_budget==10,'Independent ES-only budget is exactly B10')
    directory=condition_directory(reg);manifest=read(directory/'run/run_manifest.json')
    identity=manifest['identity'];planner=_planner(condition,identity,directory/'run')
    require(planner.identity==identity and planner.run_fingerprint==manifest['run_fingerprint'],'Independent training manifest/plan differs')
    descriptor=read(directory/'training_descriptor.json')
    require(descriptor['ablation_fingerprint']==reg['fingerprint'] and descriptor['driver_identity']==identity
            and descriptor['callback_fingerprint']==identity['callback_fingerprint'],'Independent callback/source identity differs')
    request=EvaluationRequest.from_dict(job['training_request'])
    require(request.method=='esopt' and request.run_fingerprint==planner.run_fingerprint
            and fingerprint(request.slot_identity())==request.request_id,'Wrong independent ES request')
    require(request.generation in (1,2),'Only both trained generations are registered')
    plan=planner.plan[request.generation-1];train,dev=condition.cases()
    if request.phase=='train':
        require(type(request.candidate_index) is int and request.candidate_index in (0,1)
                and request.case==train[0] and request.environment_seed==plan['environment_seeds'][0]
                and request.perturbation_seed==plan['population_seeds'][request.candidate_index]
                and request.perturbation_sigma==plan['sigma'],'Population task/seed/perturbation drift')
    else:
        require(request.phase=='dev' and request.candidate_index is None and request.case==dev[0]
                and request.environment_seed==plan['dev_environment_seeds'][0]
                and request.perturbation_seed is None and request.perturbation_sigma is None,'Development task/seed/perturbation drift')
    entry=next(x for x in read(condition.project/'configs/model_manifest.json')['models'] if x['key']=='qwen35_4b')
    expected=_builder(condition,request,entry,identity['policy_configuration_fingerprint'],reg)
    require(job==expected,'Unregistered training job fields or budget/source drift')
    return expected


def _no_risk_rows(output,request):
    rows=[json.loads(line) for line in (Path(output)/'decisions.jsonl').read_text().splitlines()]
    require(rows,'Missing actual ES-only decisions')
    for row in rows:
        identifier=row['local_decision_id'];sequence=int(identifier.split('-')[0][1:]);candidate=int(identifier.split('-c')[1])
        require(row['generation']['seed']==(request.environment_seed*10000019+sequence*97+candidate)%2**63,
                'ES-only actual generation RNG differs from the original paired seed rule')
        success=row['generation']['success']
        require(row.get('predicted_failure_probability') is None if success else row.get('predicted_failure_probability') in (None,1.0),
                'ES-only received a risk prediction; invalid-format sentinel is the only exception')
        require(not row.get('confidence_used_for_control',False) and not row.get('failure_types_used_for_control',False)
                and not row.get('failure_controller') and not row.get('failure_type_probabilities')
                and row.get('graph_status') not in ('succeeded',) and row.get('graph_seconds',0)==0,
                'Independent ES-only used uncertainty/graph/controller')
        stamp=row['policy_stamp']
        require(stamp['generation']==request.generation-int(request.phase=='train')
                and stamp.get('perturbation_seed')==request.perturbation_seed
                and stamp.get('perturbation_sigma')==request.perturbation_sigma,'Decision evaluated another ES state')
    return len(rows)


class ESOnlyCallbacks(TrainingJobCallbacks):
    def __init__(self,condition,policy,directory,reg,*,settings,rollout_factory=None):
        require(settings.failure_aware_control is False,'Risk control must be disabled')
        super().__init__(condition,policy,Path(directory)/'run',settings=settings,risk_model=None,attributor=None,
            auxiliary_files=[Path(__file__).resolve(),Path(reg['workspace'])/'configs/component_ablations.json'],
            rollout_factory=rollout_factory)
        self.reg=reg;self.expected_made_budget=10
        self.callback_identity.update(schema=SCHEMA,ablation_fingerprint=reg['fingerprint'],
            graph_artifacts='none',risk_model=None,native_attributor=None,risk_used=False,
            controller_enabled=False,uncertainty_reward_weight=0.0,made_episode_budget=10,
            evaluator_reward='official_AUDC_only',independent_training=True)
        self.fingerprint=fingerprint(self.callback_identity)

    def job_factory(self,request):
        self._unchanged()
        job=_builder(self.condition,request,self.model_entry,self.policy.configuration_fingerprint,self.reg)
        validate_training_job_scope(self.reg,job,10)
        return job

    def _scientific(self,request,job,output):
        value=reconstruct_scientific_evidence(output,job,tasks=self.tasks,partition='training',
            expected_mace_num_workers=4,expected_made_budget=10,component_registration=self.reg)
        _no_risk_rows(output,request)
        return value

    def evaluate(self,policy,job,request):
        require(policy is self.policy,'Wrong live policy');self._unchanged();output=self._directory(request)
        require(not output.exists(),'Existing/partial physical trajectory requires read-only reconciliation, not replay')
        self.rollout_factory(self.condition.project,policy,settings=self.settings,risk_model=None,attributor=None).run(job,output,collection=False)
        evidence=self._scientific(request,job,output);self._unchanged()
        receipt={'schema':SCHEMA,'complete':True,'request_identity':request.semantic_identity(),
            'job_fingerprint':fingerprint(job),'callback_fingerprint':self.fingerprint,
            'evidence':evidence,'artifacts':self._inventory(output)}
        publish(output/'completion_receipt.json',receipt)
        return self._result(request,receipt)

    def result_verifier(self,request,result):
        self._unchanged();output=self._directory(request);receipt=read(output/'completion_receipt.json');job=self.job_factory(request)
        require(receipt['schema']==SCHEMA and receipt['complete'] is True and receipt['request_identity']==request.semantic_identity()
                and receipt['job_fingerprint']==fingerprint(job) and receipt['callback_fingerprint']==self.fingerprint,
                'Independent trajectory receipt changed')
        require(self._inventory(output)==receipt['artifacts'],'Raw independent ES trajectory changed')
        evidence=self._scientific(request,job,output)
        require(evidence==receipt['evidence'] and fingerprint(asdict(result))==fingerprint(asdict(self._result(request,receipt))),
                'ES-only reward or costs not official evidence')
        return {'verified':True,'evidence':'Independent six-slot training scope; closed official RPC/ORB+MACE journal and no risk/graph/control',
                'metric_name':evidence['metric_name'],'metric_value':evidence['metric_value'],'costs':evidence['costs']}


def _proof(proof,marker):
    require(proof.get('verified') is True and proof.get('fresh_instance') is True and proof.get('allclose') is True
            and proof.get('verification_kind')=='fresh_model_tensor_hash_and_forward_logits'
            and proof['expected_state_hash']==proof['loaded_state_hash']==marker['actual_model_state_hash']
            and proof['checkpoint_file_sha256']==marker['files']['policy.pt'],'Clean instance/hash/forward reload proof failed')


def audit_es_only(condition_dir,*,registration):
    condition=condition_for(registration);directory=Path(condition_dir).resolve();run=directory/'run'
    require(directory==condition_directory(registration),'Wrong independent training directory')
    descriptor=read(directory/'training_descriptor.json');manifest=read(run/'run_manifest.json');summary=read(run/'training_summary.json')
    identity=manifest['identity'];planner=_planner(condition,identity,run)
    require(identity==planner.identity and manifest['run_fingerprint']==planner.run_fingerprint==summary['run_fingerprint'],
            'ES-only config/RNG/source run identity changed')
    require(descriptor['ablation_fingerprint']==registration['fingerprint'] and descriptor['driver_identity']==identity
            and descriptor['callback_fingerprint']==fingerprint(descriptor['callback_identity'])==identity['callback_fingerprint'],
            'Independent callback descriptor differs')
    require(descriptor['callback_identity']['risk_used'] is False and descriptor['callback_identity']['controller_enabled'] is False
            and descriptor['callback_identity']['uncertainty_reward_weight']==0,'Nonzero uncertainty involvement')
    require(summary['status']=='complete' and summary['completed_generations']==summary['required_generations']==2
            and summary['expected_evaluations']==summary['completed_evaluations']==6 and summary['test_data_used'] is False,
            'Independent G2/P2 stage incomplete')
    baseline=_parent_profile(registration);initial=manifest['initial_actual_model_state_hash']
    require(initial==baseline['initial_actual_model_state_hash']==baseline['actual_model_state_hash'],'Training started from old G2 or another base')
    entry=next(x for x in read(condition.project/'configs/model_manifest.json')['models'] if x['key']=='qwen35_4b')
    require(identity['model_id']==entry['model_id'] and identity['model_revision']==entry['revision']
            and identity['initial_checkpoint_manifest_hash']==baseline['initial_checkpoint_manifest_hash']
            and identity['policy_configuration_fingerprint']==baseline['policy_configuration_fingerprint'],
            'Independent training model/revision/runtime differs from paired parent')
    evidence={str(p.resolve()):file_sha256(p) for p in (directory/'training_descriptor.json',run/'run_manifest.json',run/'training_summary.json')}
    for path,sha in descriptor['callback_identity']['inputs'].items():
        require(file_sha256(path)==sha,'Frozen training/evaluator source changed');evidence[path]=sha
    markers={};histories={};states={}
    for g in (0,1,2):
        folder=run/'checkpoints'/f'generation_{g:04d}';marker=read(folder/'complete.json')
        require(marker['status']=='complete' and marker['generation']==g and marker['run_fingerprint']==planner.run_fingerprint
                and set(marker['files'])=={'policy.pt','policy.pt.json','es_history.json','driver_state.json'},'Missing original/base/trained checkpoint commit')
        evidence[str(folder/'complete.json')]=file_sha256(folder/'complete.json')
        for name,sha in marker['files'].items():
            require((folder/name).is_file() and not (folder/name).is_symlink() and file_sha256(folder/name)==sha,'Changed/partial/foreign independent checkpoint')
            evidence[str(folder/name)]=sha
        markers[g]=marker;histories[g]=read(folder/'es_history.json');states[g]=read(folder/'driver_state.json')
    records=states[2]['generations'];history=histories[2]
    require(len(records)==2 and states[1]['generations']==records[:1] and initial==markers[0]['actual_model_state_hash'],'Generation ledger mismatch')
    for g in (0,1,2):
        h=histories[g]
        require(len(h['history'])==g and h['history']==history['history'][:g] and h['base_state_hash']==initial
                and h['current_state_hash']==markers[g]['actual_model_state_hash'] and h['parameter_scope']=='full'
                and h['parameter_manifest']==manifest['full_parameter_manifest'],'Independent full-parameter history changed')
    files=sorted((run/'jobs').glob('*.json'));require(len(files)==6,'Independent ES needs exactly six durable jobs')
    jobs={};costs={'train':Counter(),'dev':Counter()};tasks=read(condition.project/'configs/benchmark_tasks.json')
    for path in files:
        durable=read(path);require(durable['status']=='complete','Unknown/failed independent evaluation')
        request=EvaluationRequest.from_dict(durable['request']);result=EvaluationResult(**durable['result'])
        output=run/'rollouts'/request.request_id;job=read(output/'job.json')
        if 'rollout_settings' in job:
            require(job.pop('rollout_settings')==descriptor['callback_identity']['settings'],'Saved rollout control settings changed')
        validate_training_job_scope(registration,job,10)
        require(job['training_request']==request.to_dict() and result.request_id==request.request_id,'Raw job/driver request mismatch')
        receipt=read(output/'completion_receipt.json')
        require(receipt['schema']==SCHEMA and receipt['complete'] and receipt['request_identity']==request.semantic_identity()
                and receipt['job_fingerprint']==fingerprint(job) and receipt['callback_fingerprint']==identity['callback_fingerprint'],'Raw completion differs')
        require(TrainingJobCallbacks._inventory(output)==receipt['artifacts'],'Raw training artifacts changed')
        raw=reconstruct_scientific_evidence(output,job,tasks=tasks,partition='training',expected_mace_num_workers=4,
                                            expected_made_budget=10,component_registration=registration)
        _no_risk_rows(output,request)
        require(raw==receipt['evidence'] and result.metric_name=='AUDC' and result.metric_value==raw['metric_value']
                and dict(result.costs)==raw['costs'] and result.reward_source=='made_official_orb_audc','Reported independent fitness differs from true RPC')
        for a in durable['artifacts']:
            require(file_sha256(a['path'])==a['sha256'],'Driver evidence SHA changed');evidence[a['path']]=a['sha256']
        evidence[str(path)]=file_sha256(path);evidence[str(output/'completion_receipt.json')]=file_sha256(output/'completion_receipt.json')
        slot=(request.generation,request.phase,request.candidate_index)
        require(slot not in jobs,'Duplicate training slot');jobs[slot]=(request,raw)
        costs[request.phase].update(raw['costs'])
    dev_curve=[]
    for g,record in enumerate(records,1):
        schedule=planner.plan[g-1];step=history['history'][g-1]
        fitnesses=[jobs[(g,'train',i)][1]['metric_value'] for i in (0,1)]
        require(record['generation']==g and record['fitness']==fitnesses and step['rewards']==fitnesses
                and step['seeds']==schedule['population_seeds'] and step['sigma']==record['sigma']==schedule['sigma']
                and step['alpha']==planner.config.alpha and step['generation']==g-1,'Independent ES update formula/population differs')
        require(step['normalized_rewards']==population_zscores(fitnesses,planner.config.normalization_epsilon)
                and step['ddof']==0 and step['reward_normalization']=='population_zscore','ES reward standardization changed')
        require(step['state_hash_before']==record['actual_model_state_hash_before']==markers[g-1]['actual_model_state_hash']
                and step['state_hash_after']==record['actual_model_state_hash_after']==markers[g]['actual_model_state_hash']
                and step['parameter_delta_l2']==record['actual_parameter_delta_l2'],'Actual update/hash chain differs')
        require(set(step['parameter_delta_l2'])=={p['name'] for p in manifest['full_parameter_manifest']},
                'ES update omitted parameters from its full-parameter manifest')
        require(all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in step['parameter_delta_l2'].values()),
                'Invalid actual parameter delta evidence')
        for i in (0,1):
            req,_=jobs[(g,'train',i)];population=record['population'][i]
            require(population['actual_perturbed_model_state_hash']==req.actual_model_state_hash
                    and population['results'][0]['request_id']==req.request_id,'Population evaluated wrong weights/request')
        req,raw=jobs[(g,'dev',None)]
        require(req.actual_model_state_hash==markers[g]['actual_model_state_hash'] and record['dev_mean']==raw['metric_value']
                and record['dev'][0]['request_id']==req.request_id,'Development evaluated wrong checkpoint')
        dev_curve.append({'generation':g,'AUDC':raw['metric_value'],'environment_seeds':raw['environment_seeds'],
                          'actual_model_state_hash':req.actual_model_state_hash,'parameter_delta_l2':step['parameter_delta_l2']})
    best=max(dev_curve,key=lambda r:(r['AUDC'],-r['generation']))
    require(summary['best_generation']==best['generation'] and summary['best_dev_metric']==best['AUDC']
            and summary['best_actual_model_state_hash']==best['actual_model_state_hash'] and summary['clean_reload_checked_generation']==2,
            'Selection must be complete generations1/2 on development, earliest tie')
    _proof(summary['clean_reload'],markers[2])
    require(costs['train']['candidate_oracle_attempts']==40 and costs['dev']['candidate_oracle_attempts']==20
            and _costs_match(dict(costs['train']+costs['dev']), summary['actual_evaluator_costs']),'Actual complete training/development costs differ')
    report={'verification_schema':'six_real_ES_only_B10_RPC_trajectories_v1','complete':True,
        'ablation_fingerprint':registration['fingerprint'],'run_fingerprint':planner.run_fingerprint,
        'initial_actual_model_state_hash':initial,'selected_generation':best['generation'],
        'selected_actual_model_state_hash':best['actual_model_state_hash'],'evaluations':6,
        'training':{'jobs':4,'costs':dict(costs['train'])},'development':{'jobs':2,'costs':dict(costs['dev'])},
        'candidate_oracle_attempts':60,'risk_used':False,'controller_enabled':False,'uncertainty_reward_weight':0,
        'dev_curve':dev_curve,'initial_dev_evaluated':False,'test_used':False,
        'evolution_signal_observed':any(any(v>0 for v in r['parameter_delta_l2'].values()) for r in dev_curve),
        'selected_matches_initial':best['actual_model_state_hash']==initial,
        'parameter_scope':'full','full_parameter_manifest':manifest['full_parameter_manifest'],
        'actual_parameter_delta_l2_by_generation':[history['history'][i]['parameter_delta_l2'] for i in range(2)],
        'evidence_files':[{'path':p,'sha256':s} for p,s in sorted(evidence.items())]}
    report['fingerprint']=fingerprint(report)
    return report,summary,markers


def _load_selected(policy,checkpoint,expected,config):
    optimizer=AgenticESOpt(policy.model,policy_model_id=policy.model_id+'@'+policy.revision,parameter_scope='full',
        noise_chunk_size=config.noise_chunk_size,normalization_epsilon=config.normalization_epsilon)
    optimizer.load_checkpoint(checkpoint)
    require(optimizer.model_state_hash()==expected,'Independent selected checkpoint tensor hash differs')
    return optimizer


def verify_es_only_receipt(condition_dir=None,*,registration):
    """CPU-only complete physical/ES/selection/source audit; no model loading."""
    directory=condition_directory(registration) if condition_dir is None else Path(condition_dir).resolve()
    report,summary,markers=audit_es_only(directory,registration=registration)
    receipt=read(directory/'training_receipt.json')
    require(receipt['fingerprint']==fingerprint({k:v for k,v in receipt.items() if k!='fingerprint'})
            and receipt['schema']==SCHEMA and receipt['complete'] is True and receipt['ablation_fingerprint']==registration['fingerprint'],
            'Unsealed or foreign independent ES receipt')
    require(receipt['evaluations']==6 and receipt['candidate_oracle_attempts']==60 and receipt['risk_used'] is False
            and receipt['controller_enabled'] is False and receipt['uncertainty_reward_weight']==0
            and receipt['test_used'] is False and receipt['clean_reload_passed'] is True
            and receipt['run_fingerprint']==report['run_fingerprint']
            and receipt['selected_generation']==report['selected_generation']
            and receipt['evolution_signal_observed']==report['evolution_signal_observed']
            and receipt['selected_matches_initial']==report['selected_matches_initial'],
            'Independent receipt risk/seed/count/selection fields changed')
    require(receipt['raw_training_audit']==artifact(directory/'raw_training_audit.json')
            and read(directory/'raw_training_audit.json')==report,'Independent raw cost/physics audit changed')
    best=report['selected_generation'];checkpoint=directory/'run/checkpoints'/f'generation_{best:04d}'/'policy.pt'
    require(receipt['selected_checkpoint']==artifact(checkpoint) and receipt['selected_actual_model_state_hash']==report['selected_actual_model_state_hash']
            and receipt['initial_actual_model_state_hash']==report['initial_actual_model_state_hash'],'Receipt selected a different checkpoint')
    _proof(receipt['selected_clean_reload'],markers[best]);_proof(receipt['final_clean_reload'],markers[2])
    artifacts={r['path']:r['sha256'] for r in report['evidence_files']}
    artifacts.update({str(directory/'training_receipt.json'):file_sha256(directory/'training_receipt.json'),
                      str(directory/'raw_training_audit.json'):file_sha256(directory/'raw_training_audit.json')})
    return {'independent_es_only':True,'training_method':'esopt','risk_used':False,'selected_generation':best,
        'actual_model_state_hash':report['selected_actual_model_state_hash'],'initial_actual_model_state_hash':report['initial_actual_model_state_hash'],
        'independent_training_receipt':artifact(directory/'training_receipt.json'),'training_directory':str(directory),
        'artifacts':artifacts,'evolution_signal_observed':report['evolution_signal_observed'],
        'selected_matches_initial':report['selected_matches_initial'],'run_fingerprint':report['run_fingerprint']}


def load_es_only_selected(policy,condition_dir=None,*,registration):
    condition=condition_for(registration)
    profile=verify_es_only_receipt(condition_dir,registration=registration)
    directory=Path(profile['training_directory']);best=profile['selected_generation']
    checkpoint=directory/'run/checkpoints'/f'generation_{best:04d}'/'policy.pt'
    require(policy.configuration_fingerprint==_parent_profile(registration)['policy_configuration_fingerprint'],'Final policy runtime/config differs from paired parent')
    _load_selected(policy,checkpoint,profile['actual_model_state_hash'],condition.config())
    policy.mark_state('independent_es_only_selected_reload',generation=best)
    return profile


def run_es_only(workspace,*,resume=False,plan_only=False,policy_factory=None,rollout_factory=None,
                clean_reload_validator=cpu_clean_reload_validator):
    reg=read_registration(workspace);_source_bound(reg);validate_launch(reg);condition=condition_for(reg);directory=condition_directory(reg)
    protocol=condition.protocol();config=condition.config();train,dev=condition.cases()
    if plan_only:
        base=_parent_profile(reg)
        entry=next(x for x in read(condition.project/'configs/model_manifest.json')['models'] if x['key']=='qwen35_4b')
        identity={'model_id':entry['model_id'],'model_revision':entry['revision'],
                  'initial_checkpoint_manifest_hash':base['initial_checkpoint_manifest_hash'],
                  'policy_configuration_fingerprint':base['policy_configuration_fingerprint'],
                  'callback_fingerprint':'planning_only_not_execution_identity'}
        planner=_planner(condition,identity,directory/'run')
        return {'planned_only':True,'condition':{'method':'esopt','risk_used':False},'config':asdict(config),
                'train_cases':[asdict(c) for c in train],'dev_cases':[asdict(c) for c in dev],
                'driver_rng_plan':planner.plan,'environment_seed_rule':'4*(request.environment_seed mod 2**30)+(2 if dev else 0)',
                'evaluation_seeds':[2,3,4],'evaluations':6,'candidate_oracle_attempts':60}
    from matdiscovery.policy import QwenPolicyAdapter,DecodingConfig,declared_runtime_options
    runtime=declared_runtime_options(protocol);torch.set_num_threads(runtime['torch_cpu_threads'])
    if policy_factory is None:
        from resource_guard import require_resource
        require_resource(reg)
        if torch.device(runtime['device']).type=='cuda':torch.cuda.set_per_process_memory_fraction(runtime['cuda_memory_fraction'],device=runtime['device'])
        decode=protocol['decoding']
        policy=QwenPolicyAdapter.from_verified_checkpoint(condition.project/'configs/model_manifest.json','qwen35_4b',condition.project/'data/models/qwen35_4b',
            **{k:runtime[k] for k in ('device','dtype','attn_implementation','sdpa_backend','cpu_embedding_and_lm_head','attention_checkpointing')},
            decoding=DecodingConfig(**{k:decode[k] for k in ('max_new_tokens','temperature','top_p','top_k')}),
            enable_thinking=decode['enable_thinking'],max_input_tokens=decode['max_input_tokens'])
    else:policy=policy_factory(condition)
    baseline=_parent_profile(reg);stamp=policy.model_stamp
    require(stamp.generation==0 and stamp.perturbation_seed is None and stamp.perturbation_sigma is None
            and tensor_state_hash(dict(policy.model.state_dict()))==baseline['actual_model_state_hash'],
            'Independent ES starts only at exact original theta0; old G2 is forbidden')
    require(policy.configuration_fingerprint==baseline['policy_configuration_fingerprint'],'Initial runtime/config differs from paired baseline')
    settings=RolloutSettings(max_generation_retries=protocol['risk_network']['maximum_candidate_generations_per_decision'],
        risk_threshold=protocol['risk_network']['threshold'],history_results=protocol['memory']['scientific_history_per_episode'],
        recent_tools=protocol['memory']['recent_tool_responses'],
        activation_tokens_per_decision=protocol['collection']['activation_token_rows_per_complete_decision_prefix'],failure_aware_control=False)
    callbacks=ESOnlyCallbacks(condition,policy,directory,reg,settings=settings,rollout_factory=rollout_factory)
    driver=ESTrainingDriver(policy,config,train_cases=train,dev_cases=dev,job_factory=callbacks.job_factory,
        evaluate=callbacks.evaluate,result_verifier=callbacks.result_verifier,recover_result=callbacks.recover_result,
        clean_reload_validator=clean_reload_validator,output_dir=directory/'run',callback_fingerprint=callbacks.fingerprint)
    descriptor={'ablation_fingerprint':reg['fingerprint'],'callback_identity':callbacks.callback_identity,
                'callback_fingerprint':callbacks.fingerprint,'driver_identity':driver.identity}
    publish(directory/'training_descriptor.json',descriptor)
    if (directory/'training_receipt.json').exists():
        require(resume,'Complete independent ES needs explicit read-only resume')
        return load_es_only_selected(policy,directory,registration=reg)
    if not (directory/'run/training_summary.json').exists():driver.run(resume=resume)
    else:require(resume,'Existing complete driver needs explicit resume')
    report,summary,markers=audit_es_only(directory,registration=reg)
    best=report['selected_generation'];checkpoint=directory/'run/checkpoints'/f'generation_{best:04d}'/'policy.pt'
    optimizer=_load_selected(policy,checkpoint,report['selected_actual_model_state_hash'],config)
    policy.mark_state('independent_es_only_selection',generation=best)
    proof=clean_reload_validator(policy,checkpoint,report['selected_actual_model_state_hash']);_proof(proof,markers[best])
    require(optimizer.model_state_hash()==report['selected_actual_model_state_hash'],'Reload verification mutated selected weights')
    publish(directory/'raw_training_audit.json',report)
    receipt={'schema':SCHEMA,'complete':True,'ablation_fingerprint':reg['fingerprint'],
        'initial_actual_model_state_hash':report['initial_actual_model_state_hash'],
        'selected_actual_model_state_hash':report['selected_actual_model_state_hash'],'selected_generation':best,
        'selected_checkpoint':artifact(checkpoint),'selected_clean_reload':proof,'final_clean_reload':summary['clean_reload'],
        'clean_reload_passed':True,'evaluations':6,'candidate_oracle_attempts':60,'risk_used':False,
        'uncertainty_reward_weight':0,'controller_enabled':False,'test_used':False,
        'raw_training_audit':artifact(directory/'raw_training_audit.json'),'run_fingerprint':driver.run_fingerprint,
        'evolution_signal_observed':report['evolution_signal_observed'],'selected_matches_initial':report['selected_matches_initial']}
    receipt['fingerprint']=fingerprint(receipt);publish(directory/'training_receipt.json',receipt)
    return load_es_only_selected(policy,directory,registration=reg)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--gpu-uuid');args=parser.parse_args()
    from matdiscovery.immutable_hash_cache import immutable_hash_cache
    with immutable_hash_cache():
        if args.execute:
            from resource_guard import exclusive_resource
            reg=read_registration(args.workspace)
            with exclusive_resource(reg,args.gpu_uuid):result=run_es_only(args.workspace,resume=args.resume)
        else:result=run_es_only(args.workspace,resume=args.resume,plan_only=True)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
