"""Separate component-ablation registration; no existing workspace is modified."""
from copy import deepcopy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from .accounting import file_sha256, fingerprint, write_json_atomic

SCHEMA='MADE_fixed_artifact_component_ablations_v1'
SOURCE='registered_component_ablation_evaluation'
PARENT_FP='407b6cf226a47ce6b1d52bef8e8ed617abeabb9f70d3ee9df3e4d946da0d3f95'
SCHEMA_PATCH_SHA='d8202d3a36612d62b720424ad18f8b4963b0f411cc202feb04c6188f770cb538'
ARMS={'baseline_reference':'baseline','uq_only_schema_fixed':'graph_risk','es_only_independent':'esopt'}
BUDGETS=(10,30,50)
EVALUATION_SEEDS=(2,3,4)
OLD_GUARD='type(last_stable) in (int, float) and curve[-1][1] <= last_stable <= attempt'
NEW_GUARD='type(last_stable) in (int, float) and math.isfinite(last_stable) and float(last_stable).is_integer() and 0 <= last_stable <= attempt'
_ADMITTED={}


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):
    def pairs(items):
        value={}
        for k,v in items:
            require(k not in value,'Duplicate JSON key');value[k]=v
        return value
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def artifact(path):
    p=Path(path).resolve(strict=True)
    require(p.is_file(),'Expected regular source artifact')
    return {'path':str(p),'sha256':file_sha256(p)}


def verify_artifact(value):
    require(set(value)=={'path','sha256'} and artifact(value['path'])==value,'Changed source artifact')


def code_root():return Path(__file__).resolve().parents[2]


def runtime_inventory():
    package=Path(__file__).resolve().parent
    return [{'relative_path':str(p.relative_to(package)),'sha256':file_sha256(p)} for p in sorted(package.rglob('*.py'))]


def external_inventory():
    names=('registration.py','es_only_training.py','arm_runtime.py','evaluate.py','resource_guard.py')
    return [artifact(code_root()/name) for name in names]


def parent_sweep(reference):
    verify_artifact(reference);value=read(reference['path'])
    require(value['fingerprint']==value['sweep_fingerprint']==PARENT_FP==fingerprint({k:v for k,v in value.items() if k not in ('fingerprint','sweep_fingerprint')}),'Original fixed-G2 sweep changed')
    return value


def planned_jobs(parent):
    tasks={j['task_id']:j['task'] for j in parent['new_jobs']}
    require(len(tasks)==30,'Exactly30 official parent test systems required')
    template=next(j for j in parent['new_jobs'] if j['method']=='baseline')
    jobs=[]
    for budget in BUDGETS:
        for seed in EVALUATION_SEEDS:
            for arm,route in ARMS.items():
                for task_id,task in sorted(tasks.items(),key=lambda pair:(len(pair[1]['elements']),pair[0])):
                    job=deepcopy(template)
                    for key in ('job_id','made_budget_sweep_registration','prior_all30_fingerprint'):job.pop(key,None)
                    job.update(study_id=SCHEMA,classification='post_seed1_development_component_ablation',source=SOURCE,
                        arm_id=arm,method=arm,control_route=route,task=task,task_id=task_id,group_id=task_id,
                        seed=seed,policy_sampling_seed=seed,environment_seeds=[seed],episode_ids=['0'],budget=budget,
                        expected_counts={'episodes':1,'candidate_oracle_attempts':budget,'dft_episode_attempts':0})
                    require(job['stage']=='final_eval' and job['split']=='test','Not parent test matrix')
                    job['job_id']='component-final-'+fingerprint(job)[:24];jobs.append(job)
    require(len(jobs)==len({j['job_id'] for j in jobs})==810,'Wrong component matrix')
    return jobs


def expected_training_source(original):
    old='core_protocol=None, made_all30_extension=None, made_budget_sweep=None) -> dict:'
    new='core_protocol=None, made_all30_extension=None, made_budget_sweep=None, component_registration=None) -> dict:'
    branch='''    if component_registration is not None:
        _require(core_protocol is None and made_all30_extension is None and made_budget_sweep is None, "Component scope must be exclusive")
        from .component_scope import validate_component_reconstruction
        validate_component_reconstruction(component_registration, job, expected_made_budget, partition)
    elif made_budget_sweep is not None:'''
    require(original.count(old)==1 and original.count('    if made_budget_sweep is not None:')==1 and original.count(OLD_GUARD)==1,'Original reconstruction source changed')
    return original.replace(old,new).replace('    if made_budget_sweep is not None:',branch).replace(OLD_GUARD,NEW_GUARD)


def verify_runtime_sources(value):
    package=Path(__file__).resolve().parent;parent=Path(value['parent_workspace'])/'src/matdiscovery'
    require(runtime_inventory()==value['runtime_sources'],'Executing component source changed')
    require(value['schema_patch_sha256']==SCHEMA_PATCH_SHA,'Not the reviewed schema-only controller patch')
    original={str(p.relative_to(parent)):p for p in parent.rglob('*.py')}
    require(len(original)==52 and {r['relative_path'] for r in value['runtime_sources']}==set(original)|{'component_scope.py'},'Scientific source inventory changed')
    for name,p in original.items():
        if name=='training_jobs.py':
            require((package/name).read_text()==expected_training_source(p.read_text()),'Unexpected reconstruction/math source delta')
        elif name=='failure_controller.py':
            require(file_sha256(package/name)==value['schema_patch_sha256'],'Controller is not the registered schema patch')
        else:require(file_sha256(package/name)==file_sha256(p),'Frozen scientific source changed: '+name)
    require(value['external_sources']==external_inventory(),'External entry/loader source changed')
    for name,module in list(sys.modules.items()):
        if (name=='matdiscovery' or name.startswith('matdiscovery.')) and getattr(module,'__file__',None):
            require(Path(module.__file__).resolve().is_relative_to(package),'Mixed executing scientific namespace: '+name)


def build_registration(parent_reference,workspace,*,validation,diagnostic_registration=None):
    reference=artifact(parent_reference);parent=parent_sweep(reference);workspace=Path(workspace).resolve()
    old=Path(parent['workspace']).resolve()
    require(workspace!=old and old not in workspace.parents and workspace not in old.parents,'Independent workspace required')
    validation_ref=artifact(validation);check=read(validation)
    require(check.get('passed') is True and check.get('runtime_sources')==runtime_inventory() and check.get('external_sources')==external_inventory(),'CPU validation must bind exact executing sources')
    diagnostic=artifact(diagnostic_registration) if diagnostic_registration else None
    if diagnostic:
        d=read(diagnostic['path'])
        require(d['fingerprint']==fingerprint({k:v for k,v in d.items() if k!='fingerprint'}) and d['schema']=='made_schema_repair_development_not_main_evaluation_v1','Wrong schema-only dev registration')
        require(d['patch']['sha256']==file_sha256(Path(__file__).with_name('failure_controller.py')),'Not the exact schema-fixed controller')
    value={'schema':SCHEMA,'workspace':str(workspace),'parent':reference,'parent_workspace':parent['workspace'],
        'parent_core':deepcopy(parent['parent_core']),'arms':deepcopy(ARMS),'budgets':list(BUDGETS),
        'evaluation_seeds':list(EVALUATION_SEEDS),'ES_training_seed':1,'ES_episode_budget':10,
        'ES_expected_counts':{'jobs':6,'train_jobs':4,'dev_jobs':2,'candidate_oracle_attempts':60},
        'new_jobs':planned_jobs(parent),'expected_counts':{'jobs':810,'episodes':810,'candidate_oracle_attempts':24300,'dft_episode_attempts':0},
        'baseline_reference_profile':deepcopy(parent['execution_profiles']['baseline']),
        'baseline_reference_contract':'one_shared_theta0_baseline_for_component_and_future_full_repaired_comparisons',
        'baseline_logging':'original_entropy_and_type_predictions_logging_only_never_control',
        'runtime_sources':runtime_inventory(),'external_sources':external_inventory(),
        'schema_patch_sha256':file_sha256(Path(__file__).with_name('failure_controller.py')),
        'validation':validation_ref,'diagnostic_registration':diagnostic,
        'seed1_results_used_for_development':True,'baseline_full_new_seed_results_required_for_matched_comparison':True,
        'training_and_evaluation_seeds_are_distinct_roles':True,'reuse_original_full_G2_as_ES_only':False,
        'UQ_only_weights':'exact_original_theta0','ES_only_initial_weights':'exact_original_theta0',
        'ES_only_reward':'official_AUDC_only','ES_only_UQ_or_controller':False,
        'initial_ES_dev_evaluation':False,'selection':'max_G1_G2_dev_AUDC_earliest_tie',
        'no_evolution_signal_is_valid_observed_null':True,'heldout_general_efficacy_claimed':False,'global_study_complete':False}
    value['fingerprint']=fingerprint(value);validate_registration(value)
    path=workspace/'configs/component_ablations.json';require(not path.exists(),'Preserve original registration')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    return value


def validate_registration(value):
    require(value['schema']==SCHEMA and value['fingerprint']==fingerprint({k:v for k,v in value.items() if k!='fingerprint'}),'Component registration seal changed')
    parent=parent_sweep(value['parent'])
    require(value['parent_workspace']==parent['workspace'] and value['parent_core']==parent['parent_core'],'Component ancestry changed')
    require(value['arms']==ARMS and value['budgets']==list(BUDGETS) and value['evaluation_seeds']==list(EVALUATION_SEEDS) and value['new_jobs']==planned_jobs(parent),'Exact810 evaluation scope changed')
    require(value['ES_training_seed']==1 and value['ES_episode_budget']==10 and value['ES_expected_counts']=={'jobs':6,'train_jobs':4,'dev_jobs':2,'candidate_oracle_attempts':60},'Independent ES scope changed')
    require(value['expected_counts']=={'jobs':810,'episodes':810,'candidate_oracle_attempts':24300,'dft_episode_attempts':0},'Ablation costs changed')
    require(value['baseline_reference_profile']==parent['execution_profiles']['baseline']
            and value['baseline_reference_contract']=='one_shared_theta0_baseline_for_component_and_future_full_repaired_comparisons'
            and value['baseline_logging']=='original_entropy_and_type_predictions_logging_only_never_control','Shared baseline profile or reuse contract changed')
    require(value['reuse_original_full_G2_as_ES_only'] is False and value['ES_only_UQ_or_controller'] is False and value['ES_only_reward']=='official_AUDC_only','Independent ES semantics changed')
    fixed={'seed1_results_used_for_development':True,'baseline_full_new_seed_results_required_for_matched_comparison':True,
           'training_and_evaluation_seeds_are_distinct_roles':True,'UQ_only_weights':'exact_original_theta0',
           'ES_only_initial_weights':'exact_original_theta0','initial_ES_dev_evaluation':False,
           'selection':'max_G1_G2_dev_AUDC_earliest_tie','no_evolution_signal_is_valid_observed_null':True,
           'heldout_general_efficacy_claimed':False,'global_study_complete':False}
    require(all(value.get(k)==v and type(value.get(k)) is type(v) for k,v in fixed.items()),'Registered scientific interpretation/selection changed')
    verify_runtime_sources(value);verify_artifact(value['validation'])
    check=read(value['validation']['path'])
    require(check.get('passed') is True and check.get('runtime_sources')==value['runtime_sources'] and check.get('external_sources')==value['external_sources'],'Wrong implementation validation')
    if value['diagnostic_registration']:verify_artifact(value['diagnostic_registration'])
    return value


def read_registration(workspace):
    path=Path(workspace).resolve();path=path/'configs/component_ablations.json' if path.is_dir() else path
    value=validate_registration(read(path))
    require(path==Path(value['workspace'])/'configs/component_ablations.json','Registration workspace path differs')
    return value


def _deep_parent_validation(value):
    folder=Path(value['parent_workspace'])/'src/matdiscovery';name='_component_original_'+PARENT_FP[:12]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    scope=importlib.import_module(name+'.made_budget_sweep');cache=importlib.import_module(name+'.immutable_hash_cache')
    with cache.immutable_hash_cache():
        parent=scope.read_sweep(value['parent_workspace']);proof=scope.validate_launch(parent);core=scope.admitted_parent(parent)
    return core,proof


def validate_launch(value):
    value=validate_registration(value);key=(os.getpid(),value['fingerprint'])
    if key not in _ADMITTED:
        core,proof=_deep_parent_validation(value)
        require(proof['complete'] is True and proof['sweep_fingerprint']==PARENT_FP,'Actual original admission failed')
        _ADMITTED[key]={'core':deepcopy(core),'proof':deepcopy(proof)}
    return {'complete':True,'component_fingerprint':value['fingerprint'],'parent_proof':deepcopy(_ADMITTED[key]['proof'])}


def admitted_parent(value):
    validate_registration(value);entry=_ADMITTED.get((os.getpid(),value['fingerprint']))
    require(entry is not None,'Component scope requires actual deep admission in this PID')
    return deepcopy(entry['core'])


def load_es_module(value):
    admitted_parent(value)
    refs=[r for r in value['external_sources'] if Path(r['path']).name=='es_only_training.py']
    require(len(refs)==1,'One exact ES-only loader source required');reference=refs[0];verify_artifact(reference)
    name='_component_es_'+reference['sha256'][:12]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,reference['path']);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def validate_evaluation_job(value,job,budget,*,execution=False):
    admitted_parent(value);matches=[j for j in value['new_jobs'] if j['job_id']==job.get('job_id')]
    require(len(matches)==1 and type(budget) is int and budget in BUDGETS,'Unknown component evaluation job/budget')
    expected=matches[0]
    require(expected['budget']==budget and (all(job.get(k)==v for k,v in expected.items()) if execution else job==expected),'Component job/seed/arm/budget changed')
    return expected


def validate_component_reconstruction(value,job,budget,partition):
    if partition=='training':return load_es_module(value).validate_training_job_scope(value,job,budget)
    require(partition=='final_test','Unknown component partition')
    return validate_evaluation_job(value,job,budget,execution=True)
