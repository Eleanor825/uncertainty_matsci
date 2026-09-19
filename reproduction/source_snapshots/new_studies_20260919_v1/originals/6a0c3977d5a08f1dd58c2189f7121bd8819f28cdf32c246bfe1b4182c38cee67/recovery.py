"""Explicit zero-query recovery through the original locked training body.

The frozen package and original invocation remain unchanged. This adds source
lineage to original callback inputs, not a new scientific method or job matrix.
"""
from pathlib import Path
import argparse,fcntl,os,socket,time
import common as c

SCHEMA='support_full_es_locked_body_zero_query_recovery_v1'
SOURCE_NAMES=('common.py','corrected_loader.py','cpu_check_v2.py','recovery.py')
REGISTRATION=c.ROOT/'recovery_registration.json'
RUN=c.ROOT/'runs/recovery_v1'

def sources():return [c.artifact(c.ROOT/name) for name in SOURCE_NAMES]

def cpu_proof(path,sha):
    ref={'path':str(path),'sha256':sha};c.verify(ref);v=c.read(path);c.check_seal(v)
    c.require(v['schema']=='actual_support_controller_CPU_load_v1' and v['complete'] is True and v['passed'] is True
        and v['registration']==c.REG and v['support_fingerprint']==c.FP and v['original_training_source']==c.ORIGINAL_ES
        and v['corrected_loader']==c.artifact(c.ROOT/'corrected_loader.py')
        and v['checker']==c.artifact(c.ROOT/'cpu_check_v2.py') and v['common_source']==c.artifact(c.ROOT/'common.py'),
        'Real original-source CPU controller load proof required')
    c.require(v['import_change_only'] is True and v['loaded_transcoders']==32 and len(v['layers'])==32
        and v['NN_TC_loaded_on_CPU'] is True and v['CUDA_initialized'] is False
        and v['new_model_generation_calls']==v['new_oracle_calls']==0 and v['control_ready'] is True
        and v['production_runtime_provenance_view_only'] is True and v['policy_forward_or_FD_pass_claimed'] is False
        and v['complete_training_launch_admission_claimed'] is False,
        'Incomplete controller test or overstated policy/FD evidence')
    c.require([x['module_path'] for x in v['layers']]==[f'model.language_model.layers.{i}.mlp' for i in range(32)]
        and all(x['fidelity_gate_passed'] is True and 0<=x['dev_output_fvu']<=.5 for x in v['layers']),
        'CPU proof omitted a layer or changed fidelity gates')
    c.verify(v['bank_manifest']);return v

def prepare(path,sha):
    c.require(Path(__file__).resolve()==c.ROOT/'recovery.py','Use source-bound deployed recovery entry')
    c.original_and_loader();proof=cpu_proof(path,sha);failure=c.failure_evidence(require_unstarted=True)
    value=c.seal({'schema':SCHEMA,'source_files':sources(),'original_registration':c.REG,
        'support_fingerprint':c.FP,'original_training_source':c.ORIGINAL_ES,
        'CPU_controller_admission':{'path':str(path),'sha256':sha},'original_failure_evidence':failure,
        'workspace':str(c.WORKSPACE),'recovery_output':str(RUN),
        'original_locked_entry':'_run_full_es_locked(resume=False, controller_loader=registered_controller_loader)',
        'original_training_execution_mutex':'experiments/full_es/.execution.lock',
        'fresh_driver_not_checkpoint_resume':True,
        'loader_change':{'from':'matdiscovery.core_collection.corpus_contract','to':'matdiscovery.core_protocol.corpus_contract'},
        'original_invocation_preserved':True,'original_failure_preserved':True,'automatic_physical_replay':False,
        'requires_original_GPU_guard_and_training_mutex':True,'runtime_or_controller_math_changed':False,
        'NN_refit':False,'same_registered_UQ_270_shared_without_replay':True,
        'same_registered_full_270_not_additional_jobs':True,
        'unchanged_ES_expected_counts':{'jobs':6,'train_jobs':4,'dev_jobs':2,'candidate_oracle_attempts':60},
        'unchanged_evaluation_counts':{'jobs':540,'episodes':540,'candidate_oracle_attempts':16200},
        'shared_component_registration_fingerprint':'216fa4f7df3dfc25b66d4e00a2e4788c4f2a2b3c2f8d563eae231e892cdcbed6',
        'source_lineage_delivery':'append recovery source/amendment/actual loader receipt/invocation to original callback auxiliary files',
        'created_at':time.time()})
    c.once(REGISTRATION,value);return {'registration':c.artifact(REGISTRATION),'fingerprint':value['fingerprint'],'scientific_calls':0}

def registration(path,sha):
    c.require(Path(path)==REGISTRATION,'Exact recovery registration path required')
    c.verify({'path':str(path),'sha256':sha});v=c.read(path);c.check_seal(v)
    c.require(v['schema']==SCHEMA and v['source_files']==sources() and v['original_registration']==c.REG
        and v['support_fingerprint']==c.FP and v['original_training_source']==c.ORIGINAL_ES
        and v['workspace']==str(c.WORKSPACE) and v['recovery_output']==str(RUN)
        and v['original_invocation_preserved'] is True and v['original_failure_preserved'] is True
        and v['automatic_physical_replay'] is False and v['requires_original_GPU_guard_and_training_mutex'] is True
        and v['runtime_or_controller_math_changed'] is False and v['NN_refit'] is False,
        'Recovery source/scope/one-shot contract differs')
    c.require(v['original_locked_entry']=='_run_full_es_locked(resume=False, controller_loader=registered_controller_loader)'
        and v['original_training_execution_mutex']=='experiments/full_es/.execution.lock'
        and v['fresh_driver_not_checkpoint_resume'] is True
        and v['loader_change']=={'from':'matdiscovery.core_collection.corpus_contract','to':'matdiscovery.core_protocol.corpus_contract'}
        and v['unchanged_ES_expected_counts']=={'jobs':6,'train_jobs':4,'dev_jobs':2,'candidate_oracle_attempts':60}
        and v['unchanged_evaluation_counts']=={'jobs':540,'episodes':540,'candidate_oracle_attempts':16200}
        and v['same_registered_UQ_270_shared_without_replay'] is True and v['same_registered_full_270_not_additional_jobs'] is True,
        'Recovery changed original physics budget or UQ sharing')
    c.require(v['shared_component_registration_fingerprint']=='216fa4f7df3dfc25b66d4e00a2e4788c4f2a2b3c2f8d563eae231e892cdcbed6'
        and v['source_lineage_delivery']=='append recovery source/amendment/actual loader receipt/invocation to original callback auxiliary files',
        'Shared references or source-lineage contract differs')
    c.original_and_loader();cpu_proof(**{'path':v['CPU_controller_admission']['path'],'sha':v['CPU_controller_admission']['sha256']})
    c.require(v['original_failure_evidence']==c.failure_evidence(require_unstarted=False),'Original failure lineage differs')
    return v

def lineage_paths(v):
    return [*(Path(x['path']) for x in v['source_files']),REGISTRATION,
        Path(v['CPU_controller_admission']['path']),RUN/'invocation.json',RUN/'actual_controller_loaded.json']

def registered_controller_loader(original,corrected,v):
    proof=cpu_proof(v['CPU_controller_admission']['path'],v['CPU_controller_admission']['sha256'])
    def load(condition,policy,reg):
        c.require(reg['fingerprint']==c.FP and v['source_files']==sources(),'Hook source/registration changed')
        risk,attributor,auxiliary=corrected.load_controllers(condition,policy,reg)
        from matdiscovery.esopt import tensor_state_hash
        layer_hashes=[tensor_state_hash(dict(x.transcoder.state_dict())) for x in attributor.bindings]
        c.require(layer_hashes==[x['state_hash'] for x in proof['layers']]
            and tensor_state_hash(dict(risk.model.state_dict()))==proof['scalar_state_hash']
            and tensor_state_hash(dict(risk.typed_model.state_dict()))==proof['typed_state_hash']
            and float(risk.primary.temperature)==proof['scalar_temperature'] and risk.temperatures==proof['type_temperatures']
            and attributor.source_selection_rule==proof['source_selection_rule'],'Actual controllers differ from CPU-approved original weights/config')
        c.once(RUN/'actual_controller_loaded.json',c.seal({'schema':'support_full_ES_registered_controller_hook_executed_v1',
            'complete':True,'recovery_fingerprint':v['fingerprint'],'registration':c.REG,'actual_policy_checkpoint_hash':policy.checkpoint_hash,
            'actual_policy_configuration_fingerprint':policy.configuration_fingerprint,
            'actual_policy_runtime':policy.runtime_precision_record(),'loaded_transcoders':32,
            'CPU_controller_admission':v['CPU_controller_admission'],'source_files':v['source_files'],
            'original_auxiliary_file_count':len(auxiliary),'native_FD_gate_still_required_by_original_rollout':True,
            'time':time.time()}))
        # The unmodified TrainingJobCallbacks hashes every additional file into
        # its callback identity; original audit/selected receipts recheck them.
        return risk,attributor,[*auxiliary,*lineage_paths(v)]
    return load

def verify_lineage(v,profile):
    from matdiscovery.accounting import file_sha256
    expected={str(p.resolve()):file_sha256(p) for p in lineage_paths(v)}
    c.require(all(profile['artifacts'].get(p)==sha for p,sha in expected.items()),'Original selected proof omitted recovery source lineage')
    descriptor=c.read(c.WORKSPACE/'experiments/full_es/training_descriptor.json')
    c.require(all(descriptor['callback_identity']['inputs'].get(p)==sha for p,sha in expected.items()),'Original callback did not bind registered recovery inputs')
    return expected

def start_original_body(original,corrected,v,gpu_uuid):
    """Original invocation remains evidence; only an entirely unstarted driver may enter."""
    folder=c.WORKSPACE/'experiments/full_es'
    with (folder/'.execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.failure_evidence(require_unstarted=True)
        c.once(RUN/'invocation.json',{'schema':'support_full_ES_zero_query_recovery_invocation_v1',
            'recovery_fingerprint':v['fingerprint'],'source_files':sources(),'original_invocation':c.FAILURE_REFS['invocation'],
            'pid':os.getpid(),'hostname':socket.gethostname(),'gpu_uuid':gpu_uuid,'started_at':time.time(),
            'automatic_physical_replay':False,'original_invocation_preserved':True,'fresh_driver_not_checkpoint_resume':True})
        # The original public wrapper couples invocation resume and checkpoint
        # resume. Here no checkpoint/run exists. Own its exact mutex and invoke
        # its unchanged locked body with a fresh driver; do not fake a commit.
        return original._run_full_es_locked(c.WORKSPACE,resume=False,
            controller_loader=registered_controller_loader(original,corrected,v))

def execute(v,gpu_uuid):
    c.require(Path(os.sys.executable).resolve()==c.POLICY.resolve(),'Original policy interpreter required')
    original,corrected=c.original_and_loader()
    from matdiscovery.immutable_hash_cache import immutable_hash_cache
    from resource_guard import exclusive_resource
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'.recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.require(not (RUN/'attempt.json').exists() and not (RUN/'invocation.json').exists()
            and not (RUN/'failure.json').exists(),'Recovery already attempted; no automatic replay')
        c.once(RUN/'attempt.json',{'schema':'support_full_ES_recovery_attempt_v1','recovery_fingerprint':v['fingerprint'],
            'source_files':sources(),'pid':os.getpid(),'hostname':socket.gethostname(),'time':time.time(),
            'automatic_physical_replay':False})
        with immutable_hash_cache():
            reg=original.read_registration(c.WORKSPACE);original.validate_launch(reg)
            with exclusive_resource(reg,gpu_uuid):
                profile=start_original_body(original,corrected,v,gpu_uuid)
                lineage=verify_lineage(v,profile)
                result=c.seal({'schema':'support_full_ES_zero_query_recovery_completed_v1','complete':True,
                    'recovery_fingerprint':v['fingerprint'],'original_registration':c.REG,'selected_profile':profile,
                    'recovery_inputs_in_original_selected_artifacts':lineage,'original_failure_preserved':True,
                    'finished_at':time.time()})
                c.once(RUN/'completion.json',result);return result

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('prepare','check','execute','audit'))
    a.add_argument('--cpu-proof',type=Path);a.add_argument('--cpu-proof-sha256');a.add_argument('--registration',type=Path)
    a.add_argument('--expected-sha256');a.add_argument('--gpu-uuid');args=a.parse_args()
    if args.mode=='prepare':result=prepare(args.cpu_proof,args.cpu_proof_sha256)
    else:
        v=registration(args.registration,args.expected_sha256)
        if args.mode=='check':result={'complete':True,'failure':c.failure_evidence(require_unstarted=True),'source_bound':True,'scientific_calls':0}
        elif args.mode=='audit':
            original,_=c.original_and_loader()
            from matdiscovery.immutable_hash_cache import immutable_hash_cache
            with immutable_hash_cache():
                reg=original.read_registration(c.WORKSPACE);original.validate_launch(reg)
                profile=original.verify_full_es_receipt(registration=reg);result={'complete':True,'profile':profile,'lineage':verify_lineage(v,profile)}
        else:
            try:result=execute(v,args.gpu_uuid)
            except BaseException as error:
                c.once(RUN/'failure.json',{'complete':False,'error_type':type(error).__name__,'error':str(error),
                    'automatic_physical_replay':False,'time':time.time()});raise
    print(c.digest(result),flush=True)

if __name__=='__main__':main()
