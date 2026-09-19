"""Real frozen NN/32-TC loader admission on CPU; no policy forward or oracle.

The official Qwen architecture is constructed on meta solely to check module
structure. Production identity/runtime are an explicitly declared provenance
view, not a claim that a CUDA policy ran in this process. All auxiliary weights
and their original source/corpus/metadata gates are actually loaded and checked.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,os,resource,time
import common as c

def run(output):
    c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU admission must hide CUDA')
    started=time.time();original,loader=c.original_and_loader();failure=c.failure_evidence(require_unstarted=True)
    import torch
    from transformers import AutoConfig,AutoModelForImageTextToText
    from matdiscovery.immutable_hash_cache import immutable_hash_cache
    from matdiscovery.policy import verify_checkpoint,QwenPolicyAdapter
    from matdiscovery.core_representation import paths_for
    from matdiscovery.esopt import tensor_state_hash
    torch.set_num_threads(2)
    with immutable_hash_cache() as cache:
        print('CPU: original source and core/corpus admission; training launch admission remains separate',flush=True)
        reg=original.read_registration(c.WORKSPACE)
        from matdiscovery.core_protocol import read_core
        parent=reg['parent_core'];c.verify({'path':parent['path'],'sha256':parent['sha256']})
        core=read_core(Path(parent['path']).parent.parent)
        c.require(core['fingerprint']==parent['core_fingerprint'],'Wrong original controller core')
        condition=original.FullESCondition(Path(core['workspace']).resolve(),core)
        print('CPU: original core admitted; verify exact checkpoint bytes',flush=True)
        base=original._parent_profile(reg);bank_path=paths_for(condition.core)['bank']/'transcoder_manifest.json';bank=c.read(bank_path)
        checkpoint=verify_checkpoint(condition.project/'configs/model_manifest.json','qwen35_4b',condition.project/'data/models/qwen35_4b')
        c.require(checkpoint.checkpoint_hash==base['initial_checkpoint_manifest_hash']==bank['checkpoint_hash']
            and bank['policy_configuration_fingerprint']==base['policy_configuration_fingerprint'],'Actual checkpoint/registered provenance differs')
        config=AutoConfig.from_pretrained(str(checkpoint.directory),local_files_only=True,trust_remote_code=False)
        with torch.device('meta'):
            model=AutoModelForImageTextToText.from_config(config,attn_implementation='eager')
        model.eval();view=SimpleNamespace(model=model,entry=checkpoint.entry)
        QwenPolicyAdapter._validate_structure(view)
        view.architecture_review=QwenPolicyAdapter.architecture_review.fget(view)
        view.checkpoint_hash=checkpoint.checkpoint_hash
        view.configuration_fingerprint=base['policy_configuration_fingerprint']
        view.runtime_precision_record=lambda:bank['policy_runtime']
        view.get_state_id=lambda:'CPU_controller_constructor_only_no_policy_forward'
        print('CPU: official meta structure checked; load actual NN and all32 TC via frozen production gates',flush=True)
        risk,attributor,files=loader.load_controllers(condition,view,reg)
        c.require(len(attributor.bindings)==32 and risk.control_ready is True,'Incomplete real auxiliary admission')
        c.require(all(p.device.type=='cpu' for module in (risk.model,risk.typed_model)
            for p in module.parameters()),'NN unexpectedly placed outside CPU')
        layers=[]
        for binding in attributor.bindings:
            c.require(all(p.device.type=='cpu' for p in binding.transcoder.parameters()),'TC unexpectedly outside CPU')
            layers.append({'module_path':binding.module_path,'state_hash':tensor_state_hash(dict(binding.transcoder.state_dict())),
                'fidelity_gate_passed':binding.training_metadata['fidelity_gate_passed'],
                'dev_output_fvu':binding.training_metadata['dev']['output_fvu']})
        c.require(all(x['fidelity_gate_passed'] is True and x['dev_output_fvu']<=.5 for x in layers),'Original FVU admission changed')
        c.require(not torch.cuda.is_initialized(),'CPU verification initialized CUDA')
        proof={'schema':'actual_support_controller_CPU_load_v1','complete':True,'passed':True,'registration':c.REG,
            'support_fingerprint':c.FP,'original_training_source':c.ORIGINAL_ES,
            'corrected_loader':c.artifact(c.ROOT/'corrected_loader.py'),'checker':c.artifact(__file__),
            'common_source':c.artifact(c.ROOT/'common.py'),'failure_evidence':failure,
            'import_change_only':True,'checkpoint_hash':checkpoint.checkpoint_hash,
            'bank_manifest':c.artifact(bank_path),'layers':layers,'loaded_transcoders':32,
            'scalar_state_hash':tensor_state_hash(dict(risk.model.state_dict())),
            'typed_state_hash':tensor_state_hash(dict(risk.typed_model.state_dict())),
            'scalar_temperature':float(risk.primary.temperature),'type_temperatures':risk.temperatures,
            'head_availability':risk.head_availability,'control_ready':risk.control_ready,
            'source_selection_rule':attributor.source_selection_rule,'source_selection_proof':attributor.source_selection_proof,
            'auxiliary_files':[original.artifact(p) for p in sorted(set(files))],
            'policy_structure':'official full32 meta model from verified config; no policy weights materialized',
            'production_runtime_provenance_view_only':True,'complete_training_launch_admission_claimed':False,'policy_forward_or_FD_pass_claimed':False,
            'NN_TC_loaded_on_CPU':True,'CUDA_initialized':False,'new_model_generation_calls':0,'new_oracle_calls':0,
            'started_at':started,'finished_at':time.time(),'elapsed_seconds':time.time()-started,
            'max_rss_KiB_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'hash_cache':dict(cache)}
        c.once(output,c.seal(proof));return proof

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:
        proof=run(a.output)
        print({'passed':proof['passed'],'loaded_transcoders':proof['loaded_transcoders'],'elapsed_seconds':proof['elapsed_seconds']},flush=True)
    except BaseException as error:
        c.once(a.output.with_name(a.output.stem+'.failure.json'),{'passed':False,'error_type':type(error).__name__,'error':str(error),
            'no_automatic_retry':True,'time':time.time()})
        raise
