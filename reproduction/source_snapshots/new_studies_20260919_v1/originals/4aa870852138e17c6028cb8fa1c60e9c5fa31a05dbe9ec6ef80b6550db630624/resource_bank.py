"""Require the explicit resource-sharing provenance on the final mathematical bank."""
from pathlib import Path
import common as c

AMENDMENT_SHA='309be8aac3e1fd09c40e3749d54dceae423db3729d24661338306f9800a7a06b'
AMENDMENT_FP='860feef3b5435c1b1b2a5fc40c00e7caa4378c5ab91a09447fd71e018b49a9c9'


def amendment(ref,tc):
    c.require(ref['sha256']==AMENDMENT_SHA,'Unregistered resource-sharing amendment')
    value=c.read(c.check(ref));c.verify(value)
    c.require(value['fingerprint']==AMENDMENT_FP,'Resource-sharing fingerprint differs')
    c.require(value['schema']=='summit_snar_shared_one_TC_resource_amendment_v1'
        and value['registry']==c.artifact(Path(tc['workspace'])/'registration.json')
        and value['registry_fingerprint']==tc['fingerprint'] and value['scientific_numerics_unchanged'] is True
        and value['formal_queue_layers']==32 and value['old_exclusive_leases_acquired'] is False
        and value['old_FDs_inherited'] is False and value['parent_loads_Qwen'] is False
        and value['downstream_requires_resource_bound_bank'] is True,'Wrong explicit TC sharing admission')
    for source in value['source_files']:c.check(source)
    producers=[x for x in value['source_files'] if Path(x['path']).name=='shared_tc_worker.py']
    c.require(len(producers)==1,'Missing unique source-bound resource producer')
    producer=c.load_module('_snar_original_shared_tc_producer',producers[0]['path'])
    c.require(producer.admission(ref['path'],ref['sha256'])==value,'Original resource producer rejects its amendment')
    return value,producers[0]


def load(reg,tc):
    root=Path(tc['workspace']);amend,producer=amendment(reg['tc_resource_amendment'],tc)
    terminal_path=root/'training_complete_with_shared_resources.json';terminal=c.read(terminal_path);c.verify(terminal)
    c.require(terminal['schema']=='snar_resource_bound_training_completion_v1' and terminal['complete'] is True
        and terminal['amendment']==reg['tc_resource_amendment'] and terminal['downstream_must_use_this_bank'] is True
        and terminal['main_experiment_complete'] is False and terminal['passed_layers']==32
        and terminal['pooled_development_fidelity_passed'] is True,'Resource-bound training did not pass')
    c.require(terminal['bank']['path']==str(root/'bank/transcoder_manifest_with_shared_resource_evidence.json')
        and terminal['original_training_completion']==c.artifact(root/'training_complete.json'),
        'Wrong declared resource bank or original training completion')
    original_completion=c.read(c.check(terminal['original_training_completion']));c.verify(original_completion)
    c.require(original_completion['registry_fingerprint']==tc['fingerprint'] and original_completion['passed_layers']==32
        and original_completion['pooled_development_fidelity_passed'] is True,'Mathematical training completion differs')
    original=c.read(c.check(original_completion['bank']));bank=c.read(c.check(terminal['bank']))
    c.require(bank['bank_fingerprint']==c.digest({k:v for k,v in bank.items() if k!='bank_fingerprint'}),'Derived bank fingerprint differs')
    shared=[]
    for i in range(32):
        d=root/'layers'/f'layer{i:02d}';claim=c.read(d/'claim.json');c.verify(claim)
        if claim['worker_id']!='secondary_shared_one_tc_v1':continue
        ref=c.artifact(d/'shared_resource_execution.json');execution=c.read(ref['path']);c.verify(execution)
        c.require(execution['schema']=='snar_shared_layer_resource_evidence_v1' and execution['complete'] is True
            and execution['registry_fingerprint']==tc['fingerprint'] and execution['layer_index']==i
            and execution['claim_token']==claim['claim_token'] and execution['amendment']==reg['tc_resource_amendment']
            and execution['producer']==producer and execution['terminal']==c.artifact(d/'complete.json')
            and execution['returncode']==0 and execution['scientific_numerics_changed'] is False
            and execution['new_oracle_calls']==execution['new_generation_calls']==0
            and execution['old_leases_held'] is False and execution['old_FDs_inherited'] is False
            and not execution.get('interrupt_reasons'),'Shared layer execution is incomplete or inconsistent')
        shared.append(ref)
    c.require(terminal['shared_layers']==len(shared) and bank['shared_layer_resource_receipts']==shared
        and bank['original_mathematical_bank']==original_completion['bank']
        and bank['resource_sharing_amendment']==reg['tc_resource_amendment'] and bank['resource_producer']==producer,
        'Resource layer list/source differs')
    extra={'bank_fingerprint','original_mathematical_bank','resource_sharing_amendment','resource_producer',
           'shared_layer_resource_receipts','resource_mode_note'}
    c.require({k:v for k,v in bank.items() if k not in extra}=={k:v for k,v in original.items() if k!='bank_fingerprint'},
              'Resource provenance changed mathematical bank/weights/configuration')
    return terminal['bank'],c.artifact(terminal_path)
