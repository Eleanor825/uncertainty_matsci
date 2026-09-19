#!/usr/bin/env python3
"""Fresh causal128 normalized64 bank: independent claims, unchanged TC worker.

Only new TC children are launched. No Qwen model, oracle, generation, graph,
NN, or ES is run here. Failed/unknown claims are never recycled automatically.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
import uuid

import registration as r
GIB=1024**3


def modules(reg):
    sys.dont_write_bytecode=True
    sys.path[:0]=[reg['original_source_root'],reg['method_source_root']]
    worker=importlib.import_module('method.tc_worker')
    importlib.import_module('matdiscovery.normalized_transcoders')
    r.require(Path(worker.__file__).resolve()==Path(reg['original_source_root'])/'method/tc_worker.py',
              'Wrong original worker imported')
    for name in ('normalized_transcoders','transcoders','esopt','representation_training'):
        module=importlib.import_module('matdiscovery.'+name)
        r.require(Path(module.__file__).resolve()==Path(reg['method_source_root'])/'matdiscovery'/f'{name}.py',
                  'Wrong frozen numerical module imported')
    return worker


def layer_inputs(reg,index,*,verify=True):
    r.require(type(index) is int and 0<=index<32,'Layer outside exact32 scope')
    row=r.read(reg['input_manifest']['path'])['layers'][index]
    r.require(row['layer_index']==index and row['layer_path']==r.LAYER_PATHS[index],'Wrong layer identity')
    result={}
    for split in ('train','dev'):
        result[split]=[]
        for item in row[split]:
            if verify:r.check(item['shard']);r.check(item['sidecar'])
            result[split].append(item['shard']['path'])
    return result


def claim_layer(reg,index,worker_id):
    """O_EXCL precedes any checkpoint or CUDA allocation; no lease expiration."""
    directory=Path(reg['workspace'])/'layers'/f'layer{index:02d}'
    directory.mkdir(parents=True,exist_ok=True)
    value=r.sealed(dict(schema='snar_causal128_layer_claim_v1',registry_fingerprint=reg['fingerprint'],
        layer_index=index,layer_path=r.LAYER_PATHS[index],worker_id=worker_id,hostname=socket.gethostname(),
        pid=os.getpid(),claim_token=uuid.uuid4().hex,claimed_at=time.time(),automatic_retry=False))
    try:r.publish(directory/'claim.json',value)
    except FileExistsError:return None
    r.require(not any((directory/name).exists() for name in ('request.json','receipt.json','failure.json'))
              and not (Path(reg['workspace'])/'bank'/f'layer_{index:02d}.pt').exists(),
              'Unexpected prior output without claim: reconciliation required')
    return value


def claim_next(reg,worker_id):
    for i in range(32):
        claim=claim_layer(reg,i,worker_id)
        if claim is not None:return claim
    return None


def queue_status(reg):
    completed=[];failed=[];unknown=[];unclaimed=[]
    for i in range(32):
        d=Path(reg['workspace'])/'layers'/f'layer{i:02d}'
        if not (d/'claim.json').exists():unclaimed.append(i);continue
        if (d/'complete.json').exists():
            rec=r.read(d/'complete.json');r.verify_seal(rec)
            r.require(rec['registry_fingerprint']==reg['fingerprint'] and rec['layer_index']==i,'Foreign layer completion')
            (completed if rec['fidelity_gate_passed'] else failed).append(i)
        elif (d/'failure.json').exists():failed.append(i)
        else:unknown.append(i)
    return dict(completed_layers=completed,failed_layers=failed,claimed_without_terminal=unknown,unclaimed_layers=unclaimed,
                all_terminal=len(completed)+len(failed)==32,main_experiment_complete=False)


def lease_paths(reg,gpu_uuid):
    r.require(re.fullmatch(r'GPU-[A-Za-z0-9-]+',gpu_uuid) is not None,'Invalid physical GPU UUID')
    prior=Path(reg['method_source_root']).resolve().parent
    r.require(prior.name=='made_budget_sweep_v1' and prior.parent.name=='evaluation_workspaces',
              'GPU leases must belong to the source-bound original MADE budget workspace')
    project=prior.parent.parent
    return [prior/'locks'/('gpu-'+r.digest(gpu_uuid)+'.lock'),
            project/'technical_not_main/budget_gpu_leases'/(gpu_uuid+'.lock')]


def lease_evidence(reg,gpu_uuid,fds):
    """Prove the two exact inherited flocks without relying on FUSE fdinfo.

    Re-locking the same open file description retains ownership. A different
    open description must conflict. Close only that probe; never unlock the
    inherited descriptions, including on validation failure.
    """
    r.require(len(fds)==2 and len(set(fds))==2,'Exactly two inherited GPU/resource lease descriptors are required')
    refs=[]
    for fd,path in zip(fds,lease_paths(reg,gpu_uuid)):
        st=os.fstat(fd);actual=path.stat()
        identity=lambda value:(value.st_dev,value.st_ino)
        r.require(stat.S_ISREG(st.st_mode) and identity(st)==identity(actual),
                  'Inherited lease descriptor differs from the exact expected lock inode')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        probe=os.open(path,os.O_RDWR)
        blocked=False
        try:
            r.require(identity(os.fstat(probe))==identity(st),'Lease path changed while probing')
            try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:blocked=True
            r.require(blocked,'Independent lease open did not conflict; exclusive ownership unproved')
        finally:os.close(probe)
        r.require(identity(path.stat())==identity(st),'Lease path changed after probing')
        refs.append(dict(fd=fd,path=str(path),device=st.st_dev,inode=st.st_ino,
                         exclusive_relock_passed=True,independent_open_blocked=True))
    r.require(len({(x['device'],x['inode']) for x in refs})==2,'Lease descriptors alias the same lock')
    return refs


def resource_snapshot(reg,gpu_uuid):
    worker=modules(reg)
    path=Path(reg['original_source_root'])/'runtime/resource_guard.py'
    spec=importlib.util.spec_from_file_location('_snar_v3_original_resource_guard',path)
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    return dict(gpu=worker.gpu_snapshot(gpu_uuid),ram=guard.cgroup_memory(),
                cpu_affinity_count=len(os.sched_getaffinity(0)),observed_at=time.time())


def require_capacity(snapshot,parallel):
    r.require(type(parallel) is int and 1<=parallel<=8,'Maximum8 original isolated TC children')
    # Conservative admission only. Actual per-child original 2GiB allocator is authoritative.
    gpu_min=max(12,2*parallel+8)*GIB;ram_min=max(8,4*parallel)*GIB
    r.require(snapshot['gpu']['free_bytes']>=gpu_min,'Insufficient GPU headroom for this TC concurrency')
    r.require(snapshot['ram']['direct_headroom_bytes']>=ram_min,'Insufficient direct cgroup RAM headroom')
    r.require(snapshot['cpu_affinity_count']>=2*parallel,'Insufficient CPU affinity for two threads per TC child')


def make_layer_request(reg,claim,gpu_uuid):
    worker=modules(reg);index=claim['layer_index'];paths=layer_inputs(reg,index)
    from matdiscovery.transcoders import TranscoderConfig
    return worker.make_request(TranscoderConfig(**r.ARCHITECTURE),paths['train'],paths['dev'],
        policy_fingerprint=reg['checkpoint_hash'],layer_path=r.LAYER_PATHS[index],
        output_path=Path(reg['workspace'])/'bank'/f'layer_{index:02d}.pt',seed=1729+index,
        registration=r.fit_registration(reg),expected_gpu_uuid=gpu_uuid,device='cuda:0',epochs=64,
        batch_size=256,learning_rate=.0004,max_dev_fvu=.5,cpu_threads=2,gpu_memory_bytes=2*GIB,
        minimum_free_bytes=12*GIB,unit_test=False)


def validate_layer(reg,index,*,unit_fixture=False):
    """CPU-only actual raw checkpoint+history+source verification. No refitting."""
    worker=modules(reg);d=Path(reg['workspace'])/'layers'/f'layer{index:02d}'
    claim=r.read(d/'claim.json');r.verify_seal(claim)
    r.require(claim['registry_fingerprint']==reg['fingerprint'] and claim['layer_index']==index,'Layer claim changed')
    request=r.read(d/'request.json');worker.validate_request(request)
    receipt=r.read(d/'receipt.json');r.verify_seal(receipt)
    paths=layer_inputs(reg,index)
    checkpoint=Path(reg['workspace'])/'bank'/f'layer_{index:02d}.pt'
    expected_files={split:[r.artifact(p) for p in paths[split]] for split in ('train','dev')}
    r.require(request['source_files']==expected_files and request['registration']==r.fit_registration(reg)
              and request['seed']==1729+index and request['policy_fingerprint']==reg['checkpoint_hash']
              and request['layer_path']==r.LAYER_PATHS[index] and request['output_path']==str(checkpoint),
              'Layer request is not this exact fresh corpus/seed/output')
    if not unit_fixture:
        r.require(request['unit_test'] is False and request['device']=='cuda:0' and request['config']==r.ARCHITECTURE,
                  'CPU fixture cannot satisfy production admission')
    r.require(receipt['schema']=='snar_normalized_tc_layer_receipt_v1' and receipt['complete'] is True
              and receipt['request_fingerprint']==request['fingerprint']
              and receipt['parent_pid']==request['parent_pid'] and receipt['pgid']==request['parent_pgid']
              and receipt['unit_test'] is request['unit_test']
              and receipt['scientific_oracle_calls']==receipt['policy_models_loaded']==0,
              'Original worker closure differs')
    r.require(receipt['checkpoint']['path']==str(checkpoint) and receipt['sidecar']['path']==str(checkpoint)+'.json',
              'Unexpected checkpoint destination')
    r.check(receipt['checkpoint']);r.check(receipt['sidecar'])
    from matdiscovery.transcoders import TranscoderConfig,load_transcoder,validate_shard_splits
    from matdiscovery.normalized_transcoders import verify_normalized_metadata
    config=TranscoderConfig(**request['config'])
    model,metadata=load_transcoder(checkpoint,expected_policy_fingerprint=reg['checkpoint_hash'],
                                  expected_layer_path=r.LAYER_PATHS[index])
    r.require(metadata==r.read(str(checkpoint)+'.json'),'Raw checkpoint/sidecar differ')
    verify_normalized_metadata(metadata,config=config,policy_fingerprint=reg['checkpoint_hash'],
        layer_path=r.LAYER_PATHS[index],train_paths=paths['train'],dev_paths=paths['dev'],
        expected_registration=r.fit_registration(reg))
    population=validate_shard_splits(paths['train'],paths['dev'],policy_fingerprint=reg['checkpoint_hash'],
                                     layer_path=r.LAYER_PATHS[index],config=config)
    if not unit_fixture:r.require(population['rows']=={'train':9600,'dev':6400},'Missing captured rows')
    r.require(metadata['seed']==1729+index and metadata['batch_size']==request['batch_size']
              and metadata['learning_rate']==.0004 and metadata['epochs']==receipt['epochs']==64,
              'Actual fit did not execute the registered full64 schedule')
    passed=metadata['dev']['fvu_undefined']==0 and math.isfinite(metadata['dev']['output_fvu']) and metadata['dev']['output_fvu']<=.5
    r.require(metadata['fidelity_gate_passed'] is passed and receipt['fidelity_gate_passed'] is passed
              and receipt['status']==('succeeded' if passed else 'failed_fidelity')
              and receipt['dev']==metadata['dev'] and receipt['selected_epoch']==metadata['selected_epoch']
              and receipt['population_rows']==population['rows'] and receipt['runtime']==metadata['runtime']
              and receipt['transcoder_hash']==model.checkpoint_hash()==metadata['transcoder_hash'],
              'Actual raw TC/selection/fidelity/runtime differs')
    if not unit_fixture:
        cap=receipt['capacity_before_fitting'];worker.require_capacity(request,cap)
        r.require(metadata['runtime']['device']=='cuda:0','Wrong actual fitting device')
    return dict(layer_index=index,layer_path=r.LAYER_PATHS[index],checkpoint=str(checkpoint),
        checkpoint_sha256=receipt['checkpoint']['sha256'],metadata_sha256=receipt['sidecar']['sha256'],
        metadata=metadata,transcoder_hash=metadata['transcoder_hash'],status=receipt['status'],
        elapsed_seconds=receipt['elapsed_seconds'],worker_claim=r.artifact(d/'claim.json'),
        worker_request=r.artifact(d/'request.json'),worker_receipt=r.artifact(d/'receipt.json'))


def finish_layer(reg,claim):
    i=claim['layer_index'];d=Path(reg['workspace'])/'layers'/f'layer{i:02d}'
    r.require(r.read(d/'claim.json')==claim,'Claim ownership changed')
    layer=validate_layer(reg,i)
    terminal=r.sealed(dict(schema='snar_causal128_layer_complete_v1',complete=True,registry_fingerprint=reg['fingerprint'],
        layer_index=i,claim_token=claim['claim_token'],fidelity_gate_passed=layer['status']=='succeeded',
        checkpoint=dict(path=layer['checkpoint'],sha256=layer['checkpoint_sha256']),
        worker_receipt=layer['worker_receipt'],new_oracle_calls=0,new_generation_calls=0))
    r.publish(d/'complete.json',terminal)
    return terminal


def run_worker(reg,worker_id,gpu_uuid,parallel,fds,*,timeout_seconds=12*3600):
    r.require(re.fullmatch(r'[A-Za-z0-9_.-]{1,96}',worker_id) is not None,'Unsafe worker id')
    leases=lease_evidence(reg,gpu_uuid,fds);snap=resource_snapshot(reg,gpu_uuid);require_capacity(snap,parallel)
    out=Path(reg['workspace'])/'workers'/worker_id
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Lease layer prevents collision with prior scientific owner; this lock prevents two new supervisors.
    gpu_path=Path(reg['workspace'])/'gpu_locks'/(r.digest(gpu_uuid)+'.lock')
    gpu_path.parent.mkdir(parents=True,exist_ok=True);gpu_lock=gpu_path.open('a');fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    inherited=tuple(fds)+(lock.fileno(),gpu_lock.fileno())
    r.publish(out/'invocation.json',r.sealed(dict(schema='snar_causal128_worker_invocation_v1',
        registry_fingerprint=reg['fingerprint'],worker_id=worker_id,hostname=socket.gethostname(),pid=os.getpid(),
        gpu_uuid=gpu_uuid,parallel_layers=parallel,leases=leases,resources=snap,started_at=time.time())))
    active={};started=time.monotonic();errors=[]
    try:
        worker=modules(reg)
        while True:
            lease_evidence(reg,gpu_uuid,fds)
            r.require(time.monotonic()-started<timeout_seconds,'Bounded worker timeout; active claims retained')
            while len(active)<parallel:
                # Check current reserve before making a new irreversible claim.
                current=resource_snapshot(reg,gpu_uuid)
                if active and (current['gpu']['free_bytes']<12*GIB or current['ram']['direct_headroom_bytes']<4*GIB):break
                if not active:require_capacity(current,1)
                claim=claim_next(reg,worker_id)
                if claim is None:break
                i=claim['layer_index'];d=Path(reg['workspace'])/'layers'/f'layer{i:02d}'
                try:
                    request=make_layer_request(reg,claim,gpu_uuid);r.publish(d/'request.json',request)
                    log=(d/'worker.log').open('x')
                    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
                    env['PYTHONPATH']=os.pathsep.join([reg['original_source_root'],reg['method_source_root']])
                    proc=subprocess.Popen(worker.child_command(sys.executable,d/'request.json',d/'receipt.json'),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=inherited)
                    active[i]=(proc,claim,log)
                    print(json.dumps(dict(event='layer_started',layer=i,pid=proc.pid,worker_id=worker_id)),flush=True)
                except BaseException as exc:
                    r.publish(d/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),claim_token=claim['claim_token'],automatic_retry=False))
                    raise
            for i,(proc,claim,log) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del active[i];d=Path(reg['workspace'])/'layers'/f'layer{i:02d}'
                try:
                    r.require(code==0,'Original TC child failed; no retry')
                    terminal=finish_layer(reg,claim)
                    print(json.dumps(dict(event='layer_complete',layer=i,fidelity_gate_passed=terminal['fidelity_gate_passed'])),flush=True)
                except BaseException as exc:
                    errors.append(i)
                    if not (d/'failure.json').exists():r.publish(d/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),returncode=code,automatic_retry=False))
            status=queue_status(reg)
            if not active and not status['unclaimed_layers']:break
            time.sleep(2)
        result=r.sealed(dict(schema='snar_causal128_worker_drain_v1',complete=True,registry_fingerprint=reg['fingerprint'],
            worker_id=worker_id,queue_status=queue_status(reg),execution_errors=errors,
            elapsed_seconds=time.monotonic()-started,new_oracle_calls=0,new_generation_calls=0))
        r.publish(out/'drained.json',result);return result
    finally:
        # Popen identities are ours; never signal a group or any prior MADE process.
        for proc,_,_ in active.values():
            if proc.poll() is None:proc.terminate()
        for proc,_,log in active.values():
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        # Children inherit these file descriptions, so no unlock while any own child survives.
        lock.close();gpu_lock.close()


def aggregate(reg):
    root=Path(reg['workspace']);lock=(root/'.aggregate.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=queue_status(reg)
    if status['unclaimed_layers'] or status['claimed_without_terminal']:
        return dict(complete=False,status='pending',queue_status=status,main_experiment_complete=False)
    r.require(not any((root/'layers'/f'layer{i:02d}'/'failure.json').exists() for i in range(32)),
              'Execution failure requires reconciliation; no bank published')
    layers=[validate_layer(reg,i) for i in range(32)]
    passed=sum(x['status']=='succeeded' for x in layers)
    old=r.read(r.check(reg['original_bank']))
    keys=('schema','model_key','model_id','checkpoint_hash','policy_runtime','policy_configuration','policy_configuration_fingerprint','expected_layer_paths')
    bank={k:old[k] for k in keys}
    bank.update(status='succeeded' if passed==32 else 'failed_fidelity',complete=True,ready_for_graphs=passed==32,
        expected_layers=32,completed_layers=32,passed_layers=passed,
        configuration=dict(epochs=64,batch_size=256,learning_rate=.0004,device='cuda:0',max_dev_fvu=.5),
        snar_training_performed=True,scientific_oracle_calls=0,test_used_for_training_or_fidelity=False,layers=layers,
        representation_training_registration=r.artifact(root/'registration.json'),
        fresh_all32_layers=True,old_graphs_reused=0,full125_graph_rebuild_required=True,
        native_graph_availability_validated=False,finite_difference_validated=False,main_experiment_complete=False)
    bank['bank_fingerprint']=r.digest(bank)
    path=root/'bank/transcoder_manifest.json'
    if path.exists():r.require(r.read(path)==bank,'Previously published bank changed')
    else:r.publish(path,bank)
    if passed==32:
        from matdiscovery.representation_training import validate_transcoder_bank
        validate_transcoder_bank(path,model_key='qwen35_4b',checkpoint_hash=reg['checkpoint_hash'],
            policy_runtime=old['policy_runtime'],policy_configuration_fingerprint=reg['policy_configuration_fingerprint'])
    value=r.sealed(dict(schema='snar_causal128_training_complete_v1',complete=True,training_complete=True,
        registry_fingerprint=reg['fingerprint'],passed_layers=passed,pooled_development_fidelity_passed=passed==32,
        bank=r.artifact(path),new_oracle_calls=0,new_generation_calls=0,
        downstream=reg['downstream'],native_graph_available=None,FD_passed=None,main_experiment_complete=False))
    if (root/'training_complete.json').exists():r.require(r.read(root/'training_complete.json')==value,'Completion changed')
    else:r.publish(root/'training_complete.json',value)
    return value


def main():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--aggregate',action='store_true');m.add_argument('--status',action='store_true');m.add_argument('--worker-id')
    p.add_argument('--diagnostic-summary');p.add_argument('--workspace');p.add_argument('--registry');p.add_argument('--gpu-uuid')
    p.add_argument('--parallel-layers',type=int,default=4);p.add_argument('--lease-fds');a=p.parse_args()
    if a.prepare:result=r.prepare(a.diagnostic_summary,a.workspace)
    else:
        reg=r.read_registry(a.registry)
        if a.aggregate:result=aggregate(reg)
        elif a.status:result=queue_status(reg)
        else:
            r.require(a.lease_fds and a.gpu_uuid,'Explicit GPU and inherited lease descriptors required')
            result=run_worker(reg,a.worker_id,a.gpu_uuid,a.parallel_layers,tuple(int(x) for x in a.lease_fds.split(',')))
    print(json.dumps(result,sort_keys=True),flush=True)
    if a.worker_id and result.get('execution_errors'):return 1
    if a.aggregate and result.get('complete') is False:return 2
    if a.aggregate and result.get('pooled_development_fidelity_passed') is False:return 3
    return 0


if __name__=='__main__':raise SystemExit(main())
