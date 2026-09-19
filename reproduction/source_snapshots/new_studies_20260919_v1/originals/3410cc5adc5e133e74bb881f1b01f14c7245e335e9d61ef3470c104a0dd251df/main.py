#!/usr/bin/env python3
"""One registered SnAr repair job; source/claim/lease checks before model or oracle."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import common as c


def run(reg,name,runtime_path,lease_fds):
    from repair_pipeline import execute,runtime_for
    from validation import validate
    job=c.job_for(reg,name);directory=Path(reg['output'])/'jobs'/name
    c.verify_import(reg)
    missing=[x for x in job['dependencies'] if not (Path(reg['output'])/'jobs'/x/'receipt.json').exists()]
    if job['kind']=='bank':
        tc_root=Path(reg['tc_registry']['path']).parent
        if not (tc_root/'training_complete_with_shared_resources.json').exists():
            missing.append('external_resource_bound_TC_training_complete')
    if missing:return dict(complete=False,status='pending_dependencies',job_id=name,missing=missing,new_calls=0)
    if (directory/'receipt.json').exists():
        existing=c.receipt(reg,name)
        audit=validate(reg,job,[x['path'] for x in existing['outputs']])
        c.require(audit==existing['domain_validation'],'Completed job no longer validates identically')
        return existing
    c.verify_import(reg);runtime=runtime_for(reg,runtime_path)
    expected_python=runtime['oracle_python'] if job.get('arm') in ('random','gp_ei_scalarized') else runtime['policy_python']
    c.require(Path(sys.executable).resolve()==Path(expected_python).resolve()
              and Path(sys.prefix).resolve()==Path(expected_python).parent.parent.resolve(),
              'Use original oracle environment for CPU baselines and original policy environment for all other jobs')
    locks=[];resources=None;lease_proof=[]
    try:
        if job['resource']=='gpu':
            tc_r,tc_t=c.tc_modules(reg['tc_registry']['path']);tc=tc_r.read_registry(reg['tc_registry']['path'])
            lease_proof=tc_t.lease_evidence(tc,runtime['gpu_uuid'],lease_fds)
            guard=c.load_module('_snar_main_original_resource_guard',Path(reg['original_source_root'])/'runtime/resource_guard.py')
            resources=guard.snapshot(runtime['gpu_uuid'])
            errors=guard.admission_errors(resources,socket.gethostname());c.require(not errors,'GPU/cgroup headroom: '+str(errors))
            lockpath=Path(reg['output'])/'gpu_locks'/(c.digest(runtime['gpu_uuid'])+'.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
            lock=lockpath.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        else:c.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU job must be launched with CUDA_VISIBLE_DEVICES empty')
        owned=c.claim(reg,job,c.artifact(runtime_path));start=time.time()
        c.publish(directory/'started.json',dict(claim_token=owned['claim_token'],pid=os.getpid(),hostname=socket.gethostname(),
            runtime=c.artifact(runtime_path),resources=resources,leases=lease_proof,started_at=start))
        try:
            outputs=execute(reg,job,runtime)
            audit=validate(reg,job,outputs)
            c.read_registry(Path(reg['output'])/'registration.json');c.verify_import(reg)
            refs={x['path']:x for x in [*[c.artifact(x) for x in outputs],*audit.get('evidence_files',[])]}
            result=c.seal(dict(schema='snar_main_repair_job_receipt_v2',complete=True,registry_fingerprint=reg['fingerprint'],
                job_id=name,claim_token=owned['claim_token'],oracle_queries=job['queries'],outputs=[c.artifact(x) for x in outputs],
                artifacts=list(refs.values()),domain_validation=audit,elapsed_seconds=time.time()-start))
            c.publish(directory/'receipt.json',result);return result
        except BaseException as exc:
            c.publish(directory/'failure.json',dict(complete=False,claim_token=owned['claim_token'],error_type=type(exc).__name__,
                error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-start,automatic_replay=False))
            raise
    finally:
        for lock in locks:lock.close()


def main():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
    os.environ['TOKENIZERS_PARALLELISM']='false';sys.dont_write_bytecode=True
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--prepare',action='store_true');m.add_argument('--job');m.add_argument('--status',action='store_true')
    p.add_argument('--resource-amendment');p.add_argument('--registry');p.add_argument('--tc-registry');p.add_argument('--output');p.add_argument('--runtime');p.add_argument('--lease-fds')
    a=p.parse_args()
    if a.prepare:result=c.prepare(a.tc_registry,a.output,a.resource_amendment)
    else:
        reg=c.read_registry(a.registry)
        if a.status:
            result={x['job_id']:('complete' if (Path(reg['output'])/'jobs'/x['job_id']/'receipt.json').exists() else
                'claimed_or_failed' if (Path(reg['output'])/'jobs'/x['job_id']/'claim.json').exists() else 'unclaimed') for x in reg['jobs']}
        else:result=run(reg,a.job,a.runtime,tuple(int(x) for x in a.lease_fds.split(',')) if a.lease_fds else ())
    print(canonical(result),flush=True)
    return 2 if result.get("status")=="pending_dependencies" else 0


def canonical(value):
    import json
    return json.dumps(value,sort_keys=True,allow_nan=False)


if __name__=='__main__':raise SystemExit(main())
