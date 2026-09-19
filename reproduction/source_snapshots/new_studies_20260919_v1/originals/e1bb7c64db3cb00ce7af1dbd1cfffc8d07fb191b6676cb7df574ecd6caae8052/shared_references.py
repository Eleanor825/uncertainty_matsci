"""Read original baseline/ES-only receipts in their unchanged Python namespace."""
from pathlib import Path
import json
import os
import subprocess
import sys

WORKER = '''import sys,json
from pathlib import Path
reference=json.loads(sys.argv[1]);root=Path(sys.argv[2])
sys.path[:0]=[str(root),str(root/'frozen_src')]
from matdiscovery.component_scope import read_registration,validate_launch,artifact
from matdiscovery.immutable_hash_cache import immutable_hash_cache
from evaluate import audit_job
from es_only_training import verify_es_only_receipt
with immutable_hash_cache():
    assert artifact(reference['path'])==reference
    reg=read_registration(reference['path']);validate_launch(reg)
    rows=[];pending=[]
    for job in reg['new_jobs']:
        if job['arm_id'] not in ('baseline_reference','es_only_independent'):continue
        path=Path(reg['workspace'])/'experiments/component_final/jobs'/job['job_id']
        if (path/'receipt.json').exists():rows.append(audit_job(reg,job))
        else:pending.append({'job_id':job['job_id'],'failed':(path/'failure.json').exists(),'claimed':(path/'claim.json').exists()})
    full=len(rows)==540
    training=verify_es_only_receipt(registration=reg) if full else None
    print(json.dumps({'complete':full,'rows':rows,'pending':pending,'original_registration':reference,'independent_ES_only':training},allow_nan=False))
'''

def audit_shared(reg):
    from matdiscovery.support_scope import read,verify_artifact,require
    reference=reg['component_reference'];verify_artifact(reference);old=read(reference['path'])
    sources=old['external_sources']
    for item in sources:verify_artifact(item)
    roots={str(Path(s['path']).parent) for s in sources};require(len(roots)==1,'Mixed shared source roots')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    env.pop('PYTHONPATH',None)
    # Fixed allowlisted read/audit code only, no task CLI and no model load.
    result=subprocess.run([sys.executable,'-B','-c',WORKER,json.dumps(reference),roots.pop()],
                          env=env,capture_output=True,text=True,check=True)
    value=json.loads(result.stdout)
    require(value['original_registration']==reference,'Shared audit changed registration')
    expected={j['job_id']:j for j in reg['shared_reference_jobs']}
    require(len(value['rows'])==len({r['job_id'] for r in value['rows']})
            and {r['job_id'] for r in value['rows']}|{r['job_id'] for r in value['pending']}==set(expected),
            'Shared reference inventory changed')
    for row in value['rows']:
        j=expected[row['job_id']]
        require((row['arm'],row['task_id'],row['seed'],row['budget'])==(j['arm_id'],j['task_id'],j['seed'],j['budget']),
                'Shared result pair changed')
    return value
