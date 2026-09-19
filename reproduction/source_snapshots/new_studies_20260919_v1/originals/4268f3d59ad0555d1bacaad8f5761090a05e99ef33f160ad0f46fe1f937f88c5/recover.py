"""CPU-only derived acceptance using unchanged, explicitly bound SnAr modules.

Never invokes a scientific stage or writes the failed main output. Symlinks in
the new view resolve to original evidence; only derived report files are written.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

R=Path('/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607')
ROOT=R/'benchmark_extensions/summit_snar_report_recovery_20260919_v1'
MAIN=R/'benchmark_extensions/summit_snar_main_repair_20260919_v2'
ORIGINAL=R/'benchmark_extensions/summit_snar_20260918'
OUTPUT=MAIN/'runs/v2'
COORD=R/'technical_not_main/snar_main_cpu_coordinator_20260919_v1'
POLICY=R/'environments/policy-py312/bin/python'
REG_SHA='71fe749dc4bb88f8262e43c1305d19fda818de571c9c45f47bfbd7948e9a491c'
REG_FP='69c588044a290b03450485b9c7acbb7bfc03b7c97a9ee829e7af79f67cff4753'
SOURCES={
 'study':(ORIGINAL/'study.py','cda812f771cbb10e4b27e54316c2e2dd3b39d31733c734b37d54b0e19d084db4'),
 'common':(MAIN/'common.py','c053283f153392ec29d338e802e9e9cf2cb4489c879317abda7ca53b360e7f41'),
 'resource_bank':(MAIN/'resource_bank.py','4aa870852138e17c6028cb8fa1c60e9c5fa31a05dbe9ec6ef80b6550db630624'),
 'acceptance_repair':(MAIN/'acceptance_repair.py','57567104060e594b5e7713faec62ed36f75bb060064b4b38d58ada7c3e7a26c9'),
 'reporting':(MAIN/'reporting.py','a9583364fc5edeb24366dd0fdb268c5bd8dc911d9cd92321bf5ff92d1591bb10')}
RUN=ROOT/'runs/v1'
SCHEMA='snar_failed_report_explicit_import_CPU_recovery_v1'

def need(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ref(path):
    p=Path(path);a=p.stat();b=p.read_bytes();z=p.stat()
    need(all(getattr(a,k)==getattr(z,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Artifact changed during read')
    return {'path':str(p.resolve()),'sha256':hashlib.sha256(b).hexdigest()}
def check(value):need(ref(value['path'])==value,'Source/evidence bytes changed')
def seal(value):return dict(value,fingerprint=digest(value))
def verify(value):need(value['fingerprint']==digest({k:v for k,v in value.items() if k!='fingerprint'}),'Fingerprint differs')
def publish(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(temp,p)
    finally:temp.unlink()

def bind(name,path,expected):
    need(ref(path)['sha256']==expected,'Frozen import bytes differ: '+name)
    if name in sys.modules:
        need(Path(sys.modules[name].__file__).resolve()==Path(path).resolve(),'Foreign existing module: '+name)
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:sys.modules.pop(name,None);raise
    return module

def modules():
    # Bind the original pure metrics module before either dependent import.
    # No Pipeline/create/run/generate/oracle function is imported or invoked.
    loaded={name:bind(name,*SOURCES[name]) for name in ('study','common','resource_bank','acceptance_repair','reporting')}
    return loaded['common'],loaded['acceptance_repair'],loaded['reporting']

def fixed_source():
    need(Path(__file__).resolve()==ROOT/'recover.py','Use the deployed independent recovery source')
    need(ref(OUTPUT/'registration.json')['sha256']==REG_SHA,'Wrong original main registration')
    c,a,report=modules();reg=c.read_registry(OUTPUT/'registration.json')
    need(reg['fingerprint']==REG_FP and reg['output']==str(OUTPUT),'Original main identity differs')
    c.verify_import(reg)
    runtime=c.read(c.check(reg['original_runtime']))
    need(runtime['policy_python']==str(POLICY),'Original report interpreter differs')
    return c,a,report,reg

def runtime_check():
    need(Path(sys.executable).resolve()==POLICY.resolve() and Path(sys.prefix).resolve()==POLICY.parent.parent.resolve(),'Original policy Python required')
    need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden for CPU-only recovery')
    need(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Original two-thread CPU limits required')

def failure_lineage(reg):
    directory=OUTPUT/'jobs/report';failure=read(directory/'failure.json');claim=read(directory/'claim.json');started=read(directory/'started.json')
    verify(claim)
    need(failure['error_type']=='ModuleNotFoundError' and failure['error']=="No module named 'study'"
         and failure['complete'] is False and failure['automatic_replay'] is False,'Not the registered import-only report failure')
    trace=failure['traceback']
    need('acceptance_repair.py' in trace and 'from study import BOUNDS' in trace and 'repair_pipeline.py' in trace,
         'Different failed execution path')
    need(claim['job_id']=='report' and claim['registry_fingerprint']==REG_FP
         and failure['claim_token']==claim['claim_token']==started['claim_token']
         and claim['pid']==started['pid']==186333,'Wrong original report claim/PID')
    terminal=read(COORD/'launches/report/terminal.json');verify(terminal)
    launch=read(COORD/'launches/report/started.json')
    need(terminal['schema']=='snar_CPU_coordinator_child_terminal_v1' and terminal['job_id']=='report'
         and terminal['returncode']==1 and terminal['passed'] is False and terminal['main_receipt'] is None
         and terminal['no_retry'] is True and launch['pid']==started['pid'],'No actual original report process terminal')
    completion=read(COORD/'completion.json');verify(completion)
    need(completion['complete'] is False and completion['status']=='requires_reconciliation'
         and completion['GPU_jobs_dispatched']==0
         and terminal['plan_fingerprint']==completion['plan_fingerprint']==launch['plan_fingerprint'],
         'Original CPU coordinator has not closed in the expected failed state')
    need(any(x.get('job_id')=='report' and x.get('reason')=='child_nonzero_or_missing_verified_receipt'
             for x in completion['errors']),'Expected report terminal not reconciled by original coordinator')
    forbidden=['acceptance.json','results.json','results.md','study_complete.json','jobs/report/receipt.json']
    forbidden += ['test_'+arm+'_complete.json' for arm in ('random','gp_ei_scalarized','qwen_base','uq_esopt','uq_only','es_only','random_controller')]
    need(not any((OUTPUT/name).exists() for name in forbidden),'Original report artifacts already exist; do not create competing acceptance')
    return [ref(p) for p in (directory/'failure.json',directory/'claim.json',directory/'started.json',
         COORD/'launches/report/terminal.json',COORD/'launches/report/started.json',COORD/'completion.json')]

def accepted_jobs(c,reg):
    jobs=reg['jobs'];need(len(jobs)==50 and jobs[-1]['job_id']=='report','Wrong complete main job matrix')
    refs=[]
    for job in jobs[:-1]:
        value=c.receipt(reg,job['job_id']);claim=read(OUTPUT/'jobs'/job['job_id']/'claim.json');verify(claim)
        need(value['claim_token']==claim['claim_token'] and value['oracle_queries']==job['queries']
             and value['domain_validation']['passed'] is True,'Original successful job claim/domain mismatch')
        verify(value['domain_validation']);refs.append(ref(OUTPUT/'jobs'/job['job_id']/'receipt.json'))
    need(sum(j['queries'] for j in jobs[:-1])==2150,'Original new-call matrix differs')
    return refs

def prepare():
    runtime_check();c,a,report,reg=fixed_source();failure=failure_lineage(reg);receipts=accepted_jobs(c,reg)
    validation=read(ROOT/'CPU_validation.json')
    need(validation['passed'] is True and validation['returncode']==0
         and validation['source_sha256']==ref(__file__)['sha256'],'Missing matching CPU contract evidence')
    value=seal({'schema':SCHEMA,'source':ref(__file__),'CPU_validation':ref(ROOT/'CPU_validation.json'),
        'original_registry':ref(OUTPUT/'registration.json'),'original_registry_fingerprint':REG_FP,
        'original_source_modules':{k:ref(p) for k,(p,_) in SOURCES.items()},'original_failure_evidence':failure,
        'original_completed_job_receipts':receipts,'original_output':str(OUTPUT),'new_output':str(RUN),
        'algorithm_changes':False,'recovery_kind':'explicit_original_study_module_import_and_separate_derived_report',
        'new_oracle_or_model_calls':0,'old_report_failure_preserved':True,'automatic_replay':False,'created_at':time.time()})
    publish(ROOT/'registration.json',value);return {'registration':ref(ROOT/'registration.json'),'fingerprint':value['fingerprint']}

def registration(path,expected):
    need(Path(path)==ROOT/'registration.json' and ref(path)['sha256']==expected,'Exact recovery registration required')
    v=read(path);verify(v)
    need(v['schema']==SCHEMA and v['source']==ref(__file__) and v['original_registry_fingerprint']==REG_FP
         and v['original_output']==str(OUTPUT) and v['new_output']==str(RUN)
         and v['new_oracle_or_model_calls']==0 and v['algorithm_changes'] is False
         and v['automatic_replay'] is False and v['old_report_failure_preserved'] is True,'Recovery scope differs')
    c,a,report,reg=fixed_source()
    need(v['original_registry']==ref(OUTPUT/'registration.json') and v['original_source_modules']=={k:ref(p) for k,(p,_) in SOURCES.items()},'Registered source differs')
    check(v['CPU_validation']);need(failure_lineage(reg)==v['original_failure_evidence'],'Old failed attempt evidence changed')
    for x in v['original_completed_job_receipts']:check(x)
    need([x['path'] for x in v['original_completed_job_receipts']]==[str(OUTPUT/'jobs'/j['job_id']/'receipt.json') for j in reg['jobs'][:-1]],'Old complete receipt matrix differs')
    return v,c,a,report,reg

def view_inputs(protocol):
    names=['registration.json','episodes','oracle_journals','shared_prior_archive.json','adaptation_complete.json',
           'base_policy.json','risk_fit.json','snar_risk.pt','collection_complete.json']
    for branch in ('full','es_only'):
        names += [branch+'_es_history_G'+str(g)+'.json' for g in (1,2)]
        names += [branch+'_es_development_G'+str(g)+'.json' for g in range(3)]
        names += [branch+'_'+x+'.json' for x in ('checkpoint_selection','selected_es_history','selected_clean_replay','evolution_complete')]
    return names

def make_view(source,view,protocol):
    source=Path(source).resolve();view=Path(view).resolve()
    need(view!=source and not view.is_relative_to(source) and not source.is_relative_to(view),'Derived view must be separate from original output')
    need(not view.exists(),'Independent output already exists; no automatic replay');view.mkdir(parents=True)
    records=[]
    for name in view_inputs(protocol):
        p=source/name;need(p.exists(),'Missing original evidence input: '+name)
        (view/name).symlink_to(p,target_is_directory=p.is_dir());records.append({'name':name,'original':str(p),'read_only_evidence_reference':True})
    # These are derived aggregate markers; original raw journals and 35 complete
    # per-episode receipts have already been admitted, then are audited again.
    for arm in protocol['evaluation']['arms']:publish(view/f'test_{arm}_complete.json',{'episodes':5,'oracle_queries':250})
    return records

def derive(source,view,protocol,acceptance_module,report_module,original_protocol,tc_registry):
    links=make_view(source,view,protocol)
    accepted=acceptance_module.accept_study(view,protocol)
    need(accepted['passed'] is True and accepted['complete'] is True and accepted['unique_physical_attempts']==2300,'Original full acceptance did not pass')
    accepted.update(imported_collection_oracle_calls=150,new_adaptation_oracle_calls=400,new_test_oracle_calls=1750,
                    representation_training_registry=tc_registry,old_graphs_reused=0)
    accepted['fingerprint']=digest({k:v for k,v in accepted.items() if k!='fingerprint'})
    publish(Path(view)/'acceptance.json',accepted)
    # This is only reporting's documented read/output context, not a modified
    # scientific registration. The view registration still points to original.
    paths=report_module.build({'output':str(view),'original_protocol':original_protocol},accepted)
    return accepted,links,[Path(view)/'acceptance.json',*paths]

def execute(path,expected):
    runtime_check();v,c,a,report,reg=registration(path,expected)
    publish(RUN/'invocation.json',seal({'schema':SCHEMA,'registration':ref(path),'pid':os.getpid(),'time':time.time(),
         'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'new_oracle_or_model_calls':0}))
    critical=[*v['original_failure_evidence'],v['original_registry'],*v['original_completed_job_receipts']]
    before={x['path']:(x['sha256'],Path(x['path']).stat().st_mtime_ns) for x in critical}
    start=time.time()
    try:
        accepted,links,outputs=derive(OUTPUT,RUN/'derived',c.read(reg['original_protocol']['path']),a,report,reg['original_protocol'],reg['tc_registry'])
        for x in critical:
            check(x);need(Path(x['path']).stat().st_mtime_ns==before[x['path']][1],'Original artifact mtime changed')
        need(failure_lineage(reg)==v['original_failure_evidence'],'Original failed report changed during audit')
        result=seal({'schema':'snar_main_separate_derived_terminal_acceptance_v1','complete':True,'passed':True,
            'recovery_registration':ref(path),'source':ref(__file__),'original_main_registry':v['original_registry'],
            'original_report_remains_failed':True,'original_failure_evidence':v['original_failure_evidence'],
            'original_outputs_modified':False,'reference_view':links,'outputs':[ref(p) for p in outputs],
            'original_distinct_oracle_calls':accepted['unique_physical_attempts'],'imported_collection_calls':150,
            'new_adaptation_calls':400,'new_heldout_calls':1750,'new_calls_performed_by_recovery':0,
            'new_model_or_training_calls':0,'elapsed_seconds':time.time()-start,'finished_at':time.time()})
        publish(RUN/'completion.json',result);return result
    except BaseException as exc:
        import traceback
        publish(RUN/'failure.json',{'complete':False,'error_type':type(exc).__name__,'error':str(exc),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.time()-start,'automatic_replay':False})
        raise

def cli():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('prepare','check','execute'))
    p.add_argument('--registration',type=Path);p.add_argument('--expected-sha256');args=p.parse_args()
    if args.operation=='prepare':result=prepare()
    elif args.operation=='check':
        runtime_check();v,*_=registration(args.registration,args.expected_sha256)
        result={'ready':True,'registration_fingerprint':v['fingerprint'],'new_oracle_or_model_calls':0}
    else:result=execute(args.registration,args.expected_sha256)
    print(json.dumps(result,sort_keys=True,allow_nan=False))

if __name__=='__main__':cli()
