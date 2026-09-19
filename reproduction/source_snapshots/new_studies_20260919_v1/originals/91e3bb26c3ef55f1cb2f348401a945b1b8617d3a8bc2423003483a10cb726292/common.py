"""Exact original source and closed, zero-trajectory failure identity."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,sys

R=Path('/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607')
ROOT=R/'benchmark_extensions/made_support_full_es_loader_recovery_20260920'
PACKAGE=R/'benchmark_extensions/made_support_aware_method_20260919'
WORKSPACE=R/'evaluation_workspaces/made_support_aware_method_v1'
POLICY=R/'environments/policy-py312/bin/python'
REG={'path':str(WORKSPACE/'configs/support_aware_method.json'),'sha256':'0088ab73351c0ce39f06e5b9c4c991feb9129dd39d80eeb41cd186c7f104d7ac'}
FP='baba7f16f1b5e9822ea9b584036e4e5a4a751c29b17993f6b819ea8a2bf90af7'
ORIGINAL_ES={'path':str(PACKAGE/'full_es_training.py'),'sha256':'6022bd57c0ab70f59ac1521e86b1858daa02ce28076d970293779227c67b31c9'}
FAILED=R/'benchmark_extensions/made_direct_cli_resource_20260919'
FAILURE_REFS={
 'child_log':{'path':str(FAILED/'runs/actor04_full_train_v1/science/child.log'),'sha256':'98b49a4df4bb2adbcc204afd84051f1b64b9ab325ffec8eea36b937cedee193b'},
 'child_exit':{'path':str(FAILED/'runs/actor04_full_train_v1/science/exit.json'),'sha256':'60e8c4ef9ec6bd972ac2bfbd5454770c94a4a40a78f68b4b9dfd35006e4202f5'},
 'sequence_exit':{'path':str(FAILED/'sequences/actor_04/sequence_exit.json'),'sha256':'cbdadfff99c8a74fecd99becd5041602052f160bee304f68776a6565603f7ab0'},
 'invocation':{'path':str(WORKSPACE/'experiments/full_es/invocation.json'),'sha256':'9b7d250502190562c5f9c1e53ca0c1b0ac44d41d6b2ffc5a1657acef213e4930'}}

def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def seal(value):return dict(value,fingerprint=digest(value))
def check_seal(value):require(value['fingerprint']==digest({k:v for k,v in value.items() if k!='fingerprint'}),'Seal differs')
def artifact(path):
    p=Path(path);a=p.stat();hasher=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):hasher.update(block)
    z=p.stat()
    require(all(getattr(a,k)==getattr(z,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Artifact changed during read')
    return {'path':str(p),'sha256':hasher.hexdigest()}
def verify(ref):require(artifact(ref['path'])==ref,'Source/evidence bytes changed: '+ref['path'])
def once(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    try:os.link(tmp,p)
    finally:tmp.unlink()

def verify_import_only(original_path,corrected_path):
    def function(path):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='load_controllers')
    expected=function(original_path);changed=0
    for node in ast.walk(expected):
        if isinstance(node,ast.ImportFrom) and node.module=='matdiscovery.core_collection' and [(a.name,a.asname) for a in node.names]==[('corpus_contract',None)]:
            node.module='matdiscovery.core_protocol';changed+=1
    require(changed==1 and ast.dump(expected,include_attributes=False)==ast.dump(function(corrected_path),include_attributes=False),
            'Corrected loader differs beyond the one public import')

def original_and_loader():
    verify(ORIGINAL_ES);verify(REG);require(read(REG['path'])['fingerprint']==FP,'Original scientific registration changed')
    verify_import_only(ORIGINAL_ES['path'],ROOT/'corrected_loader.py')
    sys.dont_write_bytecode=True;sys.path[:0]=[str(PACKAGE/'frozen_src'),str(PACKAGE)]
    import full_es_training as original
    require(Path(original.__file__).resolve()==PACKAGE/'full_es_training.py','Mixed original training module')
    spec=importlib.util.spec_from_file_location('_registered_corrected_controller_loader',ROOT/'corrected_loader.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    return original,loader

def failure_evidence(*,require_unstarted):
    for ref in FAILURE_REFS.values():verify(ref)
    done=read(FAILURE_REFS['child_exit']['path']);sequence=read(FAILURE_REFS['sequence_exit']['path'])
    invocation=read(FAILURE_REFS['invocation']['path'])
    require(done['returncode']==1 and done['entire_child_subtree_closed'] is True and done['process_group_closed'] is True
        and done['GPU_compute_processes_after']==[] and done['identity']=='train_full_es','Old failed training was not fully closed')
    require(sequence=={'complete':False,'evaluation_started':False,'returncode':1,'stage':'training_failed'},'Old sequence not the reviewed terminal failure')
    require(invocation['support_fingerprint']==FP and invocation['pid']==2372
        and invocation['hostname']=='tj-3041039-t-20260917212203-4mbtn-worker-1'
        and invocation['automatic_physical_replay'] is False,'Original failed invocation differs')
    log=Path(FAILURE_REFS['child_log']['path']).read_text()
    require("from matdiscovery.core_collection import corpus_contract" in log
        and "ImportError: cannot import name 'corpus_contract'" in log,'Unexpected original failure')
    folder=WORKSPACE/'experiments/full_es'
    if require_unstarted:
        require({p.name for p in folder.iterdir()}=={'invocation.json','.execution.lock'}
            and not (folder/'run').exists() and not (folder/'training_descriptor.json').exists(),
            'Any population, driver or unknown output requires separate reconciliation; never replay')
    return {'complete':True,'original_failure':FAILURE_REFS,'original_returncode':1,
        'zero_completed_or_attempted_physical_trajectories':True,'population_driver_not_constructed':True,
        'evidence_basis':'Pinned ImportError before callbacks/driver construction, closed subtree and invocation-only directory',
        'original_invocation_preserved':True,'old_failure_not_relabelled_success':True}
