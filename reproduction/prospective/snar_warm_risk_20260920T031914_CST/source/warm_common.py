"""Small source-bound metadata helpers; original science and outputs are read-only."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys
R=Path('/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607')
ROOT=R/'benchmark_extensions/summit_snar_warm_risk_repair_20260920_v1'
MAIN=R/'benchmark_extensions/summit_snar_main_repair_20260919_v2'
QUEUE=R/'benchmark_extensions/snar_gpu_queue_worker_20260919_v1/worker.py'
QUEUE_SHA='009a736d32cb03b866859b1e68c944213e751d11d1c331049cd7da922f5d33e5'
PARENT_SHA='71fe749dc4bb88f8262e43c1305d19fda818de571c9c45f47bfbd7948e9a491c'
SCHEMA='snar_warm_start_scalar_repair_v1'

def require(ok,message):
    if not ok:raise ValueError(message)
def present(p):
    try:Path(p).lstat();return True
    except FileNotFoundError:return False
def read(p):return json.loads(Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def seal(v):return dict(v,fingerprint=digest(v))
def verify(v):require(v['fingerprint']==digest({k:x for k,x in v.items() if k!='fingerprint'}),'Fingerprint differs')
def artifact(p):
    p=Path(p);a=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    b=p.stat();require((a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns),'Input changed while hashing')
    return {'path':str(p),'sha256':h.hexdigest()}
def check(ref):require(artifact(ref['path'])==ref,'Artifact changed: '+ref['path']);return Path(ref['path'])
def publish(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name('.'+p.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('x') as f:f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
    try:os.link(tmp,p)
    finally:tmp.unlink()
def module(name,path):
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve()==Path(path).resolve(),'Module identity differs');return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def helpers():
    require(artifact(QUEUE)['sha256']==QUEUE_SHA,'Original whole-subtree queue source differs')
    q=module('_snar_warm_original_queue',QUEUE);parent,tc=q.input_registries();c=q.main_common()
    require(artifact(MAIN/'runs/v2/registration.json')['sha256']==PARENT_SHA,'Original SnAr registration differs')
    return q,c,parent,tc

def job_catalog(plan):
    return [{'job_id':f'warm_risk_v1_{split}_{seed}','role':split,'arm':'qwen_base','seed':seed,'budget':30,
             'scope':f'warm_risk_v1_{split}_{seed}','prior_rows':550}
            for split,key in [('train','train_seeds'),('dev','dev_seeds')] for seed in plan[key]]

def registration(path,pin):
    require(artifact(path)['sha256']==pin,'Registration SHA differs');reg=read(path);verify(reg)
    require(reg['schema']==SCHEMA and Path(path)==ROOT/'runs/v1/registration.json','Wrong warm experiment namespace')
    for x in reg['sources']+reg['parent_sources']:check(x)
    expected_python={Path(x['path']).name for x in reg['sources'] if Path(x['path']).suffix=='.py'}
    require({p.name for p in ROOT.glob('*.py')}==expected_python,'Unregistered Python source in package')
    for k in ('plan','parent_registry','parent_protocol','parent_runtime','tc_registry'):check(reg[k])
    for x in reg['assets'].values():check(x)
    require(reg['jobs']==job_catalog(read(check(reg['plan']))) and reg['physical_calls']==150,'Job catalog/cost differs')
    return reg
