"""Immutable source/data admission for the SnAr main representation repair."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
SCHEMA='snar_main_causal128_repair_v2'
ARMS=('random','gp_ei_scalarized','qwen_base','uq_esopt','uq_only','es_only','random_controller')


def require(value,message):
    if not value:raise ValueError(message)


def read(p):return json.loads(Path(p).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    p=Path(path);before=p.stat();h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    after=p.stat();fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    require(all(getattr(before,k)==getattr(after,k) for k in fields),'File changed while read')
    return h.hexdigest()
def artifact(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def check(ref):
    require(sha(ref['path'])==ref['sha256'],'Changed artifact: '+ref['path']);return Path(ref['path'])
def seal(value):return dict(value,fingerprint=digest(value))
def verify(value):require(value['fingerprint']==digest({k:v for k,v in value.items() if k!='fingerprint'}),'Fingerprint differs')

def publish(path,value,*,identical=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    temporary=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('x') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,p)
        except FileExistsError:
            require(identical and p.read_text()==text,'Existing artifact/claim; no automatic replay: '+str(p))
    finally:temporary.unlink(missing_ok=True)


def load_module(name,path):
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    require(Path(sys.modules[name].__file__).resolve()==Path(path).resolve(),'Unexpected imported source')
    return sys.modules[name]


def tc_modules(path):
    root=Path(path).resolve().parents[2]
    require(root.name=='summit_snar_representation_training_20260919_v3','Wrong registered TC source namespace')
    registration=load_module('_snar_repair_tc_registration',root/'registration.py')
    old=sys.modules.get('registration');sys.modules['registration']=registration
    try:training=load_module('_snar_repair_tc_training',root/'training.py')
    finally:
        if old is None:sys.modules.pop('registration',None)
        else:sys.modules['registration']=old
    return registration,training



def read_tc_reference(ref):
    """Post-prepare byte-bound TC admission without importing torch in CPU baselines.

    Prepare already runs the full original TC registry/corpus validator. Runtime
    consumes exactly those sealed bytes and source files, never a JSON pass flag.
    GPU/bank stages additionally call the original full validator themselves.
    """
    tc_r,_=tc_modules(ref['path']);tc=read(check(ref));tc_r.verify_seal(tc)
    require(tc['schema']==tc_r.SCHEMA and tc['settings']==tc_r.SETTINGS and tc['gates']==tc_r.GATES
            and tc['architecture']==tc_r.ARCHITECTURE,'TC scope/configuration changed')
    for source in tc['source_files']+tc['original_source_files']+tc['diagnostic_source_files']:check(source)
    for key in ('input_manifest','diagnostic_summary','diagnostic_registration'):check(tc[key])
    return tc


def modules(reg):
    sys.dont_write_bytecode=True;sys.path[:0]=[reg['original_source_root'],reg['method_source_root']]
    base=load_module('_snar_repair_original_pipeline',Path(reg['original_source_root'])/'pipeline.py')
    return base


def jobs():
    items=[]
    def add(id,kind,deps=(),**extra):items.append(dict(job_id=id,kind=kind,dependencies=list(deps),**extra))
    add('es_only','evolve',branch='es_only',queries=200,resource='gpu')
    add('bank_admission','bank',queries=0,resource='gpu')
    add('original_failed_probe','probe',['bank_admission'],queries=0,resource='gpu')
    chunks=[f'graphs_{i:02d}' for i in range(8)]
    for i,name in enumerate(chunks):add(name,'graphs',['original_failed_probe'],chunk=i,queries=0,resource='gpu')
    add('risk_fit','risk',chunks,queries=0,resource='cpu')
    add('full','evolve',['risk_fit'],branch='full',queries=200,resource='gpu')
    add('freeze','freeze',['full','es_only','risk_fit'],queries=0,resource='cpu')
    tests=[]
    for arm in ARMS:
        for seed in range(5101,5106):
            name=f'test_{arm}_{seed}';tests.append(name)
            add(name,'evaluate',['freeze'],arm=arm,seed=seed,queries=50,
                resource='cpu' if arm in ('random','gp_ei_scalarized') else 'gpu')
    add('report','report',tests,queries=0,resource='cpu')
    require(len(items)==50 and sum(x['queries'] for x in items)==2150,'Wrong repair query matrix')
    return items


def rebase(value,old,new):
    if isinstance(value,str) and value.startswith(str(old)+'/'):return str(new)+value[len(str(old)):]
    if isinstance(value,list):return [rebase(x,old,new) for x in value]
    if isinstance(value,dict):return {k:rebase(v,old,new) for k,v in value.items()}
    return value


def prepare(tc_registry,output,resource_amendment):
    tc_r,_=tc_modules(tc_registry);tc=tc_r.read_registry(tc_registry)
    from resource_bank import amendment
    resource_ref=artifact(resource_amendment);amendment(resource_ref,tc)
    old=Path(tc['original_output']);out=Path(output).resolve()
    require(not out.exists() and not out.is_relative_to(Path(tc['original_source_root'])),'New independent empty output required')
    original=Path(tc['original_source_root']);protocol=read(original/'protocol.json')
    require(protocol['accounting']['planned_adaptation_oracle_calls']==550 and protocol['accounting']['planned_test_oracle_calls']==1750,
            'Original SnAr matrix differs')
    diag=read(check(tc['diagnostic_registration']))
    # Pinned completed diagnostic proves these are the original five collection episodes, pre-test.
    require(set(p.name for p in (old/'episodes').iterdir() if p.is_dir())==
        {f'collection_{s}_{n}' for s,seeds in [('train',(1101,1102,1103)),('dev',(2101,2102))] for n in seeds},
        'Original adaptation advanced beyond the registered completed collection')
    out.mkdir(parents=True);mapping=[]
    paths=[old/'base_policy.json',old/'technical_policy_probe.json',old/'collection_complete.json']
    for episode in sorted((old/'episodes').iterdir()):
        paths += [p for p in sorted(episode.rglob('*.json')) if p.name in ('episode.json','summary.json','generation.json','candidate.json')
                  or p.parent==episode and p.name.startswith('query')]
    paths += [p for p in sorted((old/'oracle_journals/collect_default').rglob('*'))
              if p.is_file() and (p.suffix=='.json' or p.name=='oracle_attempts.jsonl')]
    for source in paths:
        target=out/source.relative_to(old);before=artifact(source);target.parent.mkdir(parents=True,exist_ok=True)
        is_rebased=source.is_relative_to(old/'episodes')
        if is_rebased:publish(target,rebase(read(source),old,out))
        else:
            with target.open('xb') as stream:stream.write(source.read_bytes())
        require(artifact(source)==before,'Original collection changed during import')
        mapping.append(dict(original=before,derived=artifact(target),path_rebased=is_rebased))
    require(len(list((out/'episodes').glob('collection_*/query[0-9][0-9][0-9].json')))==150,'Incomplete import')
    require(not list((out/'episodes').rglob('risk_graph.json')) and not list((out/'episodes').rglob('native_graph.npz')),
            'Old graphs must never enter this representation repair')
    publish(out/'collection_import.json',seal(dict(schema='snar_readonly_collection_import_v2',complete=True,
        original_output=str(old),mapping=mapping,original_queries=150,new_queries=0,original_outputs_modified=False)))
    runtime=read(original/'runtime_config.json')
    own=[artifact(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')]
    registration=seal(dict(schema=SCHEMA,output=str(out),original_source_root=str(original),original_output=str(old),
        method_source_root=tc['method_source_root'],tc_registry=artifact(tc_registry),tc_resource_amendment=resource_ref,original_protocol=artifact(original/'protocol.json'),
        protocol_fingerprint=digest(protocol),original_runtime=artifact(original/'runtime_config.json'),
        original_registration=artifact(old/'registration.json'),original_diagnostic=tc['diagnostic_registration'],
        imported_collection=artifact(out/'collection_import.json'),sources=tc['original_source_files']+own,
        original_source_files=tc['original_source_files'],own_sources=own,oracle_source_root=runtime['oracle_source_root'],
        jobs=jobs(),imported_oracle_calls=150,new_adaptation_oracle_calls=400,new_test_oracle_calls=1750,
        total_distinct_oracle_calls=2300,study_scope='original SnAr seven-arm five-seed study with development-informed causal128 representation repair',
        pipeline_and_controller_math_unchanged=True,representation_dataset_changed=True,
        representation_amendment=dict(source_registry=artifact(tc_registry),old_rows_per_prefix=16,new_rows_per_prefix=128,
            fresh_all32_layers=True,epochs=64,train_prefixes=75,dev_prefixes=50,FVU_maximum=.5,graph_availability_minimum=.9),
        all125_graphs_new=True,old_graphs_reused=0,
        original_failed_probe='collection_train_1101/query005_proposal0',
        oracle_scope_change='test uses one independent 50-query physical journal per arm/seed; branch journals remain200',
        main_experiment_complete=False,registered_at=time.time()))
    publish(out/'registration.json',registration)
    return read_registry(out/'registration.json')


def read_registry(path):
    reg=read(path);verify(reg)
    require(reg['schema']==SCHEMA and Path(path).resolve()==Path(reg['output'])/'registration.json' and reg['jobs']==jobs(),
            'Invalid repair registry/scope')
    require(reg['own_sources']==[artifact(p) for p in sorted(HERE.glob('*.py')) if not p.name.startswith('test_')],
            'Main repair code changed')
    for ref in reg['sources']:check(ref)
    for name in ('original_protocol','original_runtime','original_registration','original_diagnostic','imported_collection','tc_registry','tc_resource_amendment'):check(reg[name])
    tc=read_tc_reference(reg['tc_registry'])
    from resource_bank import amendment
    amendment(reg['tc_resource_amendment'],tc)
    require(reg['original_source_files']==tc['original_source_files'] and reg['method_source_root']==tc['method_source_root']
            and reg['original_output']==tc['original_output'] and reg['original_source_root']==tc['original_source_root'],
            'Original source/representation registry identity differs')
    require(reg['sources']==reg['original_source_files']+reg['own_sources'],'Source inventory differs')
    original=read(reg['original_protocol']['path'])
    require(digest(original)==reg['protocol_fingerprint'] and reg['imported_oracle_calls']==150
            and reg['new_adaptation_oracle_calls']==400 and reg['new_test_oracle_calls']==1750
            and reg['total_distinct_oracle_calls']==2300 and reg['all125_graphs_new'] is True
            and reg['old_graphs_reused']==0 and reg['main_experiment_complete'] is False,'Science scope/gates changed')
    return reg


def verify_import(reg):
    manifest=read(check(reg['imported_collection']));verify(manifest)
    require(manifest['new_queries']==0 and manifest['original_queries']==150,'Import replayed or omitted queries')
    old=Path(reg['original_output']);out=Path(reg['output'])
    for item in manifest['mapping']:
        source=check(item['original']);target=check(item['derived'])
        require(source.is_relative_to(old) and target==out/source.relative_to(old),'Import path differs')
        if item['path_rebased']:require(read(target)==rebase(read(source),old,out),'Derived import changed scientific content')
        else:require(item['original']['sha256']==item['derived']['sha256'],'Raw import bytes changed')
    return manifest


def job_for(reg,name):
    found=[x for x in reg['jobs'] if x['job_id']==name];require(len(found)==1,'Unregistered job');return found[0]


def receipt(reg,name):
    path=Path(reg['output'])/'jobs'/name/'receipt.json';value=read(path);verify(value)
    require(value['complete'] is True and value['registry_fingerprint']==reg['fingerprint'] and value['job_id']==name,
            'Job receipt differs')
    for ref in value['artifacts']:check(ref)
    return value


def claim(reg,job,runtime_ref=None):
    for name in job['dependencies']:receipt(reg,name)
    directory=Path(reg['output'])/'jobs'/job['job_id'];directory.mkdir(parents=True,exist_ok=True)
    value=seal(dict(schema='snar_main_repair_claim_v2',registry_fingerprint=reg['fingerprint'],job_id=job['job_id'],
        claim_token=uuid.uuid4().hex,pid=os.getpid(),hostname=socket.gethostname(),runtime=runtime_ref,claimed_at=time.time(),automatic_replay=False))
    publish(directory/'claim.json',value);return value
