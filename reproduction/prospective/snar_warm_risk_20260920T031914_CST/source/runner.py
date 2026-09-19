"""Prepare/collect/four-fit/audit entry; no replay of old or uncertain science."""
from pathlib import Path
from contextlib import contextmanager
import argparse,fcntl,os,re,socket,sys,time,traceback
import warm_common as w

ASSETS=('shared_prior_archive.json','adaptation_complete.json','representation_selected.json','fresh_snar_transfer_fidelity.json','risk_fit.json','full_checkpoint_selection.json','es_only_checkpoint_selection.json','base_policy.json')

def prepare():
    w.require(Path(__file__).resolve()==w.ROOT/'runner.py','Use deployed source-bound entry')
    manifest=w.read(w.ROOT/'source_manifest.json')
    w.require({p.name for p in w.ROOT.glob('*.py')}=={x['path'] for x in manifest['files'] if x['path'].endswith('.py')},'Unregistered Python source')
    for item in manifest['files']:w.require(w.artifact(w.ROOT/item['path'])['sha256']==item['sha256'],'Package source changed')
    q,c,parent,tc=w.helpers();parent=c.read_registry(w.MAIN/'runs/v2/registration.json')
    plan=w.read(w.ROOT/'plan.json');jobs=w.job_catalog(plan)
    w.require(len(jobs)==5 and sum(j['budget'] for j in jobs)==150,'Expected5 new30-call episodes')
    original=Path(parent['output']);assets={name:w.artifact(original/name) for name in ASSETS}
    prior=w.read(w.check(assets['shared_prior_archive.json']));terminal=w.read(w.check(assets['adaptation_complete.json']))
    w.require(len(prior)==550 and terminal['queries']==550 and terminal['archive_fingerprint']==w.digest(prior),'Fixed550 prior not complete')
    w.require(len({r['query_id'] for r in prior})==550 and all(r['query_id'].startswith(('collection_','es_')) for r in prior),'Test/self data in fixed prior')
    protocol=w.read(w.check(parent['original_protocol']));used={1729,5201,5202,5203}
    for section,keys in [('risk',('train_seeds','dev_seeds')),('evolution',('fitness_episode_seeds','checkpoint_dev_seeds')),('evaluation',('seeds',))]:
        for key in keys:used.update(protocol[section][key])
    used.update(s for row in protocol['evolution']['mutation_seeds'] for s in row)
    w.require(not used.intersection(j['seed'] for j in jobs) and len({j['seed'] for j in jobs})==5,'New train/dev seed overlaps old scientific identities')
    selected=[w.read(w.check(assets[b+'_checkpoint_selection.json']))['selected'] for b in ('full','es_only')]
    w.require(all(x['generation']==0 for x in selected) and selected[0]['weight_hash']==selected[1]['weight_hash'],'Expected accepted originalG0 selection')
    names=w.read(w.check(assets['risk_fit.json']))['feature_names'];w.require(len(names)==285 and len(set(names))==285,'Original285 feature schema differs')
    for key in ('seed','hidden_width','epochs','batch_size','learning_rate','weight_decay','include_error_similarity'):
        w.require(plan['NN'][key]==protocol['risk'][key],'Unexpected NN hyperparameter amendment: '+key)
    import engine
    engine.cpu_contract();cpu_runtime=engine.runtime_record()
    base=w.read(w.check(assets['base_policy.json']))
    w.require(cpu_runtime['torch']==base['runtime']['torch_version'] and cpu_runtime['torch_cuda']=='12.8','Original CPU torch/CUDA build differs')
    out=w.ROOT/'runs/v1';out.mkdir(parents=True,exist_ok=False)
    sources=[w.artifact(w.ROOT/x['path']) for x in manifest['files']]+[w.artifact(w.ROOT/'source_manifest.json')]
    reg=w.seal({'schema':w.SCHEMA,'output':str(out),'jobs':jobs,'physical_calls':150,'train_calls':90,'development_calls':60,
        'plan':w.artifact(w.ROOT/'plan.json'),'sources':sources,'parent_sources':parent['sources']+parent['own_sources'],
        'parent_registry':w.artifact(w.MAIN/'runs/v2/registration.json'),'parent_protocol':parent['original_protocol'],
        'parent_runtime':parent['original_runtime'],'tc_registry':parent['tc_registry'],'assets':assets,
        'prior_rows':550,'prior_fingerprint':w.digest(prior),'base_state_hash':selected[0]['weight_hash'],
        'feature_names':names,'CPU_runtime':cpu_runtime,'original_used_seeds':sorted(used),'old_rows_used_for_NN_fit':0,
        'original_protocol_unchanged':True,'audit_adapter':'exact one-line private warm train/dev prior dispatch; exact new catalog',
        'created_epoch':time.time(),'controller_deployed':False,'new_ES_training_calls':0,'new_test_calls':0})
    w.publish(out/'registration.json',reg)
    return {'registration':w.artifact(out/'registration.json'),'fingerprint':reg['fingerprint'],'jobs':jobs,'physical_calls':150,'GPU_calls':0,'NN_fits':0}

def dependencies(reg):
    out=Path(reg['output']);missing=[];failed=[];receipts=[]
    for job in reg['jobs']:
        d=out/'jobs'/job['job_id'];p=d/'receipt.json'
        if w.present(d/'failure.json'):failed.append(job['job_id'])
        if not w.present(p):missing.append(job['job_id']);continue
        v=w.read(p);w.verify(v)
        w.require(v['complete'] is True and v['registration_fingerprint']==reg['fingerprint'] and v['job']==job and v['physical_calls']==30,'Collection receipt scope differs')
        receipts.append(w.artifact(p))
    return {'ready':not missing and not failed,'missing':missing,'failed':failed,'receipts':receipts}

def dataset(reg):
    from collection import audit_collection
    state=dependencies(reg);w.require(state['ready'],'Collection dependency incomplete')
    rows=[];audits=[]
    for job in reg['jobs']:
        d=Path(reg['output'])/'jobs'/job['job_id'];receipt=w.read(d/'receipt.json')
        old=w.read(w.check(receipt['audit']));actual=audit_collection(reg,job)
        w.require(w.canonical(actual)==w.canonical(old),'Collection does not recompute from immutable evidence')
        for ref in receipt['evidence_files']:w.check(ref)
        rows.extend(actual['rows']);audits.append(receipt['audit'])
    w.require(len(rows)==150 and len({r['query_id'] for r in rows})==150,'Wrong raw new-data membership')
    names=reg['feature_names'];train_prefixes={r['prefix_hash'] for r in rows if r['role']=='train' and r['prefix_hash']}
    for row in rows:
        if row['role']=='dev' and row['prefix_hash'] in train_prefixes:
            row['included']=False;row['reason']='exact_training_prefix_overlap'
        if row['features'] is not None:w.require(set(row['features'])==set(names),'Feature schema differs')
    return {'schema':'snar_new_warm_train_dev_dataset_v1','registration_fingerprint':reg['fingerprint'],
        'prior':reg['assets']['shared_prior_archive.json'],'prior_fingerprint':reg['prior_fingerprint'],'feature_names':names,
        'rows':rows,'collection_receipts':state['receipts'],'collection_audits':audits,'old_training_rows':0,'test_rows':0}

def fit(reg):
    state=dependencies(reg)
    if not state['ready']:return {'status':'failed_dependency' if state['failed'] else 'pending','dependencies':state,'NN_fits':0}
    old=w.read(w.check(reg['parent_runtime']))
    w.require(Path(sys.executable).resolve()==Path(old['policy_python']).resolve(),'Wrong CPU policy interpreter')
    w.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Fit requires CUDA hidden')
    import engine
    engine.cpu_contract();w.require(engine.runtime_record()==reg['CPU_runtime'],'CPU NN runtime changed');data=dataset(reg);folder=Path(reg['output'])/'nn';folder.mkdir(exist_ok=False)
    w.publish(folder/'claim.json',{'registration_fingerprint':reg['fingerprint'],'pid':os.getpid(),'hostname':socket.gethostname(),'automatic_replay':False,'started_at':time.time()})
    w.publish(folder/'dataset.json',data)
    try:
        result=engine.run(data,w.read(w.check(reg['plan'])),folder)
        return {'status':result.get('status','fitted'),'report':w.artifact(folder/'report.json'),'completion':w.artifact(folder/'completion.json'),
                'NN_fits':result.get('NN_fits',0),'optimizer_steps':result.get('optimizer_steps',0),'predictive_gate_passed':result.get('predictive_gate_passed',False)}
    except BaseException as exc:
        w.publish(folder/'failure.json',{'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'automatic_replay':False});raise

def audit(reg):
    import engine
    engine.cpu_contract();w.require(engine.runtime_record()==reg['CPU_runtime'],'CPU NN runtime changed');folder=Path(reg['output'])/'nn';stored=w.read(folder/'dataset.json');actual=dataset(reg)
    w.require(w.canonical(stored)==w.canonical(actual),'Dataset inputs changed')
    complete=w.read(folder/'completion.json');w.check(complete['report'])
    if complete['status']=='fitted':w.check(complete['model_seal'])
    result=engine.audit(stored,w.read(w.check(reg['plan'])),folder)
    w.publish(folder/'audit.json',dict(result,registration_fingerprint=reg['fingerprint'],report=w.artifact(folder/'report.json')))
    return result

@contextmanager
def owned_leases(paths):
    fds=[]
    try:
        for path in paths:
            p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);fd=os.open(p,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600);fds.append(fd)
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield tuple(fds)
    finally:
        for fd in reversed(fds):os.close(fd)

def worker(reg,worker_id,gpu,fds):
    q,c,parent,tc=w.helpers();q.lease_proof(tc,gpu,fds)
    guard=c.load_module('_snar_warm_original_resource_guard',Path(parent['original_source_root'])/'runtime/resource_guard.py')
    w.require(not guard.admission_errors(guard.snapshot(gpu),socket.gethostname()),'Original resource gate failed before dispatch')
    w.require(re.fullmatch(r'[A-Za-z0-9_-]{1,64}',worker_id),'Invalid worker ID')
    out=Path(reg['output']);mine=out/'workers'/worker_id;mine.mkdir(parents=True,exist_ok=False)
    rt=dict(w.read(w.check(reg['parent_runtime'])),gpu_uuid=gpu);w.publish(mine/'runtime.json',rt)
    visited=[]
    with q.try_lock(out/'gpu_workers'/(w.digest(gpu)+'.lock')) as own:
        w.require(own is not None,'Warm worker GPU already owned')
        w.publish(mine/'invocation.json',{'registration_fingerprint':reg['fingerprint'],'GPU_leases':q.lease_proof(tc,gpu,fds),'pid':os.getpid(),'started_at':time.time()})
        for job in reg['jobs']:
            jid=job['job_id']
            with q.try_lock(out/'dispatch_locks'/(jid+'.lock')) as assignment:
                if assignment is None:continue
                if w.present(out/'dispatch_claims'/(jid+'.json')) or w.present(out/'jobs'/jid):continue
                w.publish(out/'dispatch_claims'/(jid+'.json'),{'job':job,'registration_fingerprint':reg['fingerprint'],'worker_id':worker_id,'automatic_replay':False})
                argv=[rt['policy_python'],'-B','-u',str(w.ROOT/'runner.py'),'execute','--registry',str(out/'registration.json'),
                      '--expected-sha256',w.artifact(out/'registration.json')['sha256'],'--job-id',jid,'--runtime',str(mine/'runtime.json'),
                      '--gpu-uuid',gpu,'--lease-fds',','.join(map(str,fds))]
                end=q.launch_and_drain(argv,'SnAr_warm_train_dev',jid,mine/'dispatches'/jid,gpu,fds,(own.fileno(),assignment.fileno()))
                visited.append({'job_id':jid,'returncode':end['returncode'],'receipt':(out/'jobs'/jid/'receipt.json').is_file()})
                if end['returncode']!=0:break # Never replay unknown/failed own work.
    result={'worker_finished':True,'status':'failed' if any(x['returncode']!=0 for x in visited) else 'drained','visited':visited,'global_science_complete_not_inferred':True}
    w.publish(mine/'worker_exit.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','status','worker','execute','fit','audit'));p.add_argument('--registry');p.add_argument('--expected-sha256')
    p.add_argument('--worker-id');p.add_argument('--gpu-uuid');p.add_argument('--lease-fds');p.add_argument('--job-id');p.add_argument('--runtime');a=p.parse_args()
    if a.action=='prepare':result=prepare()
    else:
        reg=w.registration(a.registry,a.expected_sha256)
        if a.action=='status':result=dependencies(reg)
        elif a.action=='fit':result=fit(reg)
        elif a.action=='audit':result=audit(reg)
        elif a.action=='execute':
            from collection import execute
            w.require(a.lease_fds,'Scientific child requires inherited GPU leases');result=execute(reg,a.job_id,a.runtime,a.gpu_uuid,tuple(map(int,a.lease_fds.split(','))))
        else:
            q,c,parent,tc=w.helpers();_,tt=c.tc_modules(reg['tc_registry']['path'])
            if a.lease_fds:result=worker(reg,a.worker_id,a.gpu_uuid,tuple(map(int,a.lease_fds.split(','))))
            else:
                with owned_leases(tt.lease_paths(tc,a.gpu_uuid)) as fds:result=worker(reg,a.worker_id,a.gpu_uuid,fds)
    print(w.canonical(result),flush=True)
    if result.get('status')=='pending':return 2
    if result.get('status')=='failed_dependency':return 3
    if result.get('status')=='failed':return 1
    return 0
if __name__=='__main__':sys.exit(main())
