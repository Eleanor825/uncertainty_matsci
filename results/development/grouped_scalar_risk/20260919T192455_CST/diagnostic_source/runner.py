"""Source-bound standalone CPU risk diagnostic, never a controller deployment."""
from pathlib import Path
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
from types import SimpleNamespace

R = Path('/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607')
ROOT = R/'benchmark_extensions/made_scalar_grouped_training_20260919_v1'
CORE = R/'core_workspaces/made_fast_subset_causal_v5'
PACKAGE = R/'benchmark_extensions/made_support_aware_method_20260919'
POLICY = R/'environments/policy-py312/bin/python'
DIRECT = R/'benchmark_extensions/made_direct_cli_resource_20260919/launcher.py'
DIRECT_SHA = '301223da4c5754efceadf41dad9ca98a45b8e95af7949bc3262265b63a65d90e'
REFERENCE = DIRECT.parent/'science_references/support.json'
REFERENCE_SHA = '05862c099e9e36f3a0f0dba84d0e0fe0787e1ea39dc8ac85ae55a46b991c521c'
PROPOSAL_SHA = 'f8e4a32ba5f6be2cf4bff7b431e5b5bf319958af2635388e4e193db67ae91101'
CHECKPOINT = CORE/'experiments/risk_models/qwen35_4b/made/graph_risk.pt'
CHECKPOINT_SHA = '315d990a76f1329c4299a8be04dc710e47d23eb1e09f4563acb963f5ce1a2173'
FIT_SHA = '47df033a205031f9b6c006730bf489c275d2cd6d44d9da2d86e4ccb25d144ed3'
DATASET_FP = '62708aa644ced5b0225818cab66e43ff05e17c2261b69cd3aa09cd710b526879'
SCHEMA = 'standalone_MADE_grouped_scalar_training_diagnostic_v1'


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path):
    def pairs(items):
        out = {}
        for k,v in items:
            require(k not in out, 'Duplicate JSON key'); out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def artifact(path):
    p = Path(path).resolve(); before = p.stat(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024),b''): h.update(block)
    after = p.stat()
    require(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),
            'Source changed while hashing')
    return {'path':str(p),'sha256':h.hexdigest()}


def verify(ref):
    require(artifact(ref['path'])['sha256']==ref['sha256'], 'Artifact/source changed: '+ref['path'])


def seal(value): return dict(value,fingerprint=digest(value))
def check_seal(value): require(value['fingerprint']==digest({k:v for k,v in value.items() if k!='fingerprint'}),'Seal differs')


def publish(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,p)
    finally:temporary.unlink()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def sources(manifest_sha):
    verify({'path':str(ROOT/'manifest.json'),'sha256':manifest_sha});manifest=read(ROOT/'manifest.json')
    check_seal(manifest)
    for name,sha in manifest['files'].items():
        require(Path(name).name==name,'Escaping diagnostic source');verify({'path':str(ROOT/name),'sha256':sha})
    require(manifest['files']['runner.py']==artifact(__file__)['sha256'],'Unbound runner')
    verify({'path':str(ROOT/'proposal.json'),'sha256':PROPOSAL_SHA})
    verify({'path':str(DIRECT),'sha256':DIRECT_SHA})
    direct=module('_grouped_scalar_original_source',DIRECT)
    _,_,package=direct.source_reference(REFERENCE,REFERENCE_SHA)
    require(package==PACKAGE,'Wrong frozen scientific namespace')
    sys.dont_write_bytecode=True;sys.path[:0]=[str(PACKAGE/'frozen_src'),str(PACKAGE)]
    from matdiscovery import fit_risk,uncertainty,esopt
    for m in (fit_risk,uncertainty,esopt):require(Path(m.__file__).resolve().is_relative_to(PACKAGE),'Mixed original source imports')
    engine=module('_grouped_scalar_engine',ROOT/'engine.py')
    library=SimpleNamespace(**{k:getattr(uncertainty,k) for k in ('TrainOnlyFeatures','RiskMLP','RiskTrainingConfig',
        'CalibratedRiskModel','risk_metrics','_episode_weights')},tensor_state_hash=esopt.tensor_state_hash)
    return read(ROOT/'proposal.json'),fit_risk,library,engine


def materialize(manifest_sha):
    """Load only original four G0 collection sources and the small scalar NN."""
    proposal,fit,lib,engine=sources(manifest_sha)
    verify({'path':str(CHECKPOINT),'sha256':CHECKPOINT_SHA})
    fit_ref={'path':str(CHECKPOINT.with_suffix('.fit.json')),'sha256':FIT_SHA};verify(fit_ref)
    receipt=read(fit_ref['path']);old=lib.CalibratedRiskModel.load(CHECKPOINT)
    require(receipt['status']=='succeeded' and receipt['checkpoint_sha256']==CHECKPOINT_SHA
            and receipt['training_config']==asdict(old.config),'Original scalar training config differs')
    require(old.config.label_kind=='future_failure' and old.config.include_error_similarity
            and (old.config.hidden_width,old.config.epochs,old.config.batch_size,old.config.learning_rate,
                 old.config.weight_decay,old.config.patience)==(64,100,256,.001,.0001,15),'Unexpected original scalar settings')
    for ref in proposal['unchanged_source_data']:verify(ref)
    allowed={ref['path'] for ref in proposal['unchanged_source_data'] if Path(ref['path']).name=='collection_manifest.json'}
    # Preserve the original provenance list order, which is part of its hash.
    manifests=[ref['path'] for ref in old.provenance['collection_provenance']['collection_manifests']]
    require(len(manifests)==4 and set(manifests)==allowed
            and all(Path(p).is_relative_to(CORE/'experiments/collection') for p in manifests),'Wrong collection set')
    dataset=fit.load_risk_dataset(manifests,model_key='qwen35_4b',label_kind='future_failure')
    require(dataset.provenance['dataset_fingerprint']==receipt['dataset_fingerprint']==DATASET_FP,'Original scalar dataset differs')
    require(all(r['split'] in ('train','dev') and r['task_id']==('Al-Au-Hf' if r['split']=='train' else 'Al-Pd-Sm')
                for r in dataset.records),'Test or other chemistry entered diagnostic')
    data=fit.prepare_risk_data(dataset,'graph_risk')
    require(data['feature_names']==old.features.names and len(data['feature_names'])==298
            and len(data['train_y'])==293 and len(data['dev_y'])==83,'Changed actual293/83 scalar cohort')
    selected={s:[r for r in dataset.records if r['split']==s and r['label_future_failure'] is not None
                 and r['_excluded_from_fit_reason'] is None] for s in ('train','dev')}
    for split in ('train','dev'):
        data[split+'_ids']=[r['decision_id'] for r in selected[split]]
        data[split+'_episodes']=[r['episode_id'] for r in selected[split]]
    summaries={}
    for split in ('train','dev'):
        rows=[r for r in dataset.records if r['split']==split]
        summaries[split]={'all_proposals':len(rows),'included_scalar_rows':len(selected[split]),
            'null_future_labels':sum(r['label_future_failure'] is None for r in rows),
            'excluded_reasons':dict(Counter(str(r['_excluded_from_fit_reason']) for r in rows if r['_excluded_from_fit_reason'] is not None)),
            'all_proposal_graph_missing':sum(not r['_graph_available'] for r in rows),
            'included_graph_missing':sum(not r['_graph_available'] for r in selected[split]),
            'included_class_counts':dict(Counter(str(r['label_future_failure']) for r in selected[split])),
            'included_membership_hash':digest(sorted(data[split+'_ids'])),
            'episode_ids':sorted(set(data[split+'_episodes']))}
    require(summaries['train']['episode_ids']==proposal['data']['fit_episodes'],'Training episode set differs')
    folds=[]
    for f in proposal['folds']:
        held=f['heldout_training_episode'];fit_ids=sorted(r['decision_id'] for r in selected['train'] if r['episode_id']!=held)
        held_ids=sorted(r['decision_id'] for r in selected['train'] if r['episode_id']==held)
        require(digest(fit_ids)==f['fit_membership_sha256'] and digest(held_ids)==f['heldout_membership_sha256'], 'Fold membership changed')
        folds.append(f)
    evidence={p['path']:p['sha256'] for p in dataset.provenance['source_files']}
    evidence.update({str(CHECKPOINT):CHECKPOINT_SHA,str(CHECKPOINT.with_suffix('.fit.json')):FIT_SHA,
                     str(CHECKPOINT.with_suffix('.schema.json')):artifact(CHECKPOINT.with_suffix('.schema.json'))['sha256']})
    return SimpleNamespace(proposal=proposal,fit=fit,lib=lib,engine=engine,dataset=dataset,data=data,old=old,
                           counts=summaries,folds=folds,evidence=evidence)


def descriptor(ctx,manifest_sha):
    import torch,numpy,scipy,sklearn
    return seal({'schema':SCHEMA,'workspace':str(ROOT/'runs/v1'),'manifest':{'path':str(ROOT/'manifest.json'),'sha256':manifest_sha},
        'proposal':artifact(ROOT/'proposal.json'),'original_scientific_reference':{'path':str(REFERENCE),'sha256':REFERENCE_SHA},
        'source_checkpoint':{'path':str(CHECKPOINT),'sha256':CHECKPOINT_SHA},'dataset_fingerprint':DATASET_FP,
        'config':asdict(ctx.old.config),'feature_names':ctx.data['feature_names'],'folds':ctx.folds,'cohorts':ctx.counts,
        'source_evidence':ctx.evidence,'step_candidates':list(ctx.engine.STEPS),
        'models':['full_internal','external_only_control'],'selection_metric':'macro3_training_episode_raw_BCE_earliest_step_tie',
        'runtime':{'python':sys.version,'torch':str(torch.__version__),'numpy':numpy.__version__,
                   'scipy':scipy.__version__,'sklearn':sklearn.__version__,'device':'cpu','threads':2},
        'original_chemical_splits_unchanged':True,'development_role':'post-selection raw scoring and in-sample original temperature calibration',
        'CPU_only':True,'controller_deployment':False,'new_oracle_calls':0,'new_LLM_generation_calls':0})


def prepare(manifest_sha):
    ctx=materialize(manifest_sha);reg=descriptor(ctx,manifest_sha)
    publish(ROOT/'runs/v1/registration.json',reg)
    return {'registration':artifact(ROOT/'runs/v1/registration.json'),'fingerprint':reg['fingerprint'],
            'new_training_started':False,'new_oracle_calls':0}


def admission(path,sha):
    require(Path(path).resolve()==ROOT/'runs/v1/registration.json','Wrong diagnostic registry namespace')
    verify({'path':str(path),'sha256':sha});reg=read(path);check_seal(reg)
    ctx=materialize(reg['manifest']['sha256'])
    require(reg==descriptor(ctx,reg['manifest']['sha256']),'Registration/data/config/source differs')
    return reg,ctx


def train_registered(reg,ctx,out):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(),'CPU diagnostic cannot initialize CUDA')
    data=ctx.data;lib=ctx.lib;engine=ctx.engine;config=ctx.old.config
    publish(out/'all_input_row_accounting.json',{'cohorts':ctx.counts,
        'rows':[{k:r.get(k) for k in ('decision_id','episode_id','group_id','split','task_id','label_future_failure',
                                      '_excluded_from_fit_reason','_graph_available')} for r in ctx.dataset.records]})
    results={};episodes=np.asarray(data['train_episodes']);all_names=data['feature_names']
    for name in reg['models']:
        cols=[i for i,k in enumerate(all_names) if name=='full_internal' or k.startswith(('sampling.','action.'))]
        names=[all_names[i] for i in cols];x=data['train_x'][:,cols];dx=data['dev_x'][:,cols]
        folds=[]
        for f in ctx.folds:
            mask=episodes!=f['heldout_training_episode'];hold=~mask
            _,_,states=engine.fit_duration(x[mask],data['train_y'][mask],episodes[mask].tolist(),names,config,200,lib,
                output=out/name/f'fold_{f["fold"]}',heldout={'x':x[hold],'y':data['train_y'][hold],'groups':episodes[hold].tolist()},publish=publish)
            prevalence=float(data['train_y'][mask].mean())
            constants={k:lib.risk_metrics(data['train_y'][hold],np.full(int(hold.sum()),p)) for k,p in [('half',.5),('fold_train_prior',prevalence)]}
            publish(out/name/f'fold_{f["fold"]}'/'constant_controls.json',constants)
            folds.append(states)
        selected,curve=engine.choose_duration(folds)
        publish(out/name/'selection.json',{'optimizer_updates':selected,'curve':curve,
            'selection_data':'original train episodes only','dev_used_for_selection':False,'step0_selected':selected==0})
        model,features,states=engine.fit_duration(x,data['train_y'],episodes.tolist(),names,config,selected,lib,
            output=out/name/'final',publish=publish)
        development=engine.evaluate_development(model,features,dx,data['dev_y'],names,lib)
        before=lib.tensor_state_hash(dict(model.state_dict()))
        checkpoint=out/name/'final'/f'step_{selected:04d}.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(before==saved['model_state_hash'] and saved['schema']=='standalone_scalar_diagnostic_state_v1','Final diagnostic weights changed')
        publish(out/name/'development.json',development)
        results[name]={'selected_updates':selected,'selected_step0':selected==0,'feature_names':names,
            'model_state_hash':before,'checkpoint':artifact(checkpoint),'selection':artifact(out/name/'selection.json'),
            'development':artifact(out/name/'development.json'),'development_summary':{k:v for k,v in development.items() if k not in ('raw_logits','labels','calibrated_probabilities')},
            'production_admission':False,'controller_updated':False}
    with torch.no_grad():
        old_z=ctx.old.model(torch.from_numpy(ctx.old.features.transform(data['dev_x'],all_names,True))).numpy()
    baseline={'original_scalar':engine.score(data['dev_y'],old_z,lib.risk_metrics,ctx.old.temperature),
        'constant_half':lib.risk_metrics(data['dev_y'],np.full(len(data['dev_y']),.5)),
        'constant_train_prior':lib.risk_metrics(data['dev_y'],np.full(len(data['dev_y']),float(data['train_y'].mean())))}
    for p,sha in ctx.evidence.items():verify({'path':p,'sha256':sha})
    require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    return {'schema':SCHEMA,'complete':True,'registration_fingerprint':reg['fingerprint'],
        'models':results,'dev_controls':baseline,'cohorts':ctx.counts,'all_fit_failures_preserved':True,
        'neural_model_initializations_for_training':8,
        'actual_optimizer_updates_recorded':sum(len(read(p)) for p in out.glob('*/fold_*/updates.json'))
            +sum(len(read(p)) for p in out.glob('*/final/updates.json')),
        'controller_gain_demonstrated':False,'production_model_selected_or_deployed':False,
        'original_NN_unchanged':True,'new_oracle_calls':0,'new_LLM_generation_calls':0,'GPU_initialized':False,
        'limitations':['Only3 train episodes and1 dev chemistry','Dev temperature metrics are calibration in-sample',
                      'External-only control has a different input parameter count','Fresh registered dev rollouts required for controller benefit']}


def inventory(out):
    records=[]
    for p in sorted(Path(out).rglob('*')):
        require(not p.is_symlink(),'Aliased diagnostic output')
        if p.is_file():
            require(not p.name.endswith('.tmp'),'Partial diagnostic output');records.append(artifact(p))
    return records


def run(path,sha):
    reg,ctx=admission(path,sha);out=Path(reg['workspace'])/'attempt';out.mkdir(exist_ok=False)
    publish(out/'invocation.json',{'schema':SCHEMA,'registration':artifact(path),'fingerprint':reg['fingerprint'],
        'pid':os.getpid(),'hostname':socket.gethostname(),'time':time.time(),'automatic_retraining':False})
    started=time.monotonic()
    try:
        result=train_registered(reg,ctx,out)
        result['saved_model_verification']=audit_models(reg,ctx,out,result)
        result['elapsed_after_invocation_seconds']=time.monotonic()-started
        publish(out/'report.json',result)
        publish(Path(reg['workspace'])/'completion.json',seal({'schema':SCHEMA,'complete':True,'registration':artifact(path),
            'registration_fingerprint':reg['fingerprint'],'report':artifact(out/'report.json'),'artifacts':inventory(out),
            'controller_deployed':False,'new_oracle_calls':0,'finished_at':time.time()}))
        return {'complete':True,'completion':artifact(Path(reg['workspace'])/'completion.json'),'models':result['models']}
    except BaseException as error:
        publish(out/'failure.json',{'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
            'automatic_retraining':False,'partial_states_preserved':True,'new_oracle_calls':0})
        raise


def audit(path,sha):
    reg,ctx=admission(path,sha);out=Path(reg['workspace'])/'attempt';done=read(Path(reg['workspace'])/'completion.json');check_seal(done)
    require(done['schema']==SCHEMA and done['complete'] is True and done['registration']==artifact(path)
            and done['registration_fingerprint']==reg['fingerprint'] and done['artifacts']==inventory(out)
            and done['report']==artifact(out/'report.json') and done['controller_deployed'] is False,'Diagnostic acceptance changed')
    report=read(out/'report.json');require(report['complete'] and not report['production_model_selected_or_deployed'],'Wrong diagnostic meaning')
    verification=audit_models(reg,ctx,out,report)
    return {'complete':True,'report':done['report'],'saved_model_verification':verification,
            'controller_gain_demonstrated':False,'new_oracle_calls':0}


def audit_models(reg,ctx,out,report):
    """Actual saved-state predictions and train-only selection, never NN fit."""
    import numpy as np
    import torch
    torch.set_num_threads(2)
    data=ctx.data;lib=ctx.lib;engine=ctx.engine;config=ctx.old.config;episodes=np.asarray(data['train_episodes'])
    checked=[]
    for name in reg['models']:
        cols=[i for i,k in enumerate(data['feature_names']) if name=='full_internal' or k.startswith(('sampling.','action.'))]
        names=[data['feature_names'][i] for i in cols];x=data['train_x'][:,cols];dx=data['dev_x'][:,cols];folds=[]
        for f in ctx.folds:
            mask=episodes!=f['heldout_training_episode'];hold=~mask;folder=out/name/f'fold_{f["fold"]}'
            updates=read(folder/'updates.json')
            require([r['optimizer_update'] for r in updates]==list(range(1,201)), 'Missing training update records')
            states={}
            for step in engine.STEPS:
                _,_,actual=engine.verify_saved_state(folder/f'step_{step:04d}.pt',x[mask],data['train_y'][mask],
                    episodes[mask].tolist(),names,config,lib,heldout={'x':x[hold],'y':data['train_y'][hold]})
                require(actual['updates']==step and actual==read(folder/f'step_{step:04d}.json'), 'Fold score differs from actual saved tensors')
                if step:require(actual['model_state_hash']==updates[step-1]['state_hash'],'Update/candidate state mismatch')
                states[step]=actual
            folds.append(states)
        selected,curve=engine.choose_duration(folds);selection=read(out/name/'selection.json')
        require(selection=={'optimizer_updates':selected,'curve':curve,'selection_data':'original train episodes only',
                           'dev_used_for_selection':False,'step0_selected':selected==0}, 'Selection changed or used dev')
        folder=out/name/'final';path=folder/f'step_{selected:04d}.pt'
        model,features,state=engine.verify_saved_state(path,x,data['train_y'],episodes.tolist(),names,config,lib)
        require(state['updates']==selected and state==read(folder/f'step_{selected:04d}.json')
                and len(read(folder/'updates.json'))==selected, 'Final update count/state differs')
        development=engine.evaluate_development(model,features,dx,data['dev_y'],names,lib)
        require(development==read(out/name/'development.json'), 'Saved dev predictions/calibration differ')
        result=report['models'][name]
        require(result['selected_updates']==selected and result['checkpoint']==artifact(path)
                and result['model_state_hash']==state['model_state_hash'] and result['feature_names']==names
                and result['development']==artifact(out/name/'development.json'), 'Report differs from recomputed models')
        checked.append({'model':name,'candidate_states_recomputed':len(engine.STEPS)*3,
                        'selected_updates':selected,'final_dev_predictions_recomputed':len(data['dev_y'])})
    require(not torch.cuda.is_initialized(), 'CPU audit unexpectedly initialized CUDA')
    return checked


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','audit'))
    p.add_argument('--manifest-sha');p.add_argument('--registry');p.add_argument('--sha');a=p.parse_args()
    require(Path(__file__).resolve()==ROOT/'runner.py' and sys.executable==str(POLICY),'Use deployed source and original policy Python')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'Hidden CUDA and CPU2 required')
    sys.dont_write_bytecode=True
    result=prepare(a.manifest_sha) if a.action=='prepare' else run(a.registry,a.sha) if a.action=='run' else audit(a.registry,a.sha)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
