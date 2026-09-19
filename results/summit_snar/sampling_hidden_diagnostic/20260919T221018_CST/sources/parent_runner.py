"""Fixed-input, CPU-only SnAr posthoc feature-family diagnostic. Never invokes study/policy/oracle."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
R = Path('/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607')
PRODUCTION = R/'benchmark_extensions/summit_snar_feature_family_diagnostic_20260919_v1'
STUDY = R/'benchmark_extensions/summit_snar_main_repair_20260919_v2/runs/v2'
POLICY = R/'environments/policy-py312/bin/python'
SCHEMA = 'snar_posthoc_feature_family_diagnostic_v1'
CONFIG_KEYS = ('seed','hidden_width','epochs','batch_size','learning_rate','weight_decay','patience','include_error_similarity')


def require(ok, message):
    if not ok: raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    p=Path(path);a=p.stat();h=hashlib.sha256()
    require(a.st_size <= 20*2**20, 'Only small evidence and small NN files permitted')
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    b=p.stat()
    require(all(getattr(a,k)==getattr(b,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')), 'Read input changed')
    return h.hexdigest()


def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def check(r):
    require(sha(r['path'])==r['sha256'],'Source/input hash differs: '+r['path']);return Path(r['path'])


def sealed(value):
    require(value['fingerprint']==digest({k:v for k,v in value.items() if k!='fingerprint'}),'Seal differs')


def seal(value):return dict(value,fingerprint=digest(value))


def publish(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    temporary=p.with_name('.'+p.name+'.'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,p)  # never replaces any existing result/claim
    finally:temporary.unlink(missing_ok=True)


def import_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;exec(compile(Path(path).read_bytes(),str(path),'exec'),m.__dict__);return m


def source_check():
    manifest=read(ROOT/'manifest.json');sealed(manifest)
    for item in manifest['files']:
        require(Path(item['path']).name==item['path'],'Nonlocal package inventory')
        require(sha(ROOT/item['path'])==item['sha256'],'Diagnostic package changed')
    return ref(ROOT/'manifest.json')


def runtime(plan):
    require(ROOT==PRODUCTION,'Production entry must use the independent frozen namespace')
    require(Path(sys.executable).resolve()==POLICY.resolve() and Path(sys.prefix).resolve()==POLICY.parent.parent.resolve(),'Wrong original policy interpreter')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CUDA hiding required')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        require(os.environ.get(key)=='2','Two CPU threads required: '+key)
    require(not os.environ.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'),'Unsafe checkpoint override')
    import torch,numpy,scipy,sklearn
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    actual={'python':sys.version,'torch':str(torch.__version__),'numpy':numpy.__version__,
        'scipy':scipy.__version__,'sklearn':sklearn.__version__,'threads':torch.get_num_threads(),'device':'cpu'}
    require(actual==plan['runtime'],'Original verified Python/numerical runtime differs')
    return actual


def projected_features(value,names):
    require(set(value)=={'values','nonfinite_keys','nonnumeric_keys'} and not value['nonfinite_keys'] and not value['nonnumeric_keys'],'Unexpected missing/nonnumeric features')
    values=value['values']
    require(set(values)==set(names) and all(type(x) in (int,float) and math.isfinite(x) for x in values.values()),'Incomplete/nonfinite feature schema')
    return values


def dataset(projection):
    """Fixed census; no outcome-based selection and no pseudo-labels for rejected proposals."""
    require(projection['schema']=='existing_snar_query_candidate_readonly_v1' and not projection['gaps'],'Incomplete original extraction')
    rf=projection['risk_fit'];names=rf['feature_names'];require(names==sorted(set(names)),'Feature order differs')
    counts=Counter(k.split('.')[0] for k in names)
    require(counts=={'sampling':4,'hidden':128,'graph':153},'Feature family dimensions differ')
    rows={}
    prefixes=set()
    for split, seeds, n in (('train',range(1101,1104),75),('dev',range(2101,2103),50)):
        rows[split]=[]
        require(len(rf['rows'][split])==n,'Original fit row count differs')
        require(Counter(r['episode'] for r in rf['rows'][split])=={f'collection_{split}_{seed}':25 for seed in seeds},'Original episode split differs')
        for r in rf['rows'][split]:
            require(r['label'] in (0,1) and r['prefix_hash'] not in prefixes,'Unknown/duplicate or overlapping training prefix')
            prefixes.add(r['prefix_hash'])
            rows[split].append(dict(r,features=projected_features(r['features'],names)))
    require(Counter(r['label'] for r in rows['train'])=={1:70,0:5} and Counter(r['label'] for r in rows['dev'])=={1:45,0:5},'Original class balance differs')
    queries=[q for q in projection['queries'] if q['arm']=='uq_esopt']
    require(len(queries)==250 and Counter(q['kind'] for q in queries)=={'llm_proposal':224,'paired_lhs_initial_design':25,'explicit_invalid_fallback':1},'Changed heldout query census')
    qmap={(q['episode'],q['query_index']):q for q in queries};require(len(qmap)==250,'Duplicate query identity')
    selected=[];candidates=projection['candidates'];require(len(candidates)==248,'Candidate census differs')
    for c in candidates:
        require(c['episode']==f"test_uq_esopt_{c['seed']}" and c['seed'] in range(5101,5106),'Unregistered candidate')
        if not c['selected']:continue
        q=qmap[(c['episode'],c['query_index'])]
        require(c['success'] and c['features'] and q['kind']=='llm_proposal' and q['chosen_features_path']==c['source']['path']
            and q['chosen_generation']==c['generation_path'] and q['parameters']==c['parameters'] and q['risk']==c['risk'],'Executed candidate association differs')
        require(q['no_hvi']==int(q['hv_increment']<=1e-12),'Original outcome-label direction differs')
        require(type(c['risk']) in (int,float) and 0<=c['risk']<=1,'Unknown original score')
        selected.append({'episode':c['episode'],'seed':c['seed'],'query_id':q['query_id'],'query_index':q['query_index'],
            'features':projected_features(c['features'],names),'label':q['no_hvi'],'original_probability':c['risk'],
            'candidate':c['source'],'query':q['source']})
    require(len(selected)==224 and len({r['query_id'] for r in selected})==224,'Executed score cohort differs')
    counts={'train_rows':75,'dev_rows':50,'test_candidate_rows':248,'test_selected_scored_rows':224,
        'unselected_scored_rows':sum(not c['selected'] and c['risk'] is not None for c in candidates),
        'generation_failure_rows':sum(not c['success'] for c in candidates),
        'successful_missing_graph_rows':sum(c['success'] and not c['features'] for c in candidates),
        'excluded_lhs_queries':25,'excluded_invalid_fallback_queries':1,
        'training_label_counts':dict(Counter(r['label'] for r in rows['train'])),
        'development_label_counts':dict(Counter(r['label'] for r in rows['dev'])),
        'scored_test_label_counts':dict(Counter(r['label'] for r in selected))}
    require((counts['unselected_scored_rows'],counts['generation_failure_rows'],counts['successful_missing_graph_rows'])==(15,8,1),'Exclusion census differs')
    return {'names':names,'train':rows['train'],'dev':rows['dev'],'test':selected,'counts':counts}


def evidence(projection,data):
    refs=projection['file_references']
    require(len(refs)==1037 and len({r['path'] for r in refs})==1037,'Fixed extraction read scope differs')
    for r in refs:
        require(Path(r['path']).is_relative_to(R),'Reference outside original project');check(r)
    registration=read(check(projection['registration']));sealed(registration)
    require(registration['total_distinct_oracle_calls']==2300 and registration['output']==str(STUDY),'Wrong parent scientific study')
    rp=read(check(projection['risk_fit']['source']));receipt=read(check(projection['risk_fit']['receipt']));sealed(receipt)
    require(receipt['complete'] and receipt['job_id']=='risk_fit' and receipt['registry_fingerprint']==registration['fingerprint'],'Original risk fit receipt differs')
    require(rp['feature_names']==data['names'] and rp['rows']['train']==data['train'] and rp['rows']['dev']==data['dev']
        and rp['test_used'] is False and rp['graph_fraction']==1,'Projected original fitting data differs')
    outputs=receipt.get('outputs',receipt.get('artifacts'))
    expected=str(STUDY/'snar_risk.pt');checkpoint=next(r for r in outputs if r['path']==expected)
    check(checkpoint)
    require(any(r['path']==projection['risk_fit']['source']['path'] and r['sha256']==projection['risk_fit']['source']['sha256'] for r in outputs),'Risk report missing from original receipt')
    return checkpoint


def context(reg=None, *, full_evidence=False):
    manifest=source_check();plan=read(ROOT/'plan.json');actual=runtime(plan)
    projection=read(ROOT/'input_projection.json')
    require(sha(ROOT/'input_projection.json')==plan['input_projection_sha256'],'Projection pin differs')
    data=dataset(projection)
    require(sha(ROOT/'original_uncertainty.py')==plan['original_method_sha256'],'Original NN source changed')
    source=next(r for r in projection['file_references'] if r['path'].endswith('/src/matdiscovery/uncertainty.py'))
    check(source);require(source['sha256']==plan['original_method_sha256'],'Original runtime NN source differs')
    protocol=read(check(projection['protocol']));require(protocol==read(ROOT/'original_protocol.json') and protocol['risk']==plan['original_risk_config'],'Original risk configuration differs')
    checkpoint=evidence(projection,data) if full_evidence else reg['original_checkpoint']
    check(checkpoint)
    engine=import_path('_snar_feature_diagnostic_engine',ROOT/'engine.py')
    module=engine.original_module(ROOT/'original_uncertainty.py')
    original=module.CalibratedRiskModel.load(checkpoint['path'])
    expected={'label_kind':protocol['risk']['label'],**{k:protocol['risk'][k] for k in CONFIG_KEYS}}
    require(asdict(original.config)==expected and original.features.names==data['names'],'Original checkpoint schema/config differs')
    require(original.provenance['train_rows']==75 and original.provenance['development_rows']==50 and original.provenance['test_used_for_fit'] is False,'Original checkpoint population differs')
    if reg:
        sealed(reg);require(reg['schema']==SCHEMA and reg['manifest']==manifest and reg['plan']==ref(ROOT/'plan.json')
            and reg['runtime']==actual and reg['cohort_digest']==digest(data),'Registration source/data/runtime differs')
    return plan,data,checkpoint,engine,module,original,actual


def prepare():
    source_check();out=ROOT/'runs/v1';out.mkdir(parents=True,exist_ok=True)
    publish(out/'prepare_claim.json',{'pid':os.getpid(),'time':time.time(),'no_fits':True})
    plan,data,checkpoint,engine,module,original,actual=context(full_evidence=True)
    # Load only the original small NN, predict existing development inputs; no fit.
    dev=engine.raw_and_calibrated(original,data['dev'],data['names'])
    known=read(ROOT/'input_projection.json')['risk_fit']['development_metrics']
    measured=module.risk_metrics([r['label'] for r in data['dev']],dev['calibrated'])
    require(measured==known,'Original loaded-model development metrics differ')
    value=seal({'schema':SCHEMA,'manifest':ref(ROOT/'manifest.json'),'plan':ref(ROOT/'plan.json'),
        'input_projection':ref(ROOT/'input_projection.json'),'original_checkpoint':checkpoint,'runtime':actual,
        'cohort_digest':digest(data),'cohort_counts':data['counts'],'model_jobs':[{'family':f,'seed':s} for f in ('all','sampling','hidden','graph') for s in plan['initialization_seeds']],
        'fits':12,'maximum_optimizer_steps':1200,'new_oracle_calls':0,'new_LLM_calls':0,'new_graph_calls':0,
        'original_loaded_model_metrics_match':True,'original_loaded_dev_metrics':measured,'replication_gate_still_pending':True,
        'controller_deployment':False,'test_used_for_selection':False})
    publish(out/'registration.json',value)
    print(json.dumps({'prepared':True,'registration':ref(out/'registration.json'),'fingerprint':value['fingerprint'],'fits_started':0}),flush=True)


def metadata(model,family,seed,path,data,engine,module,plan):
    names=engine.names_for(data['names'],family)
    devscores=engine.raw_and_calibrated(model,data['dev'],names)
    labels=[r['label'] for r in data['dev']]
    return {'family':family,'seed':seed,'checkpoint':ref(path),'features':names,
        'input_dimensions_before_preprocessing':len(names),'input_dimensions_after_preprocessing':model.model.layers[0].in_features,
        'parameter_count':sum(t.numel() for t in model.model.parameters()),'selected_epoch':model.selected_epoch,
        'epochs_executed':len(model.history),'optimizer_steps':len(model.history)*math.ceil(len(data['train'])/plan['original_risk_config']['batch_size']),
        'temperature':model.temperature,'history':model.history,'provenance':model.provenance,
        'development':{'raw':engine.metrics(module,labels,devscores['raw'],plan['fixed_threshold']),
            'calibrated':engine.metrics(module,labels,devscores['calibrated'],plan['fixed_threshold'])},
        'development_predictions':devscores,'test_read_for_fit':False}


def run():
    out=ROOT/'runs/v1';reg=read(out/'registration.json')
    plan,data,checkpoint,engine,module,original,_=context(reg,full_evidence=True)
    publish(out/'run_invocation.json',{'registration':ref(out/'registration.json'),'pid':os.getpid(),'time':time.time(),'automatic_replay':False})
    models=[]
    try:
        first=reg['model_jobs'][0];require(first=={'family':'all','seed':1729},'Replication must be first')
        for job in reg['model_jobs']:
            name=f"{job['family']}_{job['seed']}";directory=out/'models'/name;directory.mkdir(parents=True,exist_ok=False)
            publish(directory/'claim.json',{'job':job,'registration_fingerprint':reg['fingerprint']})
            names=engine.names_for(data['names'],job['family'])
            model=engine.fit(module,data['train'],data['dev'],names,plan['original_risk_config'],job['seed'])
            path=directory/'risk.pt';model.save(path)
            if job==first:
                # Outcome labels are not passed to this numerical-reproduction gate.
                values=engine.raw_and_calibrated(model,data['test'],names,individually=True)
                gate=engine.assert_replication(model,original,[r['original_probability'] for r in data['test']],values['calibrated'],
                    atol=plan['replication_gate']['saved_action_probabilities_absolute_tolerance'])
                publish(out/'original_reproduction_gate.json',seal(dict(gate,registration_fingerprint=reg['fingerprint'],checkpoint=ref(path))))
            info=metadata(model,job['family'],job['seed'],path,data,engine,module,plan)
            publish(directory/'fit.json',info);models.append(ref(directory/'fit.json'))
            del model
        require(len(models)==12,'Incomplete fit matrix')
        modelset=seal({'registration_fingerprint':reg['fingerprint'],'fits':models,'test_outcomes_used_for_fit':False})
        publish(out/'all_models_frozen_before_test_scoring.json',modelset)
        build_report(reg,plan,data,engine,module,modelset,publish_result=True)
    except BaseException as exc:
        publish(out/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed_fits':len(models),'automatic_retry':False})
        raise


def build_report(reg,plan,data,engine,module,modelset,*,publish_result):
    out=ROOT/'runs/v1';sealed(modelset)
    require(modelset['registration_fingerprint']==reg['fingerprint'] and len(modelset['fits'])==12 and modelset['test_outcomes_used_for_fit'] is False,'Incomplete fixed model set')
    reports=[];predictions=[]
    for expected, fitref in zip(reg['model_jobs'],modelset['fits']):
        info=read(check(fitref));require({'family':info['family'],'seed':info['seed']}==expected,'Model set identity differs')
        model=module.CalibratedRiskModel.load(check(info['checkpoint']));names=engine.names_for(data['names'],info['family'])
        require(info==metadata(model,info['family'],info['seed'],info['checkpoint']['path'],data,engine,module,plan),'Model metadata/predictions changed')
        test,values=engine.score(module,model,data['test'],names,plan['fixed_threshold'])
        strata={}
        for seed in range(5101,5106):
            sub=[r for r in data['test'] if r['seed']==seed]
            strata[str(seed)]=engine.score(module,model,sub,names,plan['fixed_threshold'])[0]
        reports.append(dict(info,test=test,policy_seed_strata=strata))
        for i,row in enumerate(data['test']):
            predictions.append({'family':info['family'],'initialization_seed':info['seed'],'episode':row['episode'],
                'query_id':row['query_id'],'policy_seed':row['seed'],'no_hvi_label':row['label'],
                'original_saved_probability':row['original_probability'],**{k:v[i] for k,v in values.items()}})
    labels=[r['label'] for r in data['test']];prior=sum(r['label'] for r in data['train'])/len(data['train'])
    controls={name:engine.metrics(module,labels,[p]*len(labels),plan['fixed_threshold']) for name,p in [('constant_train_no_hvi_prior',prior),('constant_0_5',.5)]}
    report={'schema':'snar_posthoc_feature_family_scores_v1','registration_fingerprint':reg['fingerprint'],'complete':True,
        'cohort_counts':data['counts'],'model_reports':reports,'initialization_summaries':engine.summarize_initializations(reports),
        'constant_controls':controls,'predictions':predictions,'new_oracle_calls':0,'new_LLM_calls':0,'new_graph_calls':0,
        'maximum_optimizer_steps':1200,'actual_optimizer_steps':sum(x['optimizer_steps'] for x in reports),
        'controller_deployed':False,'test_used_for_model_family_or_threshold_selection':False,
        'development_metrics_are_selection_and_calibration_in_sample':True,
        'scope':'posthoc scores on original full-policy selected actions; no new action or policy-benefit estimate',
        'limitations':['Only six HVI-improving actions in fixed224; seed5105 has no HVI improvement and undefined AUROC.',
            'All224 actions were selected by the original all-feature policy; this is not an unbiased counterfactual-policy sample.',
            'Three NN initialization seeds do not measure independent scientific-experiment replication.',
            'Feature families have different first-layer parameter counts; no capacity-matched claim.']}
    require(report['actual_optimizer_steps']<=1200,'Fit budget exceeded')
    if publish_result:
        publish(out/'report.json',report)
        publish(out/'completion.json',seal({'complete':True,'registration_fingerprint':reg['fingerprint'],
            'report':ref(out/'report.json'),'modelset':ref(out/'all_models_frozen_before_test_scoring.json'),
            'reproduction_gate':ref(out/'original_reproduction_gate.json'),'new_oracle_calls':0,'new_LLM_calls':0,'new_graph_calls':0}))
    return report


def audit():
    out=ROOT/'runs/v1';reg=read(out/'registration.json');plan,data,checkpoint,engine,module,original,_=context(reg,full_evidence=True)
    completion=read(out/'completion.json');sealed(completion)
    require(completion['complete'] and completion['registration_fingerprint']==reg['fingerprint'],'Incomplete result')
    gate=read(check(completion['reproduction_gate']));sealed(gate)
    replica=module.CalibratedRiskModel.load(check(gate['checkpoint']))
    values=engine.raw_and_calibrated(replica,data['test'],data['names'],individually=True)
    actual_gate=engine.assert_replication(replica,original,[r['original_probability'] for r in data['test']],values['calibrated'],
        atol=plan['replication_gate']['saved_action_probabilities_absolute_tolerance'])
    require(all(gate[k]==v for k,v in actual_gate.items()),'Numerical gate changed')
    expected=build_report(reg,plan,data,engine,module,read(check(completion['modelset'])),publish_result=False)
    require(read(check(completion['report']))==expected,'Report does not recompute from saved models/data')
    import torch
    require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    publish(out/'audit.json',seal({'passed':True,'registration_fingerprint':reg['fingerprint'],'completion':ref(out/'completion.json'),
        'models_checked':12,'scored_actions_per_model':224,'new_fits':0,'new_oracle_calls':0,'new_LLM_calls':0,'new_graph_calls':0}))
    print(json.dumps({'passed':True,'audit':ref(out/'audit.json')}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','audit']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'audit':audit}[args.action]()


if __name__=='__main__':main()
