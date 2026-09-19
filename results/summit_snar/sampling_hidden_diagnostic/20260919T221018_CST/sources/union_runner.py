"""Only three new CPU NN fits; original twelve models, physical results and failed audit remain intact."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib,importlib.util,json,math,os,statistics,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
R=Path('/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607')
OLD=R/'benchmark_extensions/summit_snar_feature_family_diagnostic_20260919_v1'
PRODUCTION=R/'benchmark_extensions/summit_snar_sampling_hidden_diagnostic_20260919_v1'
SCHEMA='snar_sampling_hidden_additive_posthoc_diagnostic_v1'


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def wire(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def digest(value):return hashlib.sha256(wire(value).rstrip(b'\n')).hexdigest()


def original():
    plan=read(ROOT/'plan.json');path=OLD/'runner.py'
    require(sha(path)==plan['original_runner_sha256'],'Original diagnostic source changed')
    spec=importlib.util.spec_from_file_location('_snar_union_original',path);m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m;exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m


def selected_names(all_names):
    names=[k for k in all_names if k.startswith(('sampling.','hidden.'))]
    require(len(names)==132 and sum(k.startswith('sampling.') for k in names)==4 and sum(k.startswith('hidden.') for k in names)==128,'Wrong132-feature union')
    require(names==sorted(names),'Original ordered schema changed');return names


def sources():
    manifest=read(ROOT/'manifest.json')
    require(manifest['fingerprint']==digest({k:v for k,v in manifest.items() if k!='fingerprint'}),'New package seal differs')
    for r in manifest['files']:
        require(Path(r['path']).name==r['path'] and sha(ROOT/r['path'])==r['sha256'],'New diagnostic source changed')
    return artifact(ROOT/'manifest.json')


def context(reg=None):
    require(ROOT==PRODUCTION,'Use independent deployed extension namespace')
    package=sources();plan=read(ROOT/'plan.json');m=original()
    require(sha(OLD/'manifest.json')==plan['original_source_manifest_sha256'] and sha(OLD/'runs/v1/registration.json')==plan['original_registration_sha256'],'Parent source/registration changed')
    parent=read(OLD/'runs/v1/registration.json')
    cfg,data,cp,e,u,base,runtime=m.context(parent,full_evidence=True)
    require(sha(OLD/'runs/v1/report.json')==plan['original_report_sha256'],'Parent report changed')
    report=read(OLD/'runs/v1/report.json');proof=read(ROOT/'parent_exact_reconstruction.json')
    require(proof['source_sha256']==plan['parent_reconstruction_collector_sha256'] and proof['original_report_sha256']==plan['original_report_sha256']
        and proof['exact_original_publish_bytes_equal'] is True and proof['strict_canonical_JSON_difference_count']==0
        and proof['recomputed_original_publish_bytes_sha256']==plan['original_report_sha256'] and proof['numeric_comparison_tolerance_added'] is False,'Missing exact reconstruction lineage')
    require(proof['original_reproduction_gate_rechecked']['maximum_absolute_score_difference']==0 and proof['report_models']==12,'Original model gate missing')
    for r in proof['original_failure_evidence']:m.check(r)
    for r in proof['saved_model_references']:m.check(r['checkpoint'])
    names=selected_names(data['names'])
    expected={'manifest':package,'plan':artifact(ROOT/'plan.json'),'original_registration':artifact(OLD/'runs/v1/registration.json'),
        'original_report':artifact(OLD/'runs/v1/report.json'),'parent_reconstruction':artifact(ROOT/'parent_exact_reconstruction.json'),
        'runtime':runtime,'cohort_digest':m.digest(data),'features':names,'risk_config':cfg['original_risk_config']}
    if reg:
        m.sealed(reg);require(reg['schema']==SCHEMA and all(reg[k]==v for k,v in expected.items()),'New registration input/source mismatch')
        require(reg['model_jobs']==[{'family':'sampling_hidden','seed':s} for s in (1729,1730,1731)] and reg['fits']==3 and reg['maximum_optimizer_steps']==300,'Extension fit budget changed')
    return m,plan,data,e,u,report,expected


def prepare():
    m,plan,data,e,u,parent,expected=context();out=ROOT/'runs/v1'
    m.publish(out/'prepare_claim.json',{'time':time.time(),'pid':os.getpid(),'new_fits':0})
    value=m.seal({'schema':SCHEMA,**expected,'model_jobs':[{'family':'sampling_hidden','seed':s} for s in (1729,1730,1731)],
        'fits':3,'maximum_optimizer_steps':300,'scored_physical_actions':224,'new_LLM_graph_oracle_calls':0,
        'controller_deployment':False,'test_used_for_model_family_threshold_selection':False,
        'original_audit_returncode':1,'parent_reconstruction_is_separate_accepted_evidence':True,
        'posthoc':True,'physical_policy_selected_generation':0})
    m.publish(out/'registration.json',value)
    print(json.dumps({'prepared':True,'registration':artifact(out/'registration.json'),'fingerprint':value['fingerprint'],'fits_started':0}),flush=True)


def fit_metadata(model,path,seed,reg,data,e,u):
    names=reg['features'];d=e.raw_and_calibrated(model,data['dev'],names);labels=[r['label'] for r in data['dev']]
    require(model.model.layers[0].in_features==267 and sum(t.numel() for t in model.model.parameters())==21377,'Union model dimensions differ')
    require(model.features.names==names and len(model.features.error_prototype)==132,'Graph leaked into learned preprocessing')
    return {'family':'sampling_hidden','seed':seed,'features':names,'checkpoint':artifact(path),
        'configuration':asdict(model.config),'selected_epoch':model.selected_epoch,'epochs_executed':len(model.history),
        'optimizer_steps':len(model.history),'temperature':model.temperature,'history':model.history,
        'provenance':model.provenance,'raw_features':132,'transformed_features':267,'parameters':21377,
        'development_predictions':d,'development':{k:e.metrics(u,labels,d[k],.5) for k in ('raw','calibrated')},
        'test_used_for_fit':False}


def build_report(reg,data,e,u,parent,modelset):
    require(modelset['registration_fingerprint']==reg['fingerprint'] and len(modelset['fits'])==3,'Incomplete three-model seal')
    reports=[];points=[];differences=[]
    for job,ref in zip(reg['model_jobs'],modelset['fits']):
        require(sha(ref['path'])==ref['sha256'],'Fit metadata changed');fit=read(ref['path'])
        require(fit['seed']==job['seed'] and fit['family']=='sampling_hidden','Unregistered model')
        require(sha(fit['checkpoint']['path'])==fit['checkpoint']['sha256'],'New checkpoint changed')
        model=u.CalibratedRiskModel.load(fit['checkpoint']['path'])
        require(wire(fit)==wire(fit_metadata(model,fit['checkpoint']['path'],job['seed'],reg,data,e,u)),'New fit reconstruction differs')
        test,p=e.score(u,model,data['test'],reg['features'],.5)
        strata={str(s):e.score(u,model,[r for r in data['test'] if r['seed']==s],reg['features'],.5)[0] for s in range(5101,5106)}
        reports.append(dict(fit,test=test,policy_seed_strata=strata))
        for i,r in enumerate(data['test']):points.append({'family':'sampling_hidden','initialization_seed':job['seed'],'query_id':r['query_id'],'policy_seed':r['seed'],'no_hvi_label':r['label'],**{k:v[i] for k,v in p.items()}})
        old=next(r for r in parent['model_reports'] if r['family']=='all' and r['seed']==job['seed'])
        differences.append({'seed':job['seed'],'original_all_minus_new_sampling_hidden':{k:old['test']['calibrated'][k]-test['calibrated'][k] for k in ('auroc','brier','nll','ece')},'descriptive_not_IID':True})
    summary={}
    for kind in ('raw','calibrated'):
        summary[kind]={}
        for key in ('auroc','error_auprc','brier','nll','ece','risk_coverage_auc'):
            vs=[r['test'][kind][key] for r in reports];summary[kind][key]={'values':vs,'mean':statistics.mean(vs),'sample_variance':statistics.variance(vs)}
    require(sum(r['optimizer_steps'] for r in reports)<=300,'Additional optimizer budget exceeded')
    return {'schema':'snar_sampling_hidden_posthoc_scores_v1','registration_fingerprint':reg['fingerprint'],'complete':True,
        'models':reports,'predictions':points,'initialization_summary':summary,'paired_differences':differences,
        'cohort_counts':json.loads(wire(data['counts'])),'unique_physical_actions':224,'prediction_records':672,
        'new_fits':3,'actual_optimizer_steps':sum(r['optimizer_steps'] for r in reports),'max_optimizer_steps':300,
        'original_12_models_refit':False,'new_LLM_graph_oracle_calls':0,'controller_deployed':False,
        'test_used_for_configuration_family_or_threshold_selection':False,'physical_policy_selected_generation':0,
        'development_selected_epoch_and_temperature_in_sample':True,'parameter_count_difference_remains':True,
        'scope':'posthoc graph-increment comparison on same original selected actions; not a new policy improvement'}


def run():
    out=ROOT/'runs/v1';reg=read(out/'registration.json');m,plan,data,e,u,parent,_=context(reg)
    m.publish(out/'run_invocation.json',{'registration':artifact(out/'registration.json'),'pid':os.getpid(),'time':time.time(),'automatic_replay':False})
    refs=[]
    try:
        for job in reg['model_jobs']:
            d=out/'models'/str(job['seed']);d.mkdir(parents=True,exist_ok=False)
            m.publish(d/'claim.json',{'job':job,'registration_fingerprint':reg['fingerprint']})
            model=e.fit(u,data['train'],data['dev'],reg['features'],reg['risk_config'],job['seed'])
            cp=d/'risk.pt';model.save(cp);m.publish(d/'fit.json',fit_metadata(model,cp,job['seed'],reg,data,e,u));refs.append(artifact(d/'fit.json'));del model
        modelset=m.seal({'registration_fingerprint':reg['fingerprint'],'fits':refs,'test_outcomes_used_for_fit':False})
        m.publish(out/'all_models_frozen_before_test_scoring.json',modelset)
        report=build_report(reg,data,e,u,parent,modelset);m.publish(out/'report.json',report)
        m.publish(out/'completion.json',m.seal({'complete':True,'registration_fingerprint':reg['fingerprint'],
            'report':artifact(out/'report.json'),'modelset':artifact(out/'all_models_frozen_before_test_scoring.json'),'new_LLM_graph_oracle_calls':0}))
    except BaseException as error:
        m.publish(out/'failure.json',{'type':type(error).__name__,'message':str(error),'completed_fits':len(refs),'automatic_retry':False});raise


def audit():
    out=ROOT/'runs/v1';reg=read(out/'registration.json');m,plan,data,e,u,parent,_=context(reg)
    completion=read(out/'completion.json');m.sealed(completion);require(completion['complete'] and completion['registration_fingerprint']==reg['fingerprint'],'Wrong completion')
    modelset=read(m.check(completion['modelset']));m.sealed(modelset)
    expected=build_report(reg,data,e,u,parent,modelset)
    require(m.check(completion['report']).read_bytes()==wire(expected),'Report publish bytes differ; no numeric tolerance')
    import torch
    require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    m.publish(out/'audit.json',m.seal({'passed':True,'registration_fingerprint':reg['fingerprint'],'completion':artifact(out/'completion.json'),
        'comparison':'byte-exact original publish format','new_fits':0,'new_LLM_graph_oracle_calls':0}))
    print(json.dumps({'passed':True,'audit':artifact(out/'audit.json')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run','audit'));a=p.parse_args();globals()[a.action]()


if __name__=='__main__':main()
