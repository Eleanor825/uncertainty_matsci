"""Verify the saved sampling+hidden diagnostic without model inference or fitting."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse
import csv
import gzip
import io
import json
import math
import statistics as st
from metric_math import metrics,calibration_bins,digest,sha,require

D=Path(__file__).resolve().parent
SEEDS=[1729,1730,1731]
ROUND_OFF=1e-12

def json_bytes(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def csv_bytes(rows):
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return s.getvalue().encode()

def compute():
    inventory=json.loads((D/'source_manifest.json').read_text());files={};refs={}
    for r in inventory['gzip_scientific_sources']:
        p=D/r['bundle_path'];require(p.resolve().is_relative_to(D.resolve()),'Source path escape')
        zipped=p.read_bytes();require(sha(zipped)==r['gzip_sha256'],'Compressed source differs')
        raw=gzip.decompress(zipped);require(sha(raw)==r['original_sha256']and len(raw)==r['original_bytes'],'Original source bytes differ')
        files[r['logical_name']]=raw;refs[r['logical_name']]=r
    for r in inventory['scientific_code']:
        p=D/r['bundle_path'];require(sha(p.read_bytes())==r['original_sha256']and p.stat().st_size==r['bytes'],'Original science source differs')
    report=json.loads(files['union/report.json']);reg=json.loads(files['union/registration.json']);audit=json.loads(files['union/audit.json'])
    completion=json.loads(files['union/completion.json']);models=json.loads(files['union/all_models_frozen_before_test_scoring.json'])
    for name in ['registration.json','audit.json','completion.json','all_models_frozen_before_test_scoring.json']:
        v=json.loads(files['union/'+name]);require(v['fingerprint']==digest({k:x for k,x in v.items()if k!='fingerprint'}),'Original JSON seal differs')
        if name!='registration.json':require(v['registration_fingerprint']==reg['fingerprint'],'Cross-registration artifact')
    require(audit['passed']is True and audit['comparison']=='byte-exact original publish format','New actual audit not passed')
    require(audit['completion']['sha256']==sha(files['union/completion.json']),'Audit completion differs')
    require(completion['complete']and completion['report']['sha256']==sha(files['union/report.json'])and completion['modelset']['sha256']==sha(files['union/all_models_frozen_before_test_scoring.json']),'Completion source differs')
    require(report['registration_fingerprint']==reg['fingerprint']and report['complete'],'Unaccepted report')
    require(reg['original_audit_returncode']==1,'Historical parent audit must remain failed')
    parent=json.loads(files['parent/report.json']);projection=json.loads(files['parent/input_projection.json'])
    require(reg['original_report']['sha256']==sha(files['parent/report.json']),'Parent report differs')
    plan=json.loads((D/'sources/union_plan.json').read_text());source_manifest=json.loads((D/'sources/union_source_manifest.json').read_text())
    require(reg['plan']['sha256']==sha((D/'sources/union_plan.json').read_bytes())and reg['manifest']['sha256']==sha((D/'sources/union_source_manifest.json').read_bytes()),'Registered new source/plan differs')
    require(source_manifest['fingerprint']==digest({k:v for k,v in source_manifest.items()if k!='fingerprint'}),'Source manifest seal differs')
    selected=[x for x in projection['risk_fit']['feature_names']if x.startswith(('sampling.','hidden.'))]
    require(selected==sorted(selected)==reg['features']and len(selected)==132,'Different132-feature union')
    require(Counter(x.split('.')[0]for x in selected)=={'hidden':128,'sampling':4},'Unexpected feature family')
    require(reg['model_jobs']==[{'family':'sampling_hidden','seed':x}for x in SEEDS]and reg['fits']==3,'Different fit matrix')
    require(report['new_fits']==3 and not report['original_12_models_refit']and report['new_LLM_graph_oracle_calls']==0 and not report['controller_deployed'],'Diagnostic changed scope')
    require(not reg['test_used_for_model_family_threshold_selection']and not report['test_used_for_configuration_family_or_threshold_selection'],'Test-based selection claimed')
    context=json.loads(files['parent/prior_and_selection.json']);es=json.loads(files['parent/es_only_development.json'])['selection']['value']
    require(context['shared_prior_rows']==550 and context['full_checkpoint_selection']['selected']['generation']==es['selected']['generation']==reg['physical_policy_selected_generation']==0,'Physical G0 context differs')
    require(context['full_UQ_per_seed_HV_curves_equal']and context['ES_only_base_per_seed_HV_curves_equal'],'Duplicate physical curve context changed')
    devrows=projection['risk_fit']['rows']['dev'];trainrows=projection['risk_fit']['rows']['train'];devy=[r['label']for r in devrows]
    require(len(trainrows)==75 and Counter(r['label']for r in trainrows)=={0:5,1:70}and len(devrows)==50 and Counter(devy)=={0:5,1:45},'Training/development population differs')
    queries={q['query_id']:q for q in projection['queries']if q['arm']=='uq_esopt'and q['kind']=='llm_proposal'}
    require(len(queries)==224,'Different selected-action cohort')
    points=defaultdict(list)
    for r in report['predictions']:
        require(r['family']=='sampling_hidden','Unexpected new model family');points[r['initialization_seed']].append(r)
    require(set(points)==set(SEEDS)and len(report['predictions'])==672,'Incomplete new predictions')
    parent_points=defaultdict(list)
    for r in parent['predictions']:
        if r['family']=='all':parent_points[r['initialization_seed']].append(r)
    parent_models={m['seed']:m for m in parent['model_reports']if m['family']=='all'}
    new_models={m['seed']:m for m in report['models']};require(set(new_models)==set(SEEDS),'Incomplete model summaries')
    comparisons=0;max_roundoff=0.;metric_rows=[];strata=[];bins=[];development=[];development_metrics=[];paired=[];training=[];calculated={}
    def compare(a,b,label):
        nonlocal comparisons,max_roundoff
        require(a.keys()==b.keys(),'Metric keys differ: '+label)
        for k,value in a.items():
            previous=b[k];comparisons+=1
            if value is None or previous is None:require(value is previous,'Undefined metric differs: '+label+'/'+k)
            elif type(value)is int:require(value==previous,'Count differs: '+label+'/'+k)
            else:
                d=abs(value-previous);max_roundoff=max(max_roundoff,d);require(d<=ROUND_OFF,'Metric differs: '+label+'/'+k)
    def scalar(a,b,label):
        nonlocal comparisons,max_roundoff
        comparisons+=1;d=abs(a-b);max_roundoff=max(max_roundoff,d);require(d<=ROUND_OFF,'Aggregate differs: '+label)
    reference=None
    for seed,ref in zip(SEEDS,models['fits']):
        raw=files[f'union/models/{seed}/fit.json'];require(sha(raw)==ref['sha256'],'Fit reference changed')
        fit=json.loads(raw);m=new_models[seed]
        require(all(m[k]==v for k,v in fit.items()),'Report changed saved fit metadata')
        require(fit['seed']==seed and fit['features']==selected and fit['raw_features']==132 and fit['transformed_features']==267 and fit['parameters']==21377,'New model shape/schema differs')
        require(fit['test_used_for_fit']is False and fit['provenance']['test_used_for_fit']is False,'Test fitted')
        ps=points[seed];pp=parent_points[seed]
        require(len(ps)==len(pp)==224,'Wrong scored sample count')
        keys=[(r['query_id'],r['policy_seed'],r['no_hvi_label'])for r in ps]
        require(len({x[0]for x in keys})==224,'Repeated query within one model')
        if reference is None:reference=keys
        require(keys==reference==[(r['query_id'],r['policy_seed'],r['no_hvi_label'])for r in pp],'New/parent action order or outcome differs')
        for r in ps:
            q=queries[r['query_id']];require(r['policy_seed']==q['seed']and r['no_hvi_label']==q['no_hvi'],'Original physical association differs')
        require(Counter(r['no_hvi_label']for r in ps)=={0:6,1:218},'Outcome denominator differs')
        for kind in ['raw','calibrated']:
            y=[r['no_hvi_label']for r in ps];p=[r[kind]for r in ps]
            value=metrics(y,p);compare(value,m['test'][kind],f'union/{seed}/{kind}')
            calculated['sampling_hidden',seed,kind]=value
            parent_value=metrics([r['no_hvi_label']for r in pp],[r[kind]for r in pp]);compare(parent_value,parent_models[seed]['test'][kind],f'parent_all/{seed}/{kind}')
            calculated['all',seed,kind]=parent_value
            metric_rows.append(dict(family='sampling_hidden',NN_seed=seed,calibration=kind,**value))
            metric_rows.append(dict(family='original_all',NN_seed=seed,calibration=kind,**parent_value))
            bins.extend(dict(NN_seed=seed,calibration=kind,population='selected_test',policy_seed='all',**b)for b in calibration_bins(y,p))
            dp=fit['development_predictions'][kind];require(len(dp)==50,'Missing dev score')
            dv=metrics(devy,dp);compare(dv,fit['development'][kind],f'union_dev/{seed}/{kind}')
            development_metrics.append(dict(NN_seed=seed,calibration=kind,**dv))
            bins.extend(dict(NN_seed=seed,calibration=kind,population='development',policy_seed='all',**b)for b in calibration_bins(devy,dp))
            for policy_seed in range(5101,5106):
                sub=[r for r in ps if r['policy_seed']==policy_seed];sv=metrics([r['no_hvi_label']for r in sub],[r[kind]for r in sub])
                compare(sv,m['policy_seed_strata'][str(policy_seed)][kind],f'union/{seed}/{policy_seed}/{kind}')
                strata.append(dict(NN_seed=seed,calibration=kind,policy_seed=policy_seed,**sv))
                bins.extend(dict(NN_seed=seed,calibration=kind,population='selected_test',policy_seed=str(policy_seed),**b)for b in calibration_bins([r['no_hvi_label']for r in sub],[r[kind]for r in sub]))
        for i,r in enumerate(devrows):
            development.append(dict(NN_seed=seed,row_index=i,episode=r['episode'],prefix_hash=r['prefix_hash'],no_hvi_label=r['label'],**{k:fit['development_predictions'][k][i]for k in ['raw','calibrated','logits']}))
        best=math.inf;selected_epoch=None;stale=0
        for row in fit['history']:
            if row['development_bce']<best-1e-8:best=row['development_bce'];selected_epoch=row['epoch'];stale=0
            else:stale+=1
        require(selected_epoch==fit['selected_epoch']and len(fit['history'])==fit['epochs_executed']==fit['optimizer_steps'],'Fit selection/duration differs')
        require(len(fit['history'])==100 or stale==15,'Early stopping differs')
        training.append(dict(NN_seed=seed,selected_epoch=selected_epoch,optimizer_steps=fit['optimizer_steps'],temperature=fit['temperature'],raw_features=132,transformed_features=267,parameters=21377,checkpoint_sha256=fit['checkpoint']['sha256']))
        delta={k:calculated['all',seed,'calibrated'][k]-calculated['sampling_hidden',seed,'calibrated'][k]for k in ['auroc','brier','nll','ece']}
        prior=next(r for r in report['paired_differences']if r['seed']==seed)['original_all_minus_new_sampling_hidden']
        for k,x in delta.items():scalar(x,prior[k],f'paired/{seed}/{k}')
        paired.append(dict(NN_seed=seed,**{'all_minus_sampling_hidden_'+k:x for k,x in delta.items()},different_parameter_counts=True,descriptive_only_not_independent_physics=True))
    require(sum(r['optimizer_steps']for r in training)==180==report['actual_optimizer_steps'],'Optimizer cost differs')
    require(parent['actual_optimizer_steps']==746,'Parent recorded optimizer cost differs')
    stats=[]
    for family in ['sampling_hidden','all']:
        for kind in ['raw','calibrated']:
            for key in ['auroc','error_auprc','brier','nll','ece','risk_coverage_auc']:
                x=[calculated[family,seed,kind][key]for seed in SEEDS];mean=st.mean(x);variance=st.variance(x)
                original=(report['initialization_summary']if family=='sampling_hidden'else parent['initialization_summaries']['all'])[kind][key]
                scalar(mean,original['mean'],family+'/'+kind+'/'+key+'/mean');scalar(variance,original['sample_variance'],family+'/'+kind+'/'+key+'/variance')
                stats.append(dict(family=family,calibration=kind,metric=key,NN_seeds='1729;1730;1731',n=3,ddof=1,values=json.dumps(x),mean=mean,sample_variance=variance,SD=st.stdev(x)))
    pairs=[]
    for key in ['auroc','brier','nll','ece']:
        vals=[r['all_minus_sampling_hidden_'+key]for r in paired]
        pairs.append(dict(metric=key,direction='original_all_minus_sampling_hidden',n=3,ddof=1,mean=st.mean(vals),sample_variance=st.variance(vals),SD=st.stdev(vals),values=json.dumps(vals),positive=sum(x>0 for x in vals),negative=sum(x<0 for x in vals),lower_is_better=key!='auroc'))
    y=[r['no_hvi_label']for r in points[1729]];constants=[dict(predictor=name,**metrics(y,[p]*224))for name,p in [('constant_0_5',.5),('constant_training_prior',70/75)]]
    summary=dict(schema='sampling_hidden_saved_prediction_archive_v1',passed=True,new_models=3,parent_existing_models=12,cumulative_models=15,
        new_optimizer_steps_recorded=180,parent_optimizer_steps_recorded=746,cumulative_optimizer_steps_recorded=926,
        new_prediction_records=672,cumulative_prediction_records_on_same_actions=3360,unique_physical_actions=224,
        HVI_improvements=6,no_HVI_outcomes=218,training_rows=75,development_rows=50,
        calibration_bin_rows=len(bins),development_prediction_rows=len(development),individual_metric_comparisons=comparisons,
        maximum_absolute_metric_roundoff=max_roundoff,independent_math_roundoff_bound=ROUND_OFF,
        original_new_audit_passed=True,original_new_audit_comparison=audit['comparison'],historical_parent_audit_rc1_preserved=True,
        physical_selected_generation=0,physical_full_UQ_and_ES_base_curves_still_duplicate=True,
        new_fits_or_model_inferences_or_oracle_calls_by_archive=0,old_12_models_refit=False,
        stats=stats,paired=paired,paired_summary=pairs,constant_controls=constants,
        original_report_sha256=sha(files['union/report.json']),original_audit_sha256=sha(files['union/audit.json']),
        no_deployment_or_test_based_selection=True,variance_is_NN_seed_not_physical_replication=True,
        parameter_count_difference={'sampling_hidden':21377,'all':40961})
    require(len(bins)==420 and len(development)==150,'Wrong bin/dev matrix')
    outputs={'predictions.csv':csv_bytes(report['predictions']),'development_predictions.csv':csv_bytes(development),'development_metrics.csv':csv_bytes(development_metrics),
        'model_metrics.csv':csv_bytes(metric_rows),'policy_seed_strata.csv':csv_bytes(strata),'calibration_bins.csv':csv_bytes(bins),
        'training_metadata.csv':csv_bytes(training),'seed_statistics.csv':csv_bytes(stats),'paired_models.csv':csv_bytes(paired),
        'paired_summary.csv':csv_bytes(pairs),'constant_controls.csv':csv_bytes(constants),'summary.json':json_bytes(summary)}
    return outputs,summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args();out,summary=compute()
    for name,raw in out.items():
        if a.write:(D/name).write_bytes(raw)
        else:require((D/name).read_bytes()==raw,'Derived file differs: '+name)
    if not a.write:
        m=json.loads((D/'MANIFEST.json').read_text())
        for r in m['files']:
            f=D/r['path'];require(f.stat().st_size==r['bytes']and sha(f.read_bytes())==r['sha256'],'Archive manifest differs')
    print(json.dumps({k:v for k,v in summary.items()if k not in ['stats','paired','paired_summary','constant_controls']},sort_keys=True))

if __name__=='__main__':main()
