"""Portable completed-result checks; no model, fitting, network or environment."""
from pathlib import Path
import argparse,csv,hashlib,json,math

HERE=Path(__file__).resolve().parent
def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(p.read_text())
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def private_check(value):
    if isinstance(value,dict):
        forbidden={'hostname','host','pid','child_pid','password','cookie','authorization','gpu_uuid','access_token','repo'}
        require(not(forbidden&set(value)),'Private runtime field in public projection')
        for v in value.values():private_check(v)
    elif isinstance(value,list):
        for v in value:private_check(v)
    elif isinstance(value,str):
        require(not any(x in value for x in ('/Users/','/mnt/','/root/','GPU-','mioffice.cn','de-41039-','tj-3041039-')),'Private path/device/host in public projection')

def main():
    p=argparse.ArgumentParser();p.add_argument('--repository-root',type=Path,default=HERE.parents[3]);a=p.parse_args()
    family=read(HERE/'four_feature_family_result.json');fixed=read(HERE/'fixed_probability_cached_analysis.json');cache=read(HERE/'cached_predictions.json');provenance=read(HERE/'provenance.json')
    for x in (family,fixed,cache,provenance):private_check(x)
    require(family['status']=='complete'and family['counts']=={'LLM_calls':0,'dev_forward_batches':4,'environment_calls':0,'fits_completed':4,'graph_extraction_calls':0,'optimizer_updates_returned':800},'Wrong completed diagnostic counts')
    require((family['fit_rows'],family['calibration_rows'],family['development_rows'],family['development_episodes'],family['development_worlds'])==(60,60,40,2,1),'Wrong data scope')
    require(family['no_new_graphs']is True and family['existing_graphs_reused']==160 and family['heldout_outcome_reads']==0 and family['production_controller_changed']is False,'Changed scope/production claim')
    require(family['feature_dimensions']=={'OutputAction':10,'NoGraph':140,'GraphOnly':153,'Internal':293},'Wrong feature families')
    require(set(family['metrics'])==set(family['models'])==set(family['feature_dimensions']),'Missing family')
    for name,model in family['models'].items():
        require(model['family']==name and model['updates']==200 and model['tensor_changed']is True and model['initial_tensor_hash']!=model['final_tensor_hash'],'NN update evidence incomplete')
        require(model['parameters']==128*family['feature_dimensions'][name]+4481,'Wrong original NN input/parameter count')
        require(model['checkpoint']['included_in_public_bundle']is False and len(model['checkpoint']['sha256'])==64,'Wrong checkpoint publication claim')
        m=family['metrics'][name];eps=list(family['episode_metrics'][name].values())
        require(len(eps)==2 and all(x['rows']==20 and x['episodes']==x['worlds']==1 for x in eps),'Wrong episode denominators')
        require(m['rows']==40 and m['episodes']==2 and m['worlds']==1 and -.000000000001<=m['auroc']<=1.000000000001,'Wrong metric scope/range')
        for k in ('brier','nll'):require(near(m[k],sum(e[k]for e in eps)/2),'Episode-weighted metric differs')
    for name,pair in [('GraphOnly_minus_OutputAction',('GraphOnly','OutputAction')),('Internal_minus_NoGraph',('Internal','NoGraph')),('Internal_minus_OutputAction',('Internal','OutputAction'))]:
        for k in ('auroc','brier','nll'):require(near(family['paired_feature_family_differences'][name][k],family['metrics'][pair[0]][k]-family['metrics'][pair[1]][k]),'Descriptive difference arithmetic differs')
    prior=read(a.repository_root/provenance['prior_control_snapshot'])
    for name in ('Internal','OutputAction'):
        require(family['control_reproduction'][name]is True,'Missing original control reproduction')
        old=prior['neural_risk']['models']['next_action_failure']['models'][name];new=family['models'][name]
        for k in ('initial_tensor_hash','final_tensor_hash','parameters','temperature','updates','threshold'):require(new[k]==old[k],'Original tensor/calibration reproduction differs')
        for k in ('sha256','bytes'):require(new['checkpoint'][k]==old['checkpoint'][k],'Original checkpoint bytes differ')
        for k in ('auroc','brier','nll'):require(near(family['metrics'][name][k],prior['neural_risk']['heads']['next_action_failure']['metrics'][name][k]),'Original development control differs')
    require(fixed['threshold_fixed_before_computation']==.5 and fixed['fitting']is False and fixed['test_data_used']is False,'Wrong fixed-threshold scope')
    require(cache['calls_repeated_for_publication']==0 and cache['test_data_read']is False,'New predictor calls/test use in export')
    counts={'calibration':60,'development':40,'G0_development':10};checks=0
    for partition,n in counts.items():
        rows=cache['partitions'][partition];require(len(rows)==n,'Wrong cached row count')
        require(len({(r['episode_id'],r['attempt_index'])for r in rows})==n,'Duplicate cached action identity')
        worlds={r['world_seed']for r in rows};require(worlds==({1}if partition=='calibration'else{2}),'Wrong source world')
        require(all(r['label']in(0,1)and math.isfinite(r['risk'])and 0<=r['risk']<=1 for r in rows),'Unknown/nonfinite prediction')
        for rule in ('original','fixed_posterior'):
            reported=fixed['partitions'][partition][rule];t=reported['threshold'];require(t==(cache['original_threshold']if rule=='original'else .5),'Threshold differs')
            trigger=sum(r['risk']>=t for r in rows);tp=sum(r['risk']>=t and r['label']==1 for r in rows);fp=trigger-tp;fn=sum(r['label']for r in rows)-tp
            expected={'rows':n,'trigger':trigger,'TP':tp,'FP':fp,'FN':fn,'recall':tp/(tp+fn),'brier':sum((r['risk']-r['label'])**2 for r in rows)/n}
            require(all(near(reported[k],v)for k,v in expected.items()),'Cached operating-point arithmetic differs');checks+=len(expected)
        require(near(fixed['partitions'][partition]['original']['brier'],fixed['partitions'][partition]['fixed_posterior']['brier']),'Threshold changed predictor Brier')
    csv_rows=list(csv.DictReader((HERE/'feature_metrics.csv').open()))
    require(len(csv_rows)==4,'Missing CSV family')
    for row in csv_rows:
        require(int(row['raw_features'])==family['feature_dimensions'][row['family']],'CSV dimensions differ')
        for k in ('auroc','brier','nll'):require(near(row[k],family['metrics'][row['family']][k]),'CSV metrics differ')
    for row in csv.DictReader((HERE/'cached_threshold_metrics.csv').open()):
        expected=fixed['partitions'][row['partition']][row['rule']]
        require(all(near(row[k],v)for k,v in expected.items()),'Cached CSV differs')
    training=read(HERE/'training_diagnostic.json');private_check(training)
    require(training['status']=='complete'and training['final_models_reported']==8 and training['NN_states_reported']==16,'Incomplete training diagnostic')
    require(training['counts']=={'LLM_calls':0,'NN_fits_completed':4,'NN_optimizer_updates':800,'classical_fits_completed':4,'environment_calls':0,'graph_extraction_calls':0,'test_outcome_reads':0},'Wrong training diagnostic counts')
    require(training['selects_no_production_model']is True and training['heldout_outcome_reads']==0 and training['development_rows']==40 and training['development_episodes']==2 and training['development_worlds']==1,'Training diagnosis changed production or scope')
    require(set(training['NN_learning_curves'])==set(family['models']),'Missing NN family curves')
    for name,states in training['NN_learning_curves'].items():
        require([s['step']for s in states]==[0,50,100,200],'Outcome-selected NN states')
        for key in ('auroc','brier','nll'):require(near(states[-1]['calibrated']['development'][key],family['metrics'][name][key]),'200-step diagnostic differs from original family fit')
    training_rows=list(csv.DictReader((HERE/'training_metrics.csv').open()));require(len(training_rows)==108,'Missing training CSV rows')
    identities=set()
    for row in training_rows:
        identity=tuple(row[k]for k in ('model','step','split','calibration'));require(identity not in identities,'Duplicate training metric row');identities.add(identity)
        if row['model'].startswith('NN_'):
            state=next(s for s in training['NN_learning_curves'][row['family']]if s['step']==int(row['step']))
            expected=state[row['calibration']][row['split']];temperature=1 if row['calibration']=='uncalibrated'else state['calibration']['temperature']
        else:
            require(row['calibration']=='calibrated'and row['step']=='','Unexpected classical metric kind')
            expected=training['final_metrics'][row['model']][row['split']];temperature=training['classical_fit_receipts'][row['model']]['calibration']['temperature']
        require(near(row['temperature'],temperature),'CSV temperature differs')
        require(all(near(row[k],expected[k])for k in ('nll','brier','auroc','rows','episodes','worlds')),'Training CSV metric differs')
    for row in csv.DictReader((HERE/'classical_fit_status.csv').open()):
        receipt=training['classical_fit_receipts'][row['model']]
        require(json.loads(row['iterations'])==receipt['n_iter'] and json.loads(row['warnings'])==receipt['warnings'],'Classical fit status differs')
        require((row['no_convergence_warning']=='True')==receipt['converged'],'Classical warning flag differs')
        if receipt['algorithm']=='GBC':require(int(row['fixed_trees'])==receipt['parameters']['n_estimators']==100,'Wrong fixed GBC budget')
    training_report=(HERE/'training_diagnostic_report.md').read_text();require('step 0' in training_report and 'error-prototype' in training_report,'Missing supervised preprocessing caveat')
    report=(HERE/'report.md').read_text();require('not a capacity-matched causal decomposition'in report and 'already executed actions'in report,'Missing interpretation limit')
    require(provenance['scientific_calls_for_export']==0 and provenance['online_improvement_claimed']is False and provenance['full_method_complete']is False,'Unsupported effect/completion claim')
    manifest=read(HERE/'manifest.json');require(manifest['schema']=='dw_completed_CPU_diagnostics_bundle_v1','Wrong bundle manifest')
    require({e['path']for e in manifest['files']}=={p.name for p in HERE.iterdir()if p.is_file()and p.name!='manifest.json'},'Bundle inventory differs')
    for e in manifest['files']:
        f=HERE/e['path'];require(f.stat().st_size==e['bytes']and sha(f)==e['sha256'],'Bundle bytes/hash changed')
    print(json.dumps({'status':'PASS','families':4,'control_reproductions':2,'cached_rows':110,'cached_numeric_checks':checks,'training_metric_rows':108,'NN_states':16,'final_training_models':8,'model_or_environment_calls':0}))
if __name__=='__main__':main()
