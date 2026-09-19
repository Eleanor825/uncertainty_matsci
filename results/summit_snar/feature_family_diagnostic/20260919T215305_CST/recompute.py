"""Portable Python-standard-library verification of saved SnAr risk predictions.

No torch/numpy/sklearn, model loading, fitting, graph, oracle, or network access.
The 1e-12 cross-implementation roundoff check below does not replace or relax the
separate original byte-exact report reconstruction acceptance.
"""
import argparse
from collections import Counter,defaultdict
import csv,gzip,hashlib,json,math,statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROUND_OFF=1e-12


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def digest(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def mean(values):return math.fsum(values)/len(values)


def metrics(labels,probabilities):
    require(len(labels)==len(probabilities)>0,'Missing labels/scores')
    require(all(type(y) is int and y in (0,1) for y in labels),'Invalid binary label')
    require(all(type(p) in (int,float) and math.isfinite(p) and 0<=p<=1 for p in probabilities),'Invalid probability')
    n=len(labels);positives=sum(labels);negatives=n-positives
    p1=[p for y,p in zip(labels,probabilities) if y];p0=[p for y,p in zip(labels,probabilities) if not y]
    auroc=mean([float(a>b)+.5*float(a==b) for a in p1 for b in p0]) if positives and negatives else None
    # Average precision at tied-score group endpoints, as in sklearn (not trapezoidal PR area).
    ranked=sorted(zip(probabilities,labels),key=lambda r:r[0],reverse=True)
    terms=[];tp=0;i=0
    while i<n:
        j=i+1
        while j<n and ranked[j][0]==ranked[i][0]:j+=1
        block=sum(y for _,y in ranked[i:j]);tp+=block
        if positives:terms.append((block/positives)*(tp/j))
        i=j
    ap=math.fsum(terms) if positives else None
    bins=defaultdict(list)
    for y,p in zip(labels,probabilities):bins[min(int(p*10),9)].append((y,p))
    ece=math.fsum(len(rs)/n*abs(mean([p for _,p in rs])-mean([y for y,_ in rs])) for rs in bins.values())
    order=sorted(range(n),key=lambda k:probabilities[k]);running=0;coverage=[]
    for count,k in enumerate(order,1):running+=labels[k];coverage.append(running/count)
    losses=[]
    for y,p in zip(labels,probabilities):
        p=max(1e-8,min(1-1e-8,p));losses.append(-math.log(p) if y else -math.log1p(-p))
    return {'n':n,'error_rate':positives/n,'auroc':auroc,'error_auprc':ap,
        'brier':mean([(p-y)**2 for y,p in zip(labels,probabilities)]),'nll':mean(losses),'ece':ece,
        'risk_coverage_auc':mean(coverage),'overconfident_error_rate_p_le_0_1':sum(y==1 and p<=.1 for y,p in zip(labels,probabilities))/n,
        'threshold':.5,'at_or_above_fixed_threshold':sum(p>=.5 for p in probabilities)}


def calibration_bins(labels,probabilities):
    """Fixed ten equal-width bins; empty cells remain explicitly unknown."""
    result=[]
    for b in range(10):
        chosen=[(y,p) for y,p in zip(labels,probabilities) if min(int(p*10),9)==b]
        result.append({'bin':b,'lower':b/10,'upper':(b+1)/10,'upper_inclusive':b==9,
            'count':len(chosen),'mean_predicted_no_HVI':mean([p for _,p in chosen]) if chosen else None,
            'observed_no_HVI_fraction':mean([y for y,_ in chosen]) if chosen else None})
    return result


def verify():
    inventory=json.loads((HERE/'source_manifest.json').read_text())
    files={};originals={}
    for ref in inventory['gzip_scientific_sources']:
        path=HERE/ref['bundle_path'];require(path.resolve().is_relative_to(HERE.resolve()),'Nonportable path escape')
        zipped=path.read_bytes();require(sha(zipped)==ref['gzip_sha256'],'Compressed artifact changed')
        raw=gzip.decompress(zipped)
        require(len(raw)==ref['original_bytes'] and sha(raw)==ref['original_sha256'],'Uncompressed source bytes differ')
        files[ref['logical_name']]=raw
        if ref.get('original_path'):originals[ref['original_path']]=ref
    for ref in inventory['scientific_code']:
        require(sha((HERE/ref['bundle_path']).read_bytes())==ref['sha256'],'Scientific code changed')
    report=json.loads(files['report.json']);reg=json.loads(files['registration.json']);modelset=json.loads(files['all_models_frozen_before_test_scoring.json'])
    for name in ('registration.json','all_models_frozen_before_test_scoring.json','completion.json','original_reproduction_gate.json'):
        value=json.loads(files[name]);require(value['fingerprint']==digest({k:v for k,v in value.items() if k!='fingerprint'}),'Original seal differs: '+name)
    completion=json.loads(files['completion.json']);gate=json.loads(files['original_reproduction_gate.json'])
    require(report['registration_fingerprint']==reg['fingerprint']==modelset['registration_fingerprint']==completion['registration_fingerprint'],'Cross-scope result')
    for name,key in [('report.json','report'),('all_models_frozen_before_test_scoring.json','modelset'),('original_reproduction_gate.json','reproduction_gate')]:
        require(completion[key]['sha256']==sha(files[name]),'Completion source differs')
    require(gate['exact_model_tensors'] and gate['exact_training_metadata'] and gate['maximum_absolute_score_difference']==0 and gate['saved_action_scores']==224,'Original replication not exact')
    # Physical weight selection is separate from the twelve small-NN initialization seeds.
    context=json.loads(files['prior_and_selection.json']);es=json.loads(files['es_only_development.json'])['selection']['value']
    require(context['shared_prior_rows']==550 and context['full_checkpoint_selection']['selected']['generation']==0 and es['selected']['generation']==0,'G0 selection context differs')
    require(context['full_checkpoint_selection']['selected']['weight_hash']==es['selected']['weight_hash']=='6a308686126804830c81a7d2a3495feda3b3a13b074b9958e0e6026759826516','Selected physical weights differ')
    require(context['full_UQ_per_seed_HV_curves_equal'] and context['ES_only_base_per_seed_HV_curves_equal'],'Do not double-count identical observed arm curves')
    projection=json.loads(files['input_projection.json']);devrows=projection['risk_fit']['rows']['dev'];devy=[r['label'] for r in devrows]
    require(len(devy)==50 and Counter(devy)=={0:5,1:45},'Original development labels differ')
    queries={q['query_id']:q for q in projection['queries'] if q['arm']=='uq_esopt' and q['kind']=='llm_proposal'}
    require(len(queries)==224,'Changed executed scientific cohort')
    names=[(f,s) for f in ('all','sampling','hidden','graph') for s in (1729,1730,1731)]
    points=defaultdict(list)
    for row in report['predictions']:points[(row['family'],row['initialization_seed'])].append(row)
    require(set(points)==set(names) and len(report['predictions'])==2688,'Incomplete or duplicate model matrix')
    require(len(report['model_reports'])==12 and len(modelset['fits'])==12,'Incomplete source matrix')
    rows_by={(r['family'],r['seed']):r for r in report['model_reports']};require(set(rows_by)==set(names),'Duplicate model report')
    comparisons=0;max_delta=0.;deltas=[];recomputed=[];calibration=[]
    def compare(a,b,path):
        nonlocal comparisons,max_delta
        require(set(a)==set(b),'Metric schema differs: '+path)
        for k,v in a.items():
            old=b[k];comparisons+=1
            if v is None or old is None:require(v is old,'Undefined metric changed: '+path+'/'+k)
            elif type(v) is int:require(v==old,'Count differs: '+path+'/'+k)
            else:
                delta=abs(v-old);max_delta=max(max_delta,delta)
                require(delta<=ROUND_OFF,'Independent metric mismatch: '+path+'/'+k)
                if delta:deltas.append({'path':path+'/'+k,'absolute_difference':delta})
    reference_keys=None;actual_steps=0
    for (family,seed),fitref in zip(names,modelset['fits']):
        logical='models/'+family+'_'+str(seed)+'/fit.json';raw=files[logical]
        require(sha(raw)==fitref['sha256'],'Fit metadata byte identity differs')
        fit=json.loads(raw);m=rows_by[(family,seed)]
        require(all(m[k]==v for k,v in fit.items()),'Model report changed fit metadata')
        require(fit['test_read_for_fit'] is False and fit['provenance']['test_used_for_fit'] is False,'Test fitted')
        ps=points[(family,seed)];require(len(ps)==224,'Duplicate/missing action prediction')
        keys=[(r['query_id'],r['policy_seed'],r['no_hvi_label']) for r in ps]
        require(len({k[0] for k in keys})==224,'Duplicate physical identity within a model')
        if reference_keys is None:reference_keys=keys
        require(keys==reference_keys,'Cross-model action/order/label mismatch')
        for row in ps:
            q=queries[row['query_id']]
            require(row['episode']==q['episode'] and row['policy_seed']==q['seed'] and row['no_hvi_label']==q['no_hvi'],'Prediction/outcome association changed')
            require(row['original_saved_probability']==q['risk'],'Original saved score changed')
            if (family,seed)==('all',1729):require(row['calibrated']==q['risk'],'Original reproduction action score differs')
        calculated={};perseed={}
        for kind in ('raw','calibrated'):
            calibration.extend(dict(family=family,NN_seed=seed,calibration=kind,policy_seed='all',**b) for b in calibration_bins([r['no_hvi_label'] for r in ps],[r[kind] for r in ps]))
            calculated[kind]=metrics([r['no_hvi_label'] for r in ps],[r[kind] for r in ps]);compare(calculated[kind],m['test'][kind],family+'/'+str(seed)+'/test/'+kind)
            dp=fit['development_predictions'][kind];require(len(dp)==50,'Development prediction count differs')
            compare(metrics(devy,dp),fit['development'][kind],family+'/'+str(seed)+'/dev/'+kind)
            for scientific_seed in range(5101,5106):
                sub=[r for r in ps if r['policy_seed']==scientific_seed]
                calibration.extend(dict(family=family,NN_seed=seed,calibration=kind,policy_seed=str(scientific_seed),**b) for b in calibration_bins([r['no_hvi_label'] for r in sub],[r[kind] for r in sub]))
                value=metrics([r['no_hvi_label'] for r in sub],[r[kind] for r in sub])
                compare(value,m['policy_seed_strata'][str(scientific_seed)][kind],family+'/'+str(seed)+'/'+str(scientific_seed)+'/'+kind)
                perseed.setdefault(str(scientific_seed),{})[kind]=value
        best=math.inf;selected=None;stale=0
        for h in fit['history']:
            if h['development_bce']<best-1e-8:best=h['development_bce'];selected=h['epoch'];stale=0
            else:stale+=1
        require(selected==fit['selected_epoch'] and len(fit['history'])==fit['epochs_executed']==fit['optimizer_steps'],'Original duration/selection metadata inconsistent')
        require(len(fit['history'])==100 or stale==15,'Original stopping rule differs')
        actual_steps+=fit['optimizer_steps'];recomputed.append({'family':family,'seed':seed,'test':calculated,'per_policy_seed':perseed})
    require(actual_steps==746==report['actual_optimizer_steps'],'CPU update count differs')
    labels=[r['no_hvi_label'] for r in points[('all',1729)]];require(Counter(labels)=={0:6,1:218},'Physical outcome denominator differs')
    for key,p in [('constant_0_5',.5),('constant_train_no_hvi_prior',70/75)]:
        compare(metrics(labels,[p]*224),report['constant_controls'][key],key)
    for family in ('sampling','hidden','graph','all'):
        rs=[r for r in recomputed if r['family']==family]
        for kind in ('raw','calibrated'):
            for metric in ('auroc','error_auprc','brier','nll','ece','risk_coverage_auc'):
                vals=[r['test'][kind][metric] for r in rs];old=report['initialization_summaries'][family][kind][metric]
                require(old['n_defined']==3,'Seed count changed')
                require(abs(statistics.mean(vals)-old['mean'])<=ROUND_OFF and abs(statistics.variance(vals)-old['sample_variance'])<=ROUND_OFF,'Independent seed aggregate differs')
    paired=[]
    for family in ('hidden','graph','all'):
        values=[]
        for seed in (1729,1730,1731):
            a=next(r for r in recomputed if (r['family'],r['seed'])==(family,seed))['test']['calibrated']
            b=next(r for r in recomputed if (r['family'],r['seed'])==('sampling',seed))['test']['calibrated']
            values.append({'NN_seed':seed,'AUROC_difference':a['auroc']-b['auroc'],'Brier_difference':a['brier']-b['brier']})
        paired.append({'family_minus_sampling':family,'values':values,
            'AUROC_signs':dict(Counter('positive' if r['AUROC_difference']>0 else 'negative' if r['AUROC_difference']<0 else 'zero' for r in values)),
            'descriptive_pairing_only_not_IID_confidence':True})
    return {'schema':'snar_feature_family_portable_pointwise_verification_v2','passed':True,
        'source_manifest_sha256':sha((HERE/'source_manifest.json').read_bytes()),
        'original_report_sha256':sha(files['report.json']),'original_JSON_artifacts_verified':17,
        'physical_weights_selected_generation':{'full':0,'es_only':0},'physical_weight_hash':context['full_checkpoint_selection']['selected']['weight_hash'],
        'no_changed_ES_weights_explanation':True,'original_full_UQ_curves_equal':True,'original_ES_only_base_curves_equal':True,
        'unique_executed_actions':224,'model_prediction_records':2688,'models':12,'CPU_optimizer_steps':actual_steps,
        'training_rows':75,'development_rows':50,'outcome_counts':{'no_HVI':218,'HVI_improvement':6},
        'individual_metric_comparisons':comparisons,'maximum_absolute_metric_roundoff':max_delta,
        'independent_math_roundoff_bound':ROUND_OFF,'original_byte_exact_acceptance_unchanged':True,
        'all_policy_seed_strata_checked':True,'seed5105_AUROC_undefined':True,
        'paired_model_differences':paired,'results':recomputed,'calibration_bins':calibration,
        'calibration_bin_rows':len(calibration),'binning':'10 fixed equal-width bins, right-open except final includes1; no IID confidence intervals',
        'new_fits':0,'new_model_inferences':0,'new_LLM_graph_oracle_calls':0,
        'limits':'Posthoc same selected-policy cohort, six improvements, unequal model sizes; NN-seed pairing is descriptive, not independent physical evidence.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--calibration-csv',type=Path);args=p.parse_args();value=verify()
    raw=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as f:f.write(raw)
    if args.calibration_csv:
        with args.calibration_csv.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(value['calibration_bins'][0]),lineterminator='\n');writer.writeheader();writer.writerows(value['calibration_bins'])
    print(json.dumps({k:v for k,v in value.items() if k not in ('results','calibration_bins')},sort_keys=True),flush=True)


if __name__=='__main__':main()
