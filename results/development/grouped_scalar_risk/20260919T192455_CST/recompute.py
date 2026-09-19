"""Recompute exported CPU diagnostic statistics; stdlib only, no fit/model/oracle call."""
from pathlib import Path
import argparse,csv,hashlib,io,json,math
from collections import Counter
from statistics import mean
HERE=Path(__file__).resolve().parent
MODELS=('full_internal','external_only_control')
STEPS=(0,2,4,8,16,32,64,128,200)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b,tol=2e-12):
    assert abs(a-b)<=tol*max(1.,abs(a),abs(b)),(a,b,tol)
def sigmoid(z):return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
def metrics(y,p,z=None):
    n=len(y);assert n and len(p)==n and set(y)=={0,1} and all(math.isfinite(a) and 0<=a<=1 for a in p)
    positives=[p[i] for i in range(n) if y[i]];negatives=[p[i] for i in range(n) if not y[i]]
    auc=sum((a>b)+.5*(a==b) for a in positives for b in negatives)/(len(positives)*len(negatives))
    ece=0.
    for b in range(10):
        idx=[i for i,a in enumerate(p) if min(int(a*10),9)==b]
        if idx:ece+=len(idx)/n*abs(mean(p[i] for i in idx)-mean(y[i] for i in idx))
    risk_sum=0.;fail=0
    for k,i in enumerate(sorted(range(n),key=lambda i:p[i]),1):fail+=y[i];risk_sum+=fail/k
    tp=0;count=0;ap=0
    for threshold in sorted(set(p),reverse=True):
        ids=[i for i,a in enumerate(p) if a==threshold];new=sum(y[i] for i in ids)
        tp+=new;count+=len(ids);ap+=new/len(positives)*tp/count
    if z is None:
        clipped=[max(1e-8,min(1-1e-8,a)) for a in p]
        nll=-mean(t*math.log(a)+(1-t)*math.log1p(-a) for t,a in zip(y,clipped))
    else:nll=mean(max(a,0)+math.log1p(math.exp(-abs(a)))-t*a for t,a in zip(y,z))
    return {'n':n,'error_rate':mean(y),'auroc':auc,'error_auprc':ap,'brier':mean((a-t)**2 for t,a in zip(y,p)),
            'nll':nll,'ece':ece,'risk_coverage_auc':risk_sum/n,'overconfident_error_rate_p_le_0_1':mean(int(t==1 and a<=.1) for t,a in zip(y,p)),
            'probability_min':min(p),'probability_max':max(p),'n_at_or_above_fixed_point6':sum(a>=.6 for a in p)}
def csv_bytes(rows):
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return stream.getvalue().encode()
def json_bytes(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def analyze(e):
    report=e['report']['value'];reg=e['registration'];rows=e['input_rows'];n=len(rows)
    assert n==386 and len({r['decision_id'] for r in rows})==386
    assert digest([(r['decision_id'],r['episode_id'],r['group_id'],r['split'],r['label_future_failure']) for r in rows])==e['accounting']['row_identity_hash']
    assert all(r['split'] in ('train','dev') and r['task_id']==('Al-Au-Hf' if r['split']=='train' else 'Al-Pd-Sm') for r in rows)
    for r in rows:assert r['scalar_fit_mask']==(r['label_future_failure'] is not None and r['fit_exclusion'] is None)
    tr=[r for r in rows if r['split']=='train' and r['scalar_fit_mask']];dev=[r for r in rows if r['split']=='dev' and r['scalar_fit_mask']]
    assert len(tr)==293 and len(dev)==83 and all(r['graph_available'] for r in tr+dev)
    y=[r['label_future_failure'] for r in dev];assert sum(y)==42 and sum(r['label_future_failure'] for r in tr)==204
    for split,selected in [('train',tr),('dev',dev)]:
        allrows=[r for r in rows if r['split']==split];c=reg['cohorts'][split]
        assert c==report['cohorts'][split]==e['accounting']['cohorts'][split]
        assert c['all_proposals']==len(allrows) and c['included_scalar_rows']==len(selected)
        assert c['null_future_labels']==sum(r['label_future_failure'] is None for r in allrows)
        assert c['all_proposal_graph_missing']==sum(not r['graph_available'] for r in allrows)
        assert c['included_membership_hash']==digest(sorted(r['decision_id'] for r in selected))
        assert c['included_class_counts']==dict(Counter(str(r['label_future_failure']) for r in selected))
    assert reg['step_candidates']==list(STEPS) and reg['config']['seed']==1729
    assert len(reg['feature_names'])==298 and reg['config']['label_kind']=='future_failure'
    assert reg['CPU_only'] and not reg['controller_deployment'] and reg['new_oracle_calls']==0
    assert all(v['returncode']==0 for v in e['scientific_stage_status'].values())
    candidates=[];controls=[];selection=[];predictions=[];metric_rows=[];updates=[];summary_models={};features=[]
    base_prior=mean(r['label_future_failure'] for r in tr)
    for name in MODELS:
        data=e['models'][name];result=report['models'][name]
        assert not result['controller_updated'] and not result['production_admission']
        names=result['feature_names'];assert names==[x for x in reg['feature_names'] if name=='full_internal' or x.startswith(('sampling.','action.'))]
        assert len(names)==(298 if name=='full_internal' else 15)
        features.extend({'model':name,'column':i,'feature_name':x} for i,x in enumerate(names))
        foldvalues={s:[] for s in STEPS};constantvalues=[]
        for fold in data['folds']:
            j=fold['fold'];f=reg['folds'][j];held=[r for r in tr if r['episode_id']==f['heldout_training_episode']];fit=[r for r in tr if r['episode_id']!=f['heldout_training_episode']]
            assert len(fit)==f['fit_rows'] and len(held)==f['heldout_rows']
            assert digest(sorted(r['decision_id'] for r in fit))==f['fit_membership_sha256'] and digest(sorted(r['decision_id'] for r in held))==f['heldout_membership_sha256']
            h=fold['updates'];assert h['count']==200 and h['indices_contiguous'] and h['last']['optimizer_update']==200 and h['first']['optimizer_update']==1
            assert h['last']['state_hash']==fold['states']['200']['model_state_hash']
            updates.append({'model':name,'fit':'fold_'+str(j),'optimizer_updates':h['count'],'last_epoch_started':h['last']['epoch'],'last_weighted_BCE':h['last']['weighted_BCE'],'last_state_hash':h['last']['state_hash'],'source_sha256':h['sha256']})
            assert set(fold['states'])=={str(s) for s in STEPS}
            for step in STEPS:
                state=fold['states'][str(step)];assert state['updates']==step
                for split in ('train','heldout_training_episode'):
                    m=state[split];assert m['n']==len(fit if split=='train' else held)
                    candidates.append({'model':name,'fold':j,'heldout_episode':f['heldout_training_episode'],'optimizer_updates':step,'split':split,**m,'model_state_hash':state['model_state_hash'],'source_sha256':state['sha256']})
                foldvalues[step].append(state['heldout_training_episode']['nll'])
            for control,p in [('half',.5),('fold_train_prior',mean(r['label_future_failure'] for r in fit))]:
                cm=metrics([r['label_future_failure'] for r in held],[p]*len(held));saved=fold['constant_controls']['value'][control]
                for k in saved:close(cm[k],saved[k])
                controls.append({'model':name,'fold':j,'control':control,'probability':p,**saved})
                if control=='fold_train_prior':constantvalues.append(cm['nll'])
        selected=min(STEPS,key=lambda step:(mean(foldvalues[step]),step));sel=data['selection']['value']
        assert selected==sel['optimizer_updates']==result['selected_updates'] and not sel['dev_used_for_selection']
        for row in sel['curve']:
            step=row['optimizer_updates'];assert row['fold_values']==foldvalues[step];close(mean(foldvalues[step]),row['macro_training_holdout_BCE'])
            selection.append({'model':name,'optimizer_updates':step,'macro_training_holdout_BCE':mean(foldvalues[step]),'fold_0_BCE':foldvalues[step][0],'fold_1_BCE':foldvalues[step][1],'fold_2_BCE':foldvalues[step][2],'selected':step==selected,'macro_fold_train_prior_BCE':mean(constantvalues),'macro_half_BCE':math.log(2)})
        h=data['final_updates'];assert h['count']==selected and h['last']['state_hash']==result['model_state_hash']
        updates.append({'model':name,'fit':'all_train_final','optimizer_updates':h['count'],'last_epoch_started':h['last']['epoch'],'last_weighted_BCE':h['last']['weighted_BCE'],'last_state_hash':h['last']['state_hash'],'source_sha256':h['sha256']})
        d=data['development']['value'];assert d['labels']==y and len(d['raw_logits'])==83
        assert not d['dev_used_for_weight_or_duration_selection'] and d['dev_calibration_metrics_are_in_sample'] and d['fixed_controller_threshold']==.6
        z=d['raw_logits'];temperature=d['temperature'];assert d['calibration_log_bounds']==[-4,4]
        assert 0<math.exp(4)-temperature<.001
        derivative_zero=mean(a*(.5-t) for t,a in zip(y,z));close(derivative_zero,d['dNLL_d_inverse_temperature_at_zero'])
        derivative_bound=mean(a*(sigmoid(a/math.exp(4))-t) for t,a in zip(y,z));assert derivative_bound>0
        for kind,T in [('raw',1.),('calibrated',temperature)]:
            zz=[a/T for a in z];pp=[sigmoid(a) for a in zz];actual=metrics(y,pp,zz)
            for k in ('n','error_rate','auroc','error_auprc','brier','nll','ece','risk_coverage_auc','overconfident_error_rate_p_le_0_1','probability_min','probability_max','n_at_or_above_fixed_point6'):close(actual[k],d[kind][k])
            if kind=='calibrated':
                for a,b in zip(pp,d['calibrated_probabilities']):close(a,b)
            metric_rows.append({'model':name,'scoring':kind,'selected_optimizer_updates':selected,'temperature':T,**actual,'data_role':'original_dev_in_sample_calibration' if kind=='calibrated' else 'original_dev_post_train_only_selection','provenance':'actual_saved_logits'})
        for i,r in enumerate(dev):predictions.append({'model':name,'dev_row_index':i,'decision_id':r['decision_id'],'episode_id':r['episode_id'],'label_future_failure':y[i],'raw_logit':z[i],'raw_probability':sigmoid(z[i]),'temperature':temperature,'calibrated_probability':d['calibrated_probabilities'][i],'old_scalar_cached_probability':r['scalar_probability'],'constant_half_probability':.5,'constant_train_prior_probability':base_prior})
        summary_models[name]={'selected_optimizer_updates':selected,'feature_count':len(names),'macro_training_holdout_BCE':mean(foldvalues[selected]),'macro_fold_train_prior_BCE':mean(constantvalues),'macro_half_BCE':math.log(2),'raw':d['raw'],'calibrated':d['calibrated'],'temperature':temperature,'positive_temperature_limit':math.exp(4),'dNLL_d_inverse_temperature_at_zero':derivative_zero,'dNLL_d_inverse_temperature_at_lower_bound':derivative_bound,'new_controller_deployed':False}
    for name,p in [('constant_half',.5),('constant_train_prior',base_prior)]:
        actual=metrics(y,[p]*len(y));saved=report['dev_controls'][name]
        for k in saved:close(actual[k],saved[k])
        metric_rows.append({'model':name,'scoring':'constant','selected_optimizer_updates':'','temperature':'',**actual,'data_role':'original_dev_control','provenance':'recomputed_from_labels'})
    old=metrics(y,[r['scalar_probability'] for r in dev]);saved=report['dev_controls']['original_scalar']
    # The previous public cache used float32 sigmoid, while the new engine scores saved raw logits in float64.
    for k in ('n','error_rate','auroc','error_auprc','brier','nll','ece','risk_coverage_auc','probability_min','probability_max','n_at_or_above_fixed_point6'):close(old[k],saved[k],tol=2e-7)
    metric_rows.append({'model':'original_scalar','scoring':'original_temperature','selected_optimizer_updates':'','temperature':e['original_scalar_temperature'],**{k:saved.get(k,old[k]) for k in old},'data_role':'original_dev_previously_used_for_original_selection_and_calibration','provenance':'actual_control_metrics_with_prior_public_float32_probabilities_checked_2e-7'})
    assert sum(r['optimizer_updates'] for r in updates)==report['actual_optimizer_updates_recorded']==1240
    assert report['neural_model_initializations_for_training']==8 and not report['controller_gain_demonstrated'] and not report['GPU_initialized'] and not report['production_model_selected_or_deployed']
    assert report['new_oracle_calls']==report['new_LLM_generation_calls']==0
    summary={'schema':'grouped_scalar_public_summary_v1','completed_CST':e['completed_CST'],'observed_CST':e['observed_CST'],'complete_CPU_diagnostic':True,'controller_gain_demonstrated':False,'new_controller_deployed':False,'original_NN_unchanged':True,'cohorts':report['cohorts'],'models':summary_models,'dev_controls':report['dev_controls'],'fold_candidate_states':54,'optimizer_updates':1240,'training_initializations':8,'new_oracle_calls':0,'new_LLM_generation_calls':0,'GPU_initialized':False,'elapsed_after_invocation_seconds':report['elapsed_after_invocation_seconds'],'dev_rows':83,'dev_events_are_not_83_independent_episodes':True,'fixed_training_seed':1729,'original_4B_or_G2_state_changed':False}
    coverage=[{'row_index':i,**r} for i,r in enumerate(rows)]
    verification={'passed':True,'raw_rows':386,'included_train':293,'included_dev':83,'null_labels':10,'all_graph_missing':9,'included_graph_missing':0,'candidate_state_summaries_checked':54,'candidate_train_and_holdout_metric_rows':108,'dev_new_logits_recomputed':166,'selected_durations':[summary_models[n]['selected_optimizer_updates'] for n in MODELS],'optimizer_update_count':1240,'old_scalar_probability_tolerance':2e-7,'all_other_probability_metric_tolerance':2e-12,'calibration_checked':'Saved T probabilities and exact metrics; positive derivative at minimum inverse T proves the convex NLL optimum is the upper registered T boundary. No new optimization/NN fit.','saved_candidate_tensor_check':'Source-bound production audit accepted all 54 states; public export contains state hashes, not tensors. Public checker does not independently recreate logits of unpublished candidate tensors.','raw_source_checks_not_reperformed_by_publication':True,'new_science_or_training_calls_for_export':0}
    outputs={'coverage.csv':csv_bytes(coverage),'fold_candidates.csv':csv_bytes(candidates),'fold_constants.csv':csv_bytes(controls),'duration_selection.csv':csv_bytes(selection),'development_predictions.csv':csv_bytes(predictions),'development_metrics.csv':csv_bytes(metric_rows),'update_summary.csv':csv_bytes(updates),'feature_names.csv':csv_bytes(features),'summary.json':json_bytes(summary),'verification.json':json_bytes(verification)}
    lines=['# Grouped scalar-risk CPU diagnostic — completed September 19, 19:24:55 CST','',
      '**This diagnostic did not demonstrate improved risk prediction or controller benefit.** Both prescribed models completed training and saved-state audit. On the original83 development rows, full internal features give AUROC0.495935 and external action/sampling features give0.430894. Original calibrated NN AUROC is0.516260. No model replaced the original NN, and no materials call, LLM generation, GPU workload or new materials result was produced by this CPU study.','',
      f"Completion: {e['completed_CST']}. Read-only observation: {e['observed_CST']}. Eight model initializations and 1,240 optimizer updates actually ran; the zero-fit counter of the separate observation/export must not be read as zero study training. The run plus its in-process saved-state verification took {report['elapsed_after_invocation_seconds']:.3f}s after invocation; this excludes preparation and the separate audit.",'',
      '## All prescribed models and controls','',
      '| Model/scoring | Selected updates | Dev AUROC | Brier | BCE/NLL | Predictions ≥0.6 |','|---|---:|---:|---:|---:|---:|']
    for row in metric_rows:lines.append(f"| {row['model']} / {row['scoring']} | {row['selected_optimizer_updates']} | {row['auroc']:.6f} | {row['brier']:.9f} | {row['nll']:.9f} | {row['n_at_or_above_fixed_point6']}/83 |")
    lines+=['','The two positive temperatures are54.597891 and54.597932, near the registered upper bound exp(4)=54.598150. Temperature scaling preserves ranks and AUROC. It drives the poor raw predictions toward0.5; both calibrated Brier scores remain slightly worse than the constant0.5 control. Their positive NLL derivative at the smallest allowed inverse temperature verifies that the upper temperature boundary is optimal within this registered one-parameter calibration family. Raw threshold crossings23/83 and41/83 are not evidence that those triggers are useful. Calibrated crossings are0/83; the threshold was not tuned to create crossings.','',
      '## Training-only duration selection','',
      'The three folds leave out one original Al-Au-Hf collection episode at a time. Both models use the same nine registered durations (0,2,4,8,16,32,64,128,200 updates), original64/64 GELU architecture, optimizer and training seed1729. External columns are selected before fold-local preprocessing and prototype fitting. The only duration criterion is the unweighted macro mean of three held-out training-episode raw BCEs, with the earlier step winning a tie. Original dev rows do not choose either duration.','',
      '| Feature condition | Columns | Selected updates | Selected macro CV BCE | Fold-train-prior constant | Constant0.5 |','|---|---:|---:|---:|---:|---:|']
    for name in MODELS:
        m=summary_models[name];lines.append(f"| {name} | {m['feature_count']} | {m['selected_optimizer_updates']} | {m['macro_training_holdout_BCE']:.6f} | {m['macro_fold_train_prior_BCE']:.6f} | {m['macro_half_BCE']:.6f} |")
    lines+=['','Internal features do not beat the fold-training-prevalence control on the selection score. External-only features slightly beat it on these folds but fail on the original development chemistry. All54 candidate states are retained in [fold_candidates.csv](fold_candidates.csv), including step0 and later overfitting; [duration_selection.csv](duration_selection.csv) contains all18 macro scores. These are related folds of one training chemistry, not three independently sampled chemical domains.','',
      '## Coverage and dependence','',
      'The386 original proposals are302 train and84 dev. Preserved future-failure labels admit293 train rows (204 positive,89 negative) and83 dev rows (42 positive,41 negative). Ten labels are unknown, including9 train and1 dev; they are excluded, not relabelled negative. Nine of all386 proposals lack graph support (8 train,1 dev); none of the376 fitted/scored rows lacks a graph. The [386-row coverage table](coverage.csv) preserves every proposal, original episode, chemical group, label/mask and graph availability. Its exact ordered identity hash matches the newly audited production accounting.','',
      'Three original training episodes use Al-Au-Hf, and the single original dev episode uses Al-Pd-Sm. These are the project’s original train/dev collection designations; the official MADE benchmark defines the30 test systems. Repeated proposal/event labels are preserved by the original scalar-training target;83 rows are not83 independent episodes. No episode-level or chemical-generalization confidence interval is claimed. Dev temperature calibration is in-sample, and this dev data had already been used by the original NN selection/calibration. The two feature conditions have different input parameter counts. Training seed1729 is fixed; no run-to-run variance is estimated.','',
      '## Reproduction and provenance','',
      'Run `python3 recompute.py` here. It checks all published hashes and regenerates the tables from [evidence.json](evidence.json), including independent AUROC/Brier/BCE/ECE/AP/risk-coverage calculations from the166 new saved logits and labels. It verifies fold membership and duration selection, calibration boundaries, constant controls and the386-row accounting. The original scalar control is checked against previously published float32 predictions within2e-7; the newly recorded control metrics are retained rather than rounded to those cached values.','',
      'The production audit separately loaded all54 saved candidate tensors and both final models and reproduced their predictions. This public checker verifies that audit’s recorded evidence and state hashes but does not re-create unpublished tensors or re-fit a model. Raw input/evaluation/receipt hashes remain distinct from derived export hashes in [provenance.json](provenance.json) and [data_source_refs.json](data_source_refs.json).','',
      'Byte-exact scientific [runner.py](diagnostic_source/runner.py), [engine.py](diagnostic_source/engine.py), [original proposal](diagnostic_source/proposal.json), and [executed source manifest](diagnostic_source/executed_source_manifest.json) are included. The original proposal records its earlier planning status; actual completion comes from the later registration/report/audit evidence. Historical absolute scientific storage/runtime paths in these sources require adaptation and access to the pinned inputs; they are not portable standalone launch commands. Operator/observer/authentication/process-control files and.pt tensors are excluded.','',
      'An [independent arithmetic review](independent_verified_results.json) agrees on all selected durations and new raw/calibrated development metrics; the public checker verifies this agreement. The study supplies a negative predictive diagnostic, not a materials-discovery comparison. It did not alter the original NN, G2, controller, or primary results. No new scientific call or NN fit was made to produce this export.','']
    outputs['report.md']='\n'.join(lines).encode()
    return outputs,verification

def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    e=json.loads((HERE/'evidence.json').read_text());outputs,check=analyze(e)
    peer=json.loads((HERE/'independent_verified_results.json').read_text())
    assert peer['snapshot_sha256']=='16b5ca6777b7ba0f398500b499f44b233c8628aed573168d031519cd9a12fe72'
    assert peer['actual_training']['optimizer_updates']==1240 and peer['cohorts']==e['report']['value']['cohorts']
    for name in MODELS:
        pm=peer['models'][name];em=e['models'][name]['development']['value']
        assert pm['selected_optimizer_updates']==e['models'][name]['selection']['value']['optimizer_updates']
        for arm,field in [('raw','raw_development'),('calibrated','calibrated_development')]:
            for key,value in pm[field].items():close(value,em[arm][key])
        close(pm['temperature'],em['temperature'])
    for name,data in outputs.items():
        if a.write:(HERE/name).write_bytes(data)
        else:assert (HERE/name).read_bytes()==data,'Recomputed bytes differ: '+name
    if not a.write:
        m=json.loads((HERE/'manifest.json').read_text())
        for ref in m['files']:
            pp=HERE/ref['path'];assert pp.is_file() and pp.stat().st_size==ref['bytes'] and hashlib.sha256(pp.read_bytes()).hexdigest()==ref['sha256'],ref['path']
    print(json.dumps(check,sort_keys=True))
if __name__=='__main__':main()
