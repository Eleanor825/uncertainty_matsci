"""Four CPU fits maximum; training OOF selects duration and calibrates temperature."""
from dataclasses import asdict
from pathlib import Path
import copy,math,os
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit
import original_uncertainty as u
import warm_common as w


def runtime_record():
    import platform,scipy
    return {'python':platform.python_version(),'torch':str(torch.__version__),'torch_cuda':torch.version.cuda,'numpy':np.__version__,'scipy':scipy.__version__,'device':'cpu','threads':2}

def cpu_contract():
    w.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU fitting requires explicitly hidden CUDA')
    torch.set_num_threads(2)
    w.require(not torch.cuda.is_initialized(),'CUDA was initialized in CPU fitting process')

def eligible(dataset):
    rows=dataset['rows'];groups={r['episode'] for r in rows if r['role']=='train'}
    reasons=[];chosen=[r for r in rows if r['reason'] not in ('lhs','invalid_fallback')]
    included=[r for r in rows if r['included']]
    fraction=len(included)/len(chosen) if chosen else 0.
    if fraction<.9:reasons.append('native_graph_fraction_below_0.9')
    if len(groups)!=3:reasons.append('expected_three_train_episodes')
    train=[r for r in included if r['role']=='train'];dev=[r for r in included if r['role']=='dev']
    for g in sorted(groups):
        if {r['label'] for r in train if r['episode']==g}!={0,1}:reasons.append('heldout_training_episode_lacks_two_classes:'+g)
        if {r['label'] for r in train if r['episode']!=g}!={0,1}:reasons.append('training_fold_lacks_two_classes:'+g)
    if {r['label'] for r in dev}!={0,1}:reasons.append('independent_development_lacks_two_classes')
    counts={role:{'raw':sum(r['role']==role for r in rows),'included':sum(r['role']==role for r in included),
                    'no_hvi':sum(r['label'] for r in included if r['role']==role),
                    'reasons':{reason:sum(r['role']==role and r['reason']==reason for r in rows)
                               for reason in sorted({r['reason'] for r in rows})}} for role in ('train','dev')}
    return {'ready':not reasons,'reasons':reasons,'graph_fraction':fraction,'counts':counts},train,dev

def matrix(rows,names):return np.asarray([[r['features'][k] for k in names] for r in rows],dtype=np.float64)
def labels(rows):return np.asarray([r['label'] for r in rows],dtype=np.float64)
def nll(y,z):return float(np.mean(np.logaddexp(0,z)-y*z))
def cfg(plan):
    c=plan['NN'];return u.RiskTrainingConfig(label_kind='no_positive_pareto_hypervolume_increment_for_the_next_executed_query',
        **{k:c[k] for k in ('seed','hidden_width','epochs','batch_size','learning_rate','weight_decay','include_error_similarity')})

def fit_path(training,validation,names,config,epochs):
    """No development data accepted; validation is a held-out training episode."""
    torch.manual_seed(config.seed)
    preprocessing=u.TrainOnlyFeatures().fit(matrix(training,names),labels(training),names,[r['episode'] for r in training])
    x=torch.from_numpy(preprocessing.transform(matrix(training,names),names,config.include_error_similarity))
    y=torch.tensor(labels(training),dtype=torch.float32)
    weights=torch.tensor(u._episode_weights([r['episode'] for r in training]),dtype=torch.float32)
    dx=torch.from_numpy(preprocessing.transform(matrix(validation,names),names,config.include_error_similarity)) if validation else None
    model=u.RiskMLP(x.shape[1],config.hidden_width).cpu();optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    generator=torch.Generator(device='cpu').manual_seed(config.seed);states=[];logits=[];losses=[];steps=0
    for epoch in range(epochs):
        model.train()
        for batch in torch.randperm(len(x),generator=generator).split(config.batch_size):
            prediction=model(x[batch]);loss=(torch.nn.functional.binary_cross_entropy_with_logits(prediction,y[batch],reduction='none')*weights[batch]).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();steps+=1
        model.eval()
        with torch.no_grad():
            losses.append(float((torch.nn.functional.binary_cross_entropy_with_logits(model(x),y,reduction='none')*weights).mean()))
            if dx is not None:logits.append(model(dx).numpy().astype(np.float64).tolist())
        w.require(math.isfinite(losses[-1]) and (not logits or all(math.isfinite(v) for v in logits[-1])),'Nonfinite NN training/logit evidence')
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    return {'features':preprocessing.state(),'states':states,'heldout_logits':logits,'train_bce':losses,
            'epochs':epochs,'optimizer_steps':steps,'training_queries':[r['query_id'] for r in training],
            'heldout_queries':[r['query_id'] for r in validation],'config':asdict(config)}

def choose(folds,training,epochs):
    lookup={r['query_id']:r for r in training};curve=[]
    for epoch in range(epochs):
        losses=[nll(np.asarray([lookup[q]['label'] for q in f['heldout_queries']]),np.asarray(f['heldout_logits'][epoch])) for f in folds]
        curve.append({'epoch':epoch+1,'fold_nll':losses,'macro_nll':float(np.mean(losses))})
    selected=1;best=float('inf')
    for r in curve:
        if r['macro_nll']<best-1e-8:best=r['macro_nll'];selected=r['epoch']
    oof=[{'query_id':q,'label':lookup[q]['label'],'logit':z} for f in folds for q,z in zip(f['heldout_queries'],f['heldout_logits'][selected-1])]
    w.require(len(oof)==len(training) and len({r['query_id'] for r in oof})==len(training),'OOF membership is not exact')
    return selected,curve,oof

def temperature(oof):
    y=np.asarray([r['label'] for r in oof]);z=np.asarray([r['logit'] for r in oof])
    fit=minimize_scalar(lambda log_t:nll(y,z/np.exp(log_t)),bounds=(-4,4),method='bounded')
    w.require(fit.success and math.isfinite(fit.fun),'OOF temperature minimization failed')
    return float(np.exp(fit.x))

def model_from(fit,epoch,names,config,temp,training):
    risk=u.CalibratedRiskModel();risk.config=config;risk.features=u.TrainOnlyFeatures.from_state(fit['features'])
    risk.model=u.RiskMLP(fit['states'][epoch-1]['layers.0.weight'].shape[1],config.hidden_width)
    risk.model.load_state_dict(fit['states'][epoch-1]);risk.model.eval();risk.temperature=temp;risk.selected_epoch=epoch
    risk.history=[{'epoch':i+1,'training_bce':v} for i,v in enumerate(fit['train_bce'])]
    risk.provenance={'train_rows':len(training),'label_kind':config.label_kind,'test_used_for_fit':False,'independent_development_used_for_fit':False,
                     'training_group_hash':u._fingerprint([r['episode'] for r in training]),'calibration':'training_leave_episode_out_only'}
    return risk

def predictions(model,rows,names):
    data=matrix(rows,names);z=[]
    # Use the original one-candidate runtime batch shape, as a fixed scoring contract.
    for row in data:
        values=model.features.transform(row[None,:],names,model.config.include_error_similarity)
        with torch.no_grad():z.append(float(model.model(torch.from_numpy(values))[0]))
    p=expit(np.asarray(z,dtype=np.float32)/model.temperature)
    return [{'query_id':r['query_id'],'episode':r['episode'],'label':r['label'],'logit':a,'raw':float(expit(a)),'calibrated':float(b)} for r,a,b in zip(rows,z,p)]

def report(model,dev,names,training,selection,eligibility):
    pred=predictions(model,dev,names);y=labels(dev);p=np.asarray([r['calibrated'] for r in pred]);prior=float(np.mean(labels(training)))
    metrics=u.risk_metrics(y,p);baseline=u.risk_metrics(y,np.full(len(y),prior))
    gate=metrics['brier']<baseline['brier'] and metrics['nll']<baseline['nll'] and metrics['auroc'] is not None and metrics['auroc']>.5
    by_episode={g:u.risk_metrics([r['label'] for r in pred if r['episode']==g],[r['calibrated'] for r in pred if r['episode']==g]) for g in sorted({r['episode'] for r in pred})}
    return {'schema':'snar_warm_training_OOF_scalar_report_v1','complete':True,'eligibility':eligibility,'selection':selection,
        'development':{'calibrated':metrics,'raw':u.risk_metrics(y,[r['raw'] for r in pred]),'train_prior_constant':baseline,
            'constant_probability':prior,'by_episode':by_episode,'predictions':pred,'threshold':.5,'at_least_point5':int(np.sum(p>=.5))},
        'predictive_gate_passed':bool(gate),'policy_benefit_demonstrated':False,'controller_deployed':False,
        'new_test_calls':0,'new_oracle_calls_by_CPU_fit':0,'new_ES_updates':0,'test_data_read_for_fit':False,
        'independent_development_used_for_selection_or_calibration':False}

def run(dataset,plan,folder):
    cpu_contract();folder=Path(folder);ready,train,dev=eligible(dataset)
    if not ready['ready']:
        out={'schema':'snar_warm_training_data_insufficient_v1','complete':True,'status':'data_insufficient','eligibility':ready,'NN_fits':0,'optimizer_steps':0,'controller_deployed':False}
        w.publish(folder/'report.json',out);w.publish(folder/'completion.json',{'status':'data_insufficient','complete':True,'report':w.artifact(folder/'report.json'),'NN_fits':0});return out
    names=dataset['feature_names'];config=cfg(plan);groups=sorted({r['episode'] for r in train});folds=[];refs=[]
    for index,group in enumerate(groups):
        fit=fit_path([r for r in train if r['episode']!=group],[r for r in train if r['episode']==group],names,config,config.epochs)
        path=folder/f'fold_{index}.pt';torch.save(fit,path);refs.append(w.artifact(path));folds.append(fit)
    epoch,curve,oof=choose(folds,train,config.epochs);temp=temperature(oof)
    selection={'selected_epoch':epoch,'criterion':'macro training-episode OOF uncalibrated BCE; earliest 1e-8 tie','curve':curve,
               'OOF':oof,'temperature':temp,'temperature_fit_population':'training OOF only','folds':refs}
    w.publish(folder/'training_selection.json',selection)
    final=fit_path(train,[],names,config,epoch);torch.save(final,folder/'final_training.pt')
    model=model_from(final,epoch,names,config,temp,train);model.save(folder/'risk.pt')
    steps=sum(f['optimizer_steps'] for f in folds)+final['optimizer_steps'];w.require(steps<=400,'Registered optimizer cap exceeded')
    seal={'schema':'snar_warm_model_seal_before_dev_v1','NN_fits':4,'optimizer_steps':steps,'risk':w.artifact(folder/'risk.pt'),
          'final_training':w.artifact(folder/'final_training.pt'),'selection':w.artifact(folder/'training_selection.json'),
          'folds':refs,'independent_development_fit_uses':0,'GPU_calls':0}
    w.publish(folder/'model_frozen_before_development.json',seal)
    out=report(model,dev,names,train,selection,ready);out['NN_fits']=4;out['optimizer_steps']=steps
    w.publish(folder/'report.json',out);w.publish(folder/'completion.json',{'complete':True,'status':'fitted','report':w.artifact(folder/'report.json'),'model_seal':w.artifact(folder/'model_frozen_before_development.json')})
    return out

def audit(dataset,plan,folder):
    """Recompute saved small-state predictions/selection; never optimizer.step()."""
    cpu_contract();folder=Path(folder);saved=w.read(folder/'report.json');ready,train,dev=eligible(dataset)
    if not ready['ready']:
        w.require(saved['status']=='data_insufficient' and saved['eligibility']==ready and saved['NN_fits']==0,'Insufficient-data report changed')
        w.require(not list(folder.glob('*.pt')),'Ineligible data nevertheless fitted models');return {'passed':True,'status':'data_insufficient','fits':0}
    seal=w.read(folder/'model_frozen_before_development.json');selection=w.read(w.check(seal['selection']));config=cfg(plan);names=dataset['feature_names'];folds=[]
    lookup={r['query_id']:r for r in train}
    for ref in seal['folds']:
        f=torch.load(w.check(ref),map_location='cpu',weights_only=True)
        rows=[lookup[q] for q in f['heldout_queries']]
        w.require(set(f['heldout_queries']).isdisjoint(f['training_queries']),'OOF leakage')
        training=[lookup[q] for q in f['training_queries']]
        actual=u.TrainOnlyFeatures().fit(matrix(training,names),labels(training),names,[r['episode'] for r in training]).state()
        w.require(actual==f['features'],'Saved train-only preprocessing differs')
        for epoch in range(config.epochs):
            model=model_from(f,epoch+1,names,config,1.,training)
            values=model.features.transform(matrix(rows,names),names,config.include_error_similarity)
            with torch.no_grad():z=model.model(torch.from_numpy(values)).numpy().astype(np.float64).tolist()
            w.require(z==f['heldout_logits'][epoch],'Saved OOF logits differ from epoch tensors')
        folds.append(f)
    epoch,curve,oof=choose(folds,train,config.epochs)
    w.require(epoch==selection['selected_epoch'] and curve==selection['curve'] and oof==selection['OOF'] and temperature(oof)==selection['temperature'],'Train OOF selection/calibration changed')
    final=torch.load(w.check(seal['final_training']),map_location='cpu',weights_only=True)
    w.require(final['training_queries']==[r['query_id'] for r in train] and not final['heldout_queries'] and final['epochs']==epoch,'Final train membership/duration changed')
    actual=u.TrainOnlyFeatures().fit(matrix(train,names),labels(train),names,[r['episode'] for r in train]).state()
    w.require(actual==final['features'],'Final train-only preprocessing differs')
    model=u.CalibratedRiskModel.load(w.check(seal['risk']));expected=model_from(final,epoch,names,config,selection['temperature'],train)
    w.require(model.features.state()==expected.features.state() and model.temperature==expected.temperature and asdict(model.config)==asdict(expected.config),'Final saved model state differs')
    w.require(model.selected_epoch==expected.selected_epoch and model.provenance==expected.provenance and model.history==expected.history,'Saved model metadata differs')
    w.require(all(torch.equal(v,expected.model.state_dict()[k]) for k,v in model.model.state_dict().items()),'Saved final tensors differ')
    rebuilt=report(model,dev,names,train,selection,ready);rebuilt['NN_fits']=4;rebuilt['optimizer_steps']=sum(f['optimizer_steps'] for f in folds)+final['optimizer_steps']
    w.require(w.canonical(rebuilt)==w.canonical(saved),'Report does not exactly recompute')
    return {'passed':True,'status':'fitted','NN_optimizer_steps_by_audit':0,'saved_epoch_states_checked':3*config.epochs,'exact_report':True,'predictive_gate_passed':saved['predictive_gate_passed']}
