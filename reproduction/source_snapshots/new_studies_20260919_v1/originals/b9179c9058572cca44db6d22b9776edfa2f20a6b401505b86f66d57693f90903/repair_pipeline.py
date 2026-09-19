"""Explicit placement/output adapter around unchanged original scientific methods."""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
from types import SimpleNamespace
import common as c


def runtime_for(reg,path):
    original=c.read(c.check(reg['original_runtime']));runtime=c.read(path)
    c.require(set(runtime)==set(original) and all(runtime[k]==v for k,v in original.items() if k!='gpu_uuid'),
              'Only physical GPU UUID placement may differ from original runtime configuration')
    c.require(isinstance(runtime['gpu_uuid'],str) and runtime['gpu_uuid'].startswith('GPU-'),'Explicit GPU UUID required')
    return runtime


def create(reg,job,runtime,*,gpu):
    base=c.modules(reg)
    class RepairPipeline(base.Pipeline):
        def __init__(self):
            self.args=SimpleNamespace(stage=job['kind'],arm=job.get('arm'),branch=job.get('branch'))
            self.out=Path(reg['output']);self.protocol=c.read(reg['original_protocol']['path']);self.runtime=runtime
            self.event_path=self.out/'jobs'/job['job_id']/'events.jsonl'
            self.oracle=None;self.policy=self.adapter=self.optimizer=self.bank=self.attributor=self.risk=None
            self.graph_states=set();self.feature_names=[];self.weight_hash=None
            # No material call before the dedicated claim and source/runtime admission.
            if gpu:self.load_policy()
            if job['kind'] in ('evolve','evaluate'):
                scope='evolve_'+job['branch'] if job['kind']=='evolve' else 'evaluate_'+job['arm']+'__seed'+str(job['seed'])
                self.oracle=base.Oracle(self.runtime,self.out,scope,job['queries'])
        def episode(self,name,**kwargs):
            # Scientific episode implementation below remains the frozen original function.
            state=self.optimizer.model_state_hash() if self.policy is not None else None
            c.require(state==self.weight_hash,'Actual policy parameters drifted before episode')
            c.publish(self.out/'executed_policy_states'/f'{name}.json',dict(episode=name,actual_weight_hash=state,
                model_stamp=asdict(self.policy.model_stamp) if self.policy is not None else None,
                computed_from_actual_parameter_tensors=self.policy is not None),identical=True)
            return super().episode(name,**kwargs)
        def load_bank(self):
            from method.features import load_candidate_bank,make_attributor
            from matdiscovery.representation_training import GraphStageConfig
            selected=c.read(self.out/'representation_selected.json');report=c.read(self.out/'fresh_snar_transfer_fidelity.json')
            admission=c.receipt(reg,'bank_admission')
            self.graph_config=GraphStageConfig(**self.protocol['graph'])
            self.bank=load_candidate_bank(selected['path'],self.policy)
            c.require(self.bank.bank_fingerprint==selected['bank_fingerprint'] and report['passed'] is True
                and report['fingerprint']==selected['fidelity_fingerprint']
                and report['bank_fingerprint']==self.bank.bank_fingerprint,'Changed selected actual transfer proof')
            c.require(report['fingerprint']==c.digest({k:v for k,v in report.items() if k!='fingerprint'}),'Transfer report hash differs')
            self.bank.transfer_report=report
            self.attributor=make_attributor(self.policy,self.bank,config=self.graph_config)
    return RepairPipeline()


def selected_generation(p,branch):
    p.optimizer.replay_history(p.out/f'{branch}_selected_es_history.json')
    generation=len(p.optimizer.history);p.policy.mark_state('snar_fixed_test_checkpoint',generation=generation)
    p.weight_hash=p.optimizer.model_state_hash()
    c.require(p.weight_hash==c.read(p.out/f'{branch}_checkpoint_selection.json')['selected']['weight_hash'],
              'Actual clean selected-weight replay differs')
    c.publish(p.out/f'{branch}_selected_clean_replay.json',dict(verified=True,weight_hash=p.weight_hash,generation=generation),identical=True)


def bank_admission(reg,p):
    tc_r,tc_t=c.tc_modules(reg['tc_registry']['path']);tc=tc_r.read_registry(reg['tc_registry']['path'])
    from resource_bank import load
    bank_ref,terminal_ref=load(reg,tc);bankpath=c.check(bank_ref)
    from method.features import load_candidate_bank,validate_transfer_on_shards
    bank=load_candidate_bank(bankpath,p.policy)
    inputs=c.read(tc['input_manifest']['path'])
    data={split:{row['layer_path']:[x['shard']['path'] for x in row[split]] for row in inputs['layers']} for split in ('train','dev')}
    report=validate_transfer_on_shards(bank,data['train'],data['dev'])
    c.publish(p.out/'fresh_snar_transfer_fidelity.json',report)
    c.require(report['passed'] is True,'Actual full-corpus transfer gate failed')
    c.publish(p.out/'representation_selected.json',dict(path=str(bankpath),bank_fingerprint=bank.bank_fingerprint,fidelity_fingerprint=report['fingerprint']))
    c.publish(p.out/'representation_dev_prefix_filter.json',dict(rejected_proposal_directories=[],test_used=False))
    return [p.out/'fresh_snar_transfer_fidelity.json',p.out/'representation_selected.json',p.out/'representation_dev_prefix_filter.json',bankpath,Path(terminal_ref['path'])]


def eligible(reg):
    out=Path(reg['output']);diag=c.read(reg['original_diagnostic']['path']);items=[]
    for source in diag['prefixes']:
        original=Path(source['generation']['path']);generation=out/original.relative_to(reg['original_output'])
        items.append(dict(index=source['index'],split=source['split'],episode=source['episode'],label=source['label'],
            generation=generation,query=out/Path(source['query']['path']).relative_to(reg['original_output']),
            graph=generation.parent/'risk_graph.json',prefix_hash=source['original_generation_prefix_hash']))
    c.require(len(items)==125 and [x['index'] for x in items]==list(range(125)),'Incomplete125-prefix graph population')
    return items


def original_probe(reg,p,*,save_graph):
    from method.features import validate_snar_backend
    p.load_bank();item=eligible(reg)[0]
    c.require(item['episode']=='collection_train_1101' and item['generation'].parent.name=='query005_proposal0',
              'Do not substitute a different finite-difference probe')
    saved=c.read(item['generation']);generation=p.hydrate(saved['generation'])
    if save_graph:
        features,metadata=p.graph(generation,item['graph'].parent)
        result=dict(available=True,features=features,metadata=metadata)
        c.publish(item['graph'],result)
        return [item['graph'],Path(metadata['graph_file'])]
    report=validate_snar_backend(p.attributor,p.policy,generation,config=p.graph_config)
    p.graph_states.add(p.policy.model_stamp.state_id)
    c.publish(p.out/'jobs'/p.args.job_id/'original_probe_revalidation.json',report)
    return report


def graph_chunk(reg,p,index):
    original_probe(reg,p,save_graph=False)
    paths=[]
    for row in eligible(reg):
        if row['index']%8!=index:continue
        if row['index']==0:
            # This exact graph was generated by the mandatory first-probe task.
            c.receipt(reg,'original_failed_probe');paths.append(row['graph']);continue
        c.require(not row['graph'].exists(),'Existing graph without completed chunk; reconciliation required')
        saved=c.read(row['generation'])
        try:
            values,metadata=p.graph(p.hydrate(saved['generation']),row['graph'].parent)
            result=dict(available=True,features=values,metadata=metadata)
        except Exception as exc:result=dict(available=False,error=str(exc),type=type(exc).__name__)
        c.publish(row['graph'],result);paths.append(row['graph'])
    marker=p.out/'jobs'/p.args.job_id/'graph_partition.json'
    c.publish(marker,dict(indices=[x['index'] for x in eligible(reg) if x['index']%8==index],
        total_population=125,old_graphs_used=0,graphs=[c.artifact(x) for x in paths]))
    return [marker,*paths]


def fit_risk(reg):
    # Exact original fit tail: no policy/Oracle construction and no outcome features.
    c.modules(reg)
    import numpy as np
    import torch
    from matdiscovery.uncertainty import CalibratedRiskModel,RiskTrainingConfig,risk_metrics
    torch.set_num_threads(2);out=Path(reg['output']);protocol=c.read(reg['original_protocol']['path'])
    rows={'train':[],'dev':[]};okay=0;total=125;dev_total=50
    for item in eligible(reg):
        graph=c.read(item['graph'])
        if not graph['available']:continue
        okay+=1;rows[item['split']].append(dict(features=graph['features'],label=item['label'],episode=item['episode'],
            query_path=str(item['query']),graph_path=str(item['graph']),prefix_hash=item['prefix_hash']))
    c.require(okay/total>=.9,'SnAr native training graph availability below registered90%')
    names=sorted({k for row in rows['train'] for k in row['features']})
    arrays={s:np.array([[row['features'].get(k,float('nan')) for k in names] for row in rs]) for s,rs in rows.items()}
    ys={s:np.array([row['label'] for row in rs]) for s,rs in rows.items()};cfg=protocol['risk']
    config=RiskTrainingConfig(label_kind=cfg['label'],**{k:cfg[k] for k in
        ('seed','hidden_width','epochs','batch_size','learning_rate','weight_decay','patience','include_error_similarity')})
    risk=CalibratedRiskModel().fit(arrays['train'],ys['train'],arrays['dev'],ys['dev'],feature_names=names,
        train_groups=[x['episode'] for x in rows['train']],dev_groups=[x['episode'] for x in rows['dev']],
        train_episode_ids=[x['episode'] for x in rows['train']],config=config)
    risk.save(out/'snar_risk.pt');probabilities=risk.predict_proba(arrays['dev'],names)
    report=dict(feature_names=names,rows=rows,graph_fraction=okay/total,provenance=risk.provenance,
        development_metrics=risk_metrics(ys['dev'],probabilities),
        random_controller_retry_probability=float((np.sum(probabilities>=.5)+dev_total-len(rows['dev']))/dev_total),
        random_controller_rate_population=dict(valid_development_actions_after_prefix_filter=dev_total,
            available_graphs=len(rows['dev']),unavailable_graphs_counted_as_retry=dev_total-len(rows['dev'])),test_used=False)
    c.publish(out/'risk_fit.json',report)
    return [out/'risk_fit.json',out/'snar_risk.pt']


def execute(reg,job,runtime):
    out=Path(reg['output']);kind=job['kind'];p=None;completed=False
    try:
        if kind=='risk':return fit_risk(reg)
        if kind=='report':
            import acceptance_repair as a
            for arm in c.ARMS:
                c.publish(out/f'test_{arm}_complete.json',dict(episodes=5,oracle_queries=250),identical=True)
            report=a.accept_study(out,c.read(reg['original_protocol']['path']))
            report.update(imported_collection_oracle_calls=150,new_adaptation_oracle_calls=400,new_test_oracle_calls=1750,
                          representation_training_registry=reg['tc_registry'],old_graphs_reused=0)
            report['fingerprint']=c.digest({k:v for k,v in report.items() if k!='fingerprint'})
            c.publish(out/'acceptance.json',report)
            from reporting import build
            return [out/'acceptance.json',*build(reg,report)]
        p=create(reg,job,runtime,gpu=job['resource']=='gpu');p.args.job_id=job['job_id']
        if kind=='evolve':
            p.evolve(job['branch'])
            paths=[out/f"{job['branch']}_evolution_complete.json",out/f"{job['branch']}_checkpoint_selection.json"]
        elif kind=='bank':paths=bank_admission(reg,p)
        elif kind=='probe':paths=original_probe(reg,p,save_graph=True)
        elif kind=='graphs':paths=graph_chunk(reg,p,job['chunk'])
        elif kind=='freeze':
            p.freeze();paths=[out/'shared_prior_archive.json',out/'adaptation_complete.json']
        elif kind=='evaluate':
            prior=c.read(out/'shared_prior_archive.json')
            if job['arm'] in ('uq_esopt','uq_only'):p.load_method()
            if job['arm'] in ('uq_esopt','es_only'):selected_generation(p,'full' if job['arm']=='uq_esopt' else 'es_only')
            p.episode(job['job_id'],arm=job['arm'],seed=job['seed'],budget=50,prior=prior)
            paths=[out/'episodes'/job['job_id']/'summary.json']
        else:raise ValueError('Unregistered job kind')
        completed=True;return paths
    finally:
        if p:
            if p.oracle:p.oracle.close(completed=completed)
            if p.policy:
                import torch
                p.event('gpu_peak',allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
