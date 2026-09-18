"""CPU provenance tests: raw bank is fitted; diagnostic metrics are synthetic.

Synthetic diagnostic receipts test admission, not learned fidelity or material
outcomes. No GPU, policy or material oracle is invoked by this test module.
"""
import copy
import json
from pathlib import Path
import pytest
from matdiscovery.accounting import fingerprint, write_json_atomic, file_sha256
from matdiscovery.core_protocol import (CoreProtocolError, NORMALIZED_REGISTRATION, NORMALIZED_RECIPE,
    artifact, collection_jobs, collection_manifest_paths, registered_transcoder_epochs, registered_transcoder_recipe)
from matdiscovery.core_collection import import_training, collect_development, completed_core_collections, audit_core_corpus_costs
from matdiscovery.tc64_recovery import prepare_tc64_workspace
from matdiscovery.normalized_recovery import prepare_normalized_workspace, validate_normalized_registration, DIAGNOSTIC_LAYERS
from matdiscovery.deferred_dev_transcoders import precompute_layer
from matdiscovery.deferred_transcoder_publisher import _provider
from matdiscovery.representation_training import QWEN_MLP_PATHS, train_transcoders_from_collections, read_collections
from matdiscovery.transcoders import TranscoderConfig
from test_tc64_recovery import recovery


def sealed(path, data):
    data=copy.deepcopy(data);data['fingerprint']=fingerprint(data);write_json_atomic(path,data);return artifact(path)


@pytest.fixture
def normalized_case(recovery,tmp_path):
    f,old_args=recovery
    old=prepare_tc64_workspace(f['parent'],tmp_path/'raw64',**old_args)
    import_training(old);collect_development(old)
    import matdiscovery.core_representation as representation
    cfg=representation.transcoder_config(old);manifests=collection_manifest_paths(old)
    collection=read_collections(manifests,model_key='qwen35_4b');out=tmp_path/'raw64_training';pre=out/'layers';receipts=[]
    for i,layer in enumerate(QWEN_MLP_PATHS):
        paths=[x['path'] for x in collection['activation_shards'] if x['layer_path']==layer and x['split']=='train']
        precompute_layer(TranscoderConfig(64,64,128,64),paths,pre/f'layer_{i:02d}',policy_fingerprint=collection['checkpoint_hash'],
            layer_path=layer,seed=1729+i,epochs=64,batch_size=cfg.batch_size,learning_rate=cfg.learning_rate,max_dev_fvu=.5,device='cpu')
        receipts.append(artifact(pre/f'layer_{i:02d}/complete.json'))
    bankdir=tmp_path/'raw64_failed_bank'
    with _provider(pre):bank=train_transcoders_from_collections(manifests,bankdir,model_key='qwen35_4b',config=cfg)
    assert bank['complete'] and not bank['ready_for_graphs']
    write_json_atomic(out/'result.json',{'complete':True,'core_fingerprint':old['fingerprint'],'layers':32,'epochs_per_layer':64,'layer_receipts':receipts})
    for name in ('failure.json','selection.json'):write_json_atomic(out/name,{'synthetic_process_closure':True})
    closure=tmp_path/'closure.json'
    write_json_atomic(closure,{'schema':'tc64_fidelity_failure_closed_v1','complete':True,'closed':True,'core_fingerprint':old['fingerprint'],
        'failed_bank':artifact(bankdir/'transcoder_manifest.json'),'training_result':artifact(out/'result.json'),
        'queue_was_not_launched':True,'later_graph_risk_ES_test_not_started':True,'evidence_preserved':True,'new_policy_or_oracle_calls':0,
        'failure':artifact(out/'failure.json'),'selection':artifact(out/'selection.json')})
    diagnostics=tmp_path/'synthetic_diagnostics_not_training';diagnostics.mkdir()
    source=diagnostics/'source.txt';source.write_text('synthetic schema fixture; not an experimental implementation')
    raw128=diagnostics/'raw128';raw128.mkdir();write_json_atomic(raw128/'protocol.json',{'unit_only':True});rawcases=[]
    for i in DIAGNOSTIC_LAYERS:
        meta=copy.deepcopy(bank['layers'][i]['metadata']);meta['history'] += [{**copy.deepcopy(r),'epoch':64+e} for e,r in enumerate(meta['history'])];meta['epochs']=128
        item=sealed(raw128/f'layer_{i:02d}.json',{'complete':True,'index':i,'metadata':meta,'checkpoint':artifact(source),'sidecar':artifact(source)})
        rawcases.append({'index':i,'result':item})
    write_json_atomic(raw128/'summary.json',{'complete':True,'prefix64_exact':True,'scientific_main_result':False,'test_data_used':False,
        'all_four_fidelity_passed':False,'layers':rawcases,'protocol':artifact(raw128/'protocol.json')})
    guard=diagnostics/'guard';guard.mkdir();tasks=[]
    def task_for(i,d):
        ref=bank['layers'][i]['metadata'];c=json.loads((pre/f'layer_{i:02d}/complete.json').read_text())['contract']
        dev=ref['execution_provenance']['development_files']
        return {'layer_index':i,'core_fingerprint':old['fingerprint'],'output':str(d),'seed':1729+i,'config':ref['config'],
            'policy_fingerprint':c['policy_fingerprint'],'layer_path':c['layer_path'],'train_paths':c['train_paths'],
            'dev_paths':[item['path'] for item in dev],'sources':c['training']['files']+dev}
    for i in DIAGNOSTIC_LAYERS:
        d=guard/f'layer_{i:02d}';d.mkdir();task=task_for(i,d);tasks.append(task)
        write_json_atomic(d/'failure.json',{'complete':False,'task':task,'new_policy_or_oracle_calls':0,'error':"RuntimeError('Raw affine folding differs numerically')"})
        (d/'history.jsonl').write_text('{"epoch":0}\n')
    sealed(guard/'registration.json',{'schema':'registered_normalized64_diagnostic_v1','core_fingerprint':old['fingerprint'],
        'layers':DIAGNOSTIC_LAYERS,'epochs':64,'tasks':tasks,'source':artifact(source),'affine_proof_source':artifact(source)})
    boundary=diagnostics/'boundary';boundary.mkdir();write_json_atomic(boundary/'registration.json',{'unit_only':True})
    write_json_atomic(boundary/'result.json',{'fixed_epochs_complete':5,'new_policy_or_oracle_calls':0,'float64_refold_support_equal':True,
        'float64_refold_output_max_error':1e-16,'affected_rows':1})
    success=diagnostics/'success';success.mkdir();tasks=[];cases=[];audits=[]
    for i in DIAGNOSTIC_LAYERS:
        d=success/f'layer_{i:02d}';d.mkdir();ref=bank['layers'][i]['metadata'];c=json.loads((pre/f'layer_{i:02d}/complete.json').read_text())['contract']
        task=task_for(i,d);tasks.append(task)
        tr,dr=ref['provenance']['rows']['train'],ref['provenance']['rows']['dev'];metric={'rows':dr,'output_mse':.01,'output_fvu':.25,'fvu_undefined':0.}
        history=[{'epoch':e,'train_rows':tr,'train_raw_end_epoch':{'rows':tr},'dev_raw':metric} for e in range(64)]
        stats={'rows':tr,'scale_x':1.,'scale_y':.1,'sources':[artifact(p) for p in task['train_paths']],'tensor_file':artifact(source),
            'policy_fingerprint':ref['provenance']['policy_fingerprint'],'layer_path':ref['provenance']['layer_path']}
        write_json_atomic(d/'result.json',{'classification':'synthetic_fixture_not_model_measurement','complete':True,'epochs':64,'task':task,
            'fidelity_gate_passed':True,'original_FVU_threshold':.5,'test_data_used':False,'new_policy_or_oracle_calls':0,'history':history,
            'selected_epoch':0,'dev_raw':metric,'statistics':stats,'weights':artifact(source)})
        (d/'selected_normalized.pt').write_bytes(b'unit bytes; not a model')
        cases.append({'layer':i,'best_epoch':0,'dev_FVU':.25,'passed':True});audits.append({'layer':i,'float64_fixed_train_roundtrip':{'passed':True},
            'splits':{'train':{'rows':tr},'dev':{'rows':dr}},'evidence':[artifact(d/'result.json')]})
    registration=sealed(success/'registration.json',{'schema':'registered_normalized64_diagnostic_v2','core_fingerprint':old['fingerprint'],
        'epochs':64,'layers':DIAGNOSTIC_LAYERS,'workers':4,'tasks':tasks,'source':artifact(source),'affine_proof_source':artifact(source)})
    write_json_atomic(success/'summary.json',{'complete':True,'all_four_passed':True,'new_policy_or_oracle_calls':0,'cases':cases,'registration':registration})
    write_json_atomic(success/'selected_export_audit.json',{'complete':True,'registration':registration,'new_optimizer_steps':0,'new_policy_or_oracle_calls':0,'test_data_used':False,'layers':audits})
    args=dict(predecessor_workspace=old['workspace'],failed_bank_path=bankdir/'transcoder_manifest.json',failure_reconciliation_path=closure,
        prior_precompute_result=out/'result.json',raw128_summary=raw128/'summary.json',normalized_guard_registration=guard/'registration.json',
        fold_boundary_result=boundary/'result.json',normalized_diagnostic_summary=success/'summary.json',normalized_export_audit=success/'selected_export_audit.json',
        normalized_output_root=tmp_path/'future_normalized',normalized_bank_path=tmp_path/'future_bank')
    return f,old,args


def test_normalized_registration_preserves_four_corpus_and_all_old_gates(normalized_case,tmp_path):
    f,old,args=normalized_case;evidence={p:(file_sha256(p),p.stat().st_mtime_ns) for p in Path(old['workspace']).rglob('*') if p.is_file()}
    core=prepare_normalized_workspace(f['parent'],tmp_path/'normalized',**args)
    assert core['registration']==NORMALIZED_REGISTRATION
    assert registered_transcoder_recipe(core)==NORMALIZED_RECIPE and registered_transcoder_epochs(core)==64
    assert registered_transcoder_recipe(old)=='raw' and registered_transcoder_epochs(old)==64
    assert core['transcoder']==old['transcoder'] and core['graph']==old['graph'] and collection_jobs(core)==collection_jobs(old)
    import_training(core);collect_development(core,policy_factory=lambda *_:pytest.fail('No recollection'))
    assert len(completed_core_collections(core))==4
    cost=audit_core_corpus_costs(core)
    assert cost['total']['costs']['candidate_oracle_attempts']==200 and cost['new_development']['jobs']==0
    assert not any(cost['incremental_physical_costs'].values()) and len(core['normalization_amendment']['offline_history'])==6
    assert all('normalized_corpus_reuse' in json.loads(p.read_text()) for p in collection_manifest_paths(core))
    assert evidence=={p:(file_sha256(p),p.stat().st_mtime_ns) for p in evidence}
    assert prepare_normalized_workspace(f['parent'],tmp_path/'normalized',**args)==core
    changed=copy.deepcopy(core);changed['fit_recipe']='raw'
    with pytest.raises(CoreProtocolError):validate_normalized_registration(changed)
    changed=copy.deepcopy(core);changed['transcoder']['max_development_output_fvu']=.7
    with pytest.raises(CoreProtocolError,match='unregistered'):validate_normalized_registration(changed)
    from matdiscovery.normalized_recovery import _diagnostic_task
    reg=json.loads(Path(args['normalized_diagnostic_summary']).with_name('registration.json').read_text())
    task=copy.deepcopy(reg['tasks'][0]);task['dev_paths']=['invented-test-shard.pt']
    failed=json.loads(Path(args['failed_bank_path']).read_text())['layers'][1]['metadata']
    pre=json.loads(Path(failed['execution_provenance']['precomputation']['path']).read_text())['contract']
    with pytest.raises(CoreProtocolError,match='exact ordered'):_diagnostic_task(task,1,old,failed,pre)
    target=collection_manifest_paths(core)[-1].parent/'unregistered_partial.txt';target.write_text('unknown')
    with pytest.raises(CoreProtocolError,match='Unknown/partial'):completed_core_collections(core)


def test_normalized_admission_rejects_prior_test_and_changed_diagnostic(normalized_case,tmp_path):
    f,old,args=normalized_case;marker=Path(old['workspace'])/'experiments/core_es/started.json';write_json_atomic(marker,{'unit':True})
    with pytest.raises(CoreProtocolError,match='pre-test'):prepare_normalized_workspace(f['parent'],tmp_path/'blocked',**args)
    marker.unlink();marker.parent.rmdir();path=Path(args['normalized_diagnostic_summary']);record=json.loads(path.read_text())
    record['all_four_passed']=False;write_json_atomic(path,record)
    with pytest.raises(CoreProtocolError,match='all four'):prepare_normalized_workspace(f['parent'],tmp_path/'bad_diag',**args)
