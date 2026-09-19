"""Admission for fresh causal128 SnAr transcoders, never a main-result receipt."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

SCHEMA = 'snar_causal128_fresh_normalized64_training_v3'
SUMMARY_SHA = 'fd311d8b7e741b7e0dfd9d56495229b2750c8e46f41391e4fa3d45e6d79176ac'
DIAGNOSTIC_SHA = '655d93fd990c15e915b411b70d7d431b678fab1cbe4fda9e39b6391e56a4faac'
LAYER_PATHS = [f'model.language_model.layers.{i}.mlp' for i in range(32)]
ARCHITECTURE = dict(input_dim=2560, output_dim=2560, feature_dim=5120, top_k=64)
SETTINGS = dict(epochs=64, batch_size=256, learning_rate=.0004, max_dev_fvu=.5,
                seed=1729, device='cuda:0', cpu_threads=2, gpu_memory_bytes=2*1024**3,
                fit_recipe='train_centered_scalar_rms_fp32_export_v1')
GATES = dict(max_dev_fvu=.5, native_graph_minimum_availability=.9,
             finite_difference_epsilon=.001, finite_difference_rtol=.05,
             finite_difference_atol=.0001, finite_difference_max_edges=4,
             source_selection_rule='qwen_final_mlp_causal_activation_v1')


def require(test, message):
    if not test: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    p=Path(path); before=p.stat(); h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    after=p.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(identity(before)==identity(after),'File changed during hashing: '+str(p))
    return h.hexdigest()


def artifact(path):
    p=Path(path).resolve(); return dict(path=str(p),sha256=sha(p))


def check(ref):
    require(Path(ref['path']).is_absolute() and sha(ref['path'])==ref['sha256'],'Changed artifact: '+ref['path'])
    return Path(ref['path'])


def publish(path, value):
    """Atomic, durable, exclusive publication; other workers never read half JSON."""
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name('.'+p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('x') as stream:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.link(temp,p)  # atomic O_EXCL semantics, including concurrent same-layer claimers
        fd=os.open(p.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if temp.exists():temp.unlink()


def sealed(value):
    return dict(value,fingerprint=digest(value))


def verify_seal(value):
    require(value.get('fingerprint')==digest({k:v for k,v in value.items() if k!='fingerprint'}),'Invalid fingerprint')


def verify_prefix(receipt, source, diagnostic, matrix_row):
    require(receipt['schema']=='snar_causal128_prefix_fvu_receipt_v1' and receipt['complete'] is True,
            'Incomplete causal activation receipt')
    require(receipt['source']==source and receipt['registration_fingerprint']==diagnostic['fingerprint'],
            'Different original prefix/diagnostic')
    require(receipt['generation_calls']==receipt['scientific_oracle_calls']==0
            and receipt['capture_parity_passed'] is True and receipt['not_a_native_graph_admission'] is True,
            'Capture is not the fixed read-only replay')
    require(receipt['finite_difference_passed'] is None and receipt['native_graph_available'] is None,
            'Diagnostic cannot certify graph availability')
    horizon=receipt['target']['causal_horizon']; positions=receipt['selected_token_positions']
    import torch
    expected=torch.linspace(0,horizon,128).long().unique(sorted=True).tolist()
    require(positions==expected and len(positions)==128 and 0<=horizon<receipt['complete_prefix_tokens'],
            'Causal128 positions changed')
    require(receipt['complete_prefix_tokens']==source['prefix_tokens'] and source['split'] in ('train','dev'),
            'Prefix length/split changed')
    rows=receipt['all_layer_fvu'];shards=receipt['shards']
    require(len(rows)==len(shards)==32,'Missing full-layer capture')
    for i,(row,item) in enumerate(zip(rows,shards)):
        require(row['layer_index']==item['layer_index']==i and row['layer_path']==item['layer_path']==LAYER_PATHS[i]
                and item['rows']==128 and row['threshold']==.5 and row['evaluated_positions']==horizon+1,
                'Layer/row/fidelity definition changed')
        value=row['output_fvu'];passed=row['fvu_undefined']==0 and math.isfinite(value) and value<=.5
        require(value==matrix_row[i] and row['passed'] is passed,'FVU matrix/pass flag disagrees')
    require(receipt['all_layer_fvu_passed'] is all(r['passed'] for r in rows),'Whole-prefix FVU flag disagrees')
    return receipt


def diagnostic_inputs(summary_path, *, verify_shards, verify_sidecars=True):
    summary_path=Path(summary_path).resolve()
    require(sha(summary_path)==SUMMARY_SHA,'Use the registered complete125-prefix diagnostic')
    summary=read(summary_path);dref=summary['registration']
    require(dref['sha256']==DIAGNOSTIC_SHA,'Unregistered diagnostic source')
    diagnostic=read(check(dref));verify_seal(diagnostic)
    require(summary['schema']=='snar_offline_all_prefix_fvu_and_causal128_v1' and summary['complete'] is True
            and summary['registration_fingerprint']==diagnostic['fingerprint']
            and summary['counts']==dict(train=75,dev=50,prefixes=125,layers=32)
            and summary['new_generation_calls']==summary['new_oracle_calls']==0
            and summary['training_performed'] is False and summary['FD_performed'] is False
            and summary['source_files_reverified'] is True and summary['original_model_weights_unchanged'] is True,
            'Diagnostic scope/completion changed')
    require(len(summary['prefix_receipts'])==len(summary['fvu_matrix'])==len(diagnostic['prefixes'])==125,
            'Incomplete diagnostic population')
    expected=[(split,f'collection_{split}_{seed}',f'collection_{split}_{seed}/q{q:03d}')
              for split,seeds in [('train',(1101,1102,1103)),('dev',(2101,2102))]
              for seed in seeds for q in range(5,30)]
    layers=[dict(layer_index=i,layer_path=p,train=[],dev=[]) for i,p in enumerate(LAYER_PATHS)]
    prefixes=[]; all_paths=set(); split_prefixes={'train':set(),'dev':set()}
    for i,(ref,source,matrix) in enumerate(zip(summary['prefix_receipts'],diagnostic['prefixes'],summary['fvu_matrix'])):
        require(source['index']==i and (source['split'],source['episode'],source['query_id'])==expected[i],
                'Missing/reordered/held-out prefix')
        receipt=verify_prefix(read(check(ref)),source,diagnostic,matrix)
        prefix=receipt['full_prefix_hash'];split_prefixes[source['split']].add(prefix)
        prefixes.append(dict(index=i,split=source['split'],episode=source['episode'],query_id=source['query_id'],
                             full_prefix_hash=prefix,receipt=ref,source=source))
        for item in receipt['shards']:
            require(item['shard']['path'] not in all_paths,'Duplicate activation shard')
            all_paths.add(item['shard']['path'])
            require(Path(item['shard']['path']).parent==Path(ref['path']).parent
                    and item['sidecar']['path']==item['shard']['path']+'.json','Shard leaves captured prefix directory')
            sidecar=read(check(item['sidecar'])) if verify_sidecars else None
            if verify_sidecars:require(sidecar['split']==source['split'] and sidecar['layer_path']==item['layer_path']
                    and sidecar['rows']==128 and sidecar['input_dim']==sidecar['output_dim']==2560
                    and sidecar['tensor_hash']==item['tensor_hash']
                    and sidecar['group_ids']==[source['episode']]*128 and sidecar['prefix_hashes']==[prefix]*128,
                    'Captured tensor sidecar differs from exact source identity')
            if verify_shards:check(item['shard'])
            layers[item['layer_index']][source['split']].append(dict(**item, prefix_index=i))
    require(not(split_prefixes['train'] & split_prefixes['dev']),'Train/dev prefix leakage')
    return summary,diagnostic,dict(schema='snar_causal128_all_layer_inputs_v1',prefixes=prefixes,layers=layers,
                                  rows_per_layer={'train':9600,'dev':6400},test_used=False)


def prepare(summary_path, workspace):
    workspace=Path(workspace).resolve()
    require(not workspace.exists(),'Training workspace must be new; never overwrite an existing attempt')
    summary,diagnostic,data=diagnostic_inputs(summary_path,verify_shards=True)
    oldroot=Path(diagnostic['original']).resolve();oldout=Path(diagnostic['original_output']).resolve()
    require(not workspace.is_relative_to(oldroot) and not workspace.is_relative_to(Path(diagnostic['output'])),
            'New outputs must be separate from original/diagnostic data')
    # Every original source remains authoritative; actual math is imported from here.
    for ref in diagnostic['original_sources']+diagnostic['own_sources']+diagnostic['fixed_evidence']:check(ref)
    oldbank=read(check(diagnostic['bank']));base=read(oldout/'base_policy.json')
    require(oldbank['complete'] is True and oldbank['passed_layers']==32
            and oldbank['checkpoint_hash']==base['checkpoint_hash'],'Original completed bank identity differs')
    require(oldbank['snar_request']['seed']==1729 and oldbank['configuration']==dict(epochs=64,batch_size=256,
                learning_rate=.0004,device='cuda:0',max_dev_fvu=.5),'Original optimization recipe differs')
    require(data['rows_per_layer']==dict(train=9600,dev=6400),'Different data population')
    for layer in data['layers']:
        for split in ('train','dev'):
            for item in layer[split]:
                require(read(item['sidecar']['path'])['policy_fingerprint']==base['checkpoint_hash'],
                        'Captured shard checkpoint differs')
    manifest=workspace/'input_manifest.json';publish(manifest,data)
    runtime=read(oldroot/'runtime_config.json')
    payload=dict(schema=SCHEMA,workspace=str(workspace),diagnostic_summary=artifact(summary_path),
        diagnostic_registration=summary['registration'],input_manifest=artifact(manifest),
        original_source_root=str(oldroot),original_output=str(oldout),method_source_root=runtime['method_source_root'],
        original_protocol=artifact(oldroot/'protocol.json'),original_bank=diagnostic['bank'],
        original_source_files=diagnostic['original_sources'],diagnostic_source_files=diagnostic['own_sources'],
        original_evidence_files=diagnostic['fixed_evidence'],
        source_files=[artifact(Path(__file__).resolve()),artifact(Path(__file__).with_name('training.py'))],
        checkpoint_hash=base['checkpoint_hash'],base_state_hash=base['base_state_hash'],
        policy_configuration_fingerprint=base['configuration_fingerprint'],settings=SETTINGS,architecture=ARCHITECTURE,
        gates=GATES,counts=dict(layers=32,train_prefixes=75,dev_prefixes=50,train_rows_per_layer=9600,dev_rows_per_layer=6400),
        initialization='fresh original seeded TopK for every layer; no old weights/optimizer reused',
        amendment={'only_training_data_coverage_changed':'16 to128 uniformly sampled original causal positions per prefix',
                   'epochs_and_optimizer_unchanged':True,'effective_minibatch_rows':128,
                   'minibatches_per_epoch':75,'old_minibatch_rows':16,'old_minibatches_per_epoch':75,
                   'independent_layer_scheduling':True,'maximum_parallel_children_per_gpu':8,
                   'selected_by':'minimum raw FP32 development MSE over all64 epochs',
                   'development_informed_pre_test_repair':True},
        downstream=dict(rebuild_all_graphs=125,reuse_old_graphs=0,fit_nn_after_graph_gate=True,
                        real_es_and_heldout_and_ablations_required=True),
        new_oracle_calls=0,new_generation_calls=0,main_experiment_complete=False,registered_at=time.time())
    reg=sealed(payload);publish(workspace/'registration.json',reg)
    return read_registry(workspace/'registration.json')


def read_registry(path):
    reg=read(path);verify_seal(reg)
    require(reg['schema']==SCHEMA and reg['settings']==SETTINGS and reg['architecture']==ARCHITECTURE
            and reg['gates']==GATES and reg['counts']==dict(layers=32,train_prefixes=75,dev_prefixes=50,
                 train_rows_per_layer=9600,dev_rows_per_layer=6400),'Training settings/scope/gates changed')
    require(reg['diagnostic_summary']['sha256']==SUMMARY_SHA and reg['diagnostic_registration']['sha256']==DIAGNOSTIC_SHA,
            'Unregistered diagnostic input')
    require(Path(path).resolve()==Path(reg['workspace'])/'registration.json','Registration moved to a different workspace')
    require(reg['source_files']==[artifact(Path(__file__).resolve()),artifact(Path(__file__).with_name('training.py'))],
            'New training source changed')
    check(reg['diagnostic_summary']);check(reg['diagnostic_registration']);check(reg['input_manifest'])
    for ref in reg['original_source_files']+reg['diagnostic_source_files']+reg['original_evidence_files']:check(ref)
    data=read(reg['input_manifest']['path'])
    _,diagnostic,expected=diagnostic_inputs(reg['diagnostic_summary']['path'],verify_shards=False,verify_sidecars=False)
    require(data==expected,'Corpus does not exactly match pinned diagnostic receipts')
    require(reg['original_source_files']==diagnostic['original_sources']
            and reg['diagnostic_source_files']==diagnostic['own_sources']
            and reg['original_evidence_files']==diagnostic['fixed_evidence']
            and reg['original_bank']==diagnostic['bank']
            and reg['original_source_root']==diagnostic['original']
            and reg['original_output']==diagnostic['original_output'],'Unbound source/parent evidence inventory')
    oldroot=Path(diagnostic['original']).resolve();workspace=Path(reg['workspace']).resolve()
    runtime=read(oldroot/'runtime_config.json');base=read(Path(reg['original_output'])/'base_policy.json')
    require(reg['method_source_root']==runtime['method_source_root']
            and reg['original_protocol']==artifact(oldroot/'protocol.json')
            and reg['checkpoint_hash']==base['checkpoint_hash'] and reg['base_state_hash']==base['base_state_hash']
            and reg['policy_configuration_fingerprint']==base['configuration_fingerprint'],
            'Policy/source/runtime binding changed')
    require(not workspace.is_relative_to(oldroot) and not workspace.is_relative_to(Path(diagnostic['output'])),
            'Training workspace overlaps immutable original data')
    require(reg['initialization']=='fresh original seeded TopK for every layer; no old weights/optimizer reused'
            and reg['new_oracle_calls']==reg['new_generation_calls']==0 and reg['main_experiment_complete'] is False
            and reg['downstream']==dict(rebuild_all_graphs=125,reuse_old_graphs=0,fit_nn_after_graph_gate=True,
                                       real_es_and_heldout_and_ablations_required=True),
            'Fresh-fit/downstream boundary changed')
    require([x['layer_path'] for x in data['layers']]==LAYER_PATHS and data['rows_per_layer']=={'train':9600,'dev':6400},
            'Layer corpus changed')
    require(all(len(x['train'])==75 and len(x['dev'])==50 for x in data['layers']),'Omitted train/development prefixes')
    return reg


def fit_registration(reg):
    return dict(schema=SCHEMA,training_registry_fingerprint=reg['fingerprint'],
                original_protocol=reg['original_protocol'],input_manifest=reg['input_manifest'],
                changed_only='captured causal128 rows and independent layer placement',test_used=False)
