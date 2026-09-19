"""Synthetic CPU unit contracts only; no SnAr oracle or pretrained policy use."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from matdiscovery.action_targets import ActionTargetError
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.native_attribution import NativeAttributor, PolicyStamp, TranscoderBinding
from matdiscovery.policy import ActionGeneration
from matdiscovery.transcoders import TopKTranscoder, TranscoderConfig, write_activation_shard

from .snar_policy import (SnArContractError, SnArPolicyAdapter, SnArPolicyConfig,
                         build_messages, parse_action_json, validate_action, replay_base_generation)
from .features import (CandidateBank, activation_rows, build_snar_action_targets,
                       validate_transfer_on_shards)


ACTION = {"tau": 1.0, "equiv_pldn": 2.0, "conc_dfnb": 0.3, "temperature": 60.0}


class CharTokenizer:
    all_special_ids = [1]
    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return ''.join('<|im_end|>' if t == 1 else chr(t-10) if t >= 10 else 'P' for t in tokens)


def generated(text=None, *, suffix="", stamp=None):
    text = json.dumps(ACTION, separators=(',', ':')) if text is None else text
    ids = torch.tensor([[3, 4]+[ord(c)+10 for c in text]+[1]+[ord(c)+10 for c in suffix]])
    stamp = stamp or PolicyStamp('unit-test-only', 'base-unit-hash', 'synthetic-unit-model', 0)
    return ActionGeneration(True, text, dict(ACTION), ids, 2, ids.shape[1]-2, (), (), stamp,
                            'unit-config', 1, 'original-generation-ids-only-hash', 'eos')


@pytest.mark.parametrize('mutation', [
    {**ACTION, 'equivalents': 2}, {k:v for k,v in ACTION.items() if k!='tau'},
    {**ACTION, 'tau': True}, {**ACTION, 'tau': .49}, {**ACTION, 'temperature': float('inf')},
])
def test_reject_wrong_names_bounds_booleans_and_nonfinite(mutation):
    with pytest.raises(SnArContractError): validate_action(mutation)


def test_strict_json_rejects_prose_duplicate_keys_and_multiple_actions():
    text = json.dumps(ACTION)
    assert parse_action_json(text) == ACTION
    for bad in ('explanation '+text, text+text, text[:-1]+', "tau":1.0}', text.replace('1.0', 'NaN', 1)):
        with pytest.raises(SnArContractError): parse_action_json(bad)


def test_history_whitelists_observed_xy_and_declares_window_without_token_truncation():
    history = [{'parameters': ACTION, 'objectives': {'sty': i+1., 'e_factor': 1000.},
                'future_label': 'must not appear'} for i in range(3)]
    messages, proof = build_messages({'query_count':3,'budget':10}, history,
        config=SnArPolicyConfig(history_window=2), objective_description='Unit-test objective only.')
    payload = json.loads(messages[-1]['content'])
    assert len(payload['history'])==2 and payload['history_omitted_count']==1
    assert 'future_label' not in messages[-1]['content']
    assert payload['history'][-1]['objectives']['e_factor']==1000  # not clipped to official domain500
    assert proof['context_budget']==2048 and proof['max_input_tokens']==1920 and proof['max_new_tokens']==128
    with pytest.raises(SnArContractError):
        build_messages({'query_count':2,'budget':10},history,config=SnArPolicyConfig(),objective_description='x')


def test_qwen_wrapper_generates_once_and_preserves_invalid_original_tokens():
    class Fake:
        max_input_tokens=1920;decoding=SnArPolicyConfig().decoding();tokenizer=CharTokenizer();calls=0
        def generate_action(self, messages, **kw):
            self.calls+=1
            assert set(kw['legal_schema']['properties'])==set(ACTION)
            return generated('reason '+json.dumps(ACTION))
    fake=Fake();adapter=SnArPolicyAdapter(fake,objective_description='Synthetic objective')
    result,prompt=adapter.generate_action({'query_count':0,'budget':10},[],seed=1)
    assert fake.calls==1 and not result.success and result.failure_code=='snar_strict_action_json'
    assert torch.equal(result.input_ids_with_completion,generated('reason '+json.dumps(ACTION)).input_ids_with_completion)


def test_total_context_cap_includes_completion_and_rejects_larger_silent_budget():
    with pytest.raises(SnArContractError,match='Complete prefix'):
        SnArPolicyConfig(max_input_tokens=2048)
    with pytest.raises(SnArContractError,match='Complete prefix'):
        SnArPolicyConfig(max_new_tokens=129)


def test_prior_is_separate_from_online_budget_and_at_most_four_rows():
    prior=[{'parameters':ACTION,'objectives':{'sty':1.,'e_factor':9.}}]*4
    msgs,proof=build_messages({'query_count':0,'budget':50},[],config=SnArPolicyConfig(),
        objective_description='Registered synthetic objective',prior_history=prior)
    data=json.loads(msgs[-1]['content'])
    assert data['query_count']==0 and data['remaining_budget']==50 and data['history']==[]
    assert len(data['shared_prior'])==4 and proof['shared_prior_rows']==4
    with pytest.raises(SnArContractError,match='at most4'):
        build_messages({'query_count':0,'budget':50},[],config=SnArPolicyConfig(),
            objective_description='x',prior_history=prior+[prior[0]])


def test_replay_rebinds_only_verified_base_stamp_preserving_tokens_without_oracle():
    g=generated();g.raw_text=CharTokenizer().decode(g.input_ids_with_completion[0,2:].tolist(),skip_special_tokens=False,clean_up_tokenization_spaces=False)
    g.prefix_hash=tensor_state_hash({'input_ids':g.input_ids_with_completion});g.policy_runtime={'unit_test_only':True}
    record=g.to_record();original=json.dumps(record,sort_keys=True)
    p=SimpleNamespace(model_stamp=replace(g.model_stamp,state_id='new-verified-base-session'),
                      configuration_fingerprint=g.configuration_fingerprint,
                      runtime_precision_record=lambda:g.policy_runtime,tokenizer=CharTokenizer())
    copied,proof=replay_base_generation(record,p)
    assert copied.model_stamp.state_id!=g.model_stamp.state_id
    assert torch.equal(copied.input_ids_with_completion,g.input_ids_with_completion)
    assert proof['scientific_oracle_calls']==0 and json.dumps(record,sort_keys=True)==original
    p.model_stamp=replace(p.model_stamp,generation=2)
    with pytest.raises(SnArContractError,match='initial-base'):replay_base_generation(record,p)
    p.model_stamp=replace(g.model_stamp,state_id='new')
    corrupted=dict(record,input_ids_with_completion=[[2,3]])
    with pytest.raises(SnArContractError):replay_base_generation(corrupted,p)


def test_numeric_targets_off_by_one_special_exclusion_source_hash_and_raw_scores():
    g=generated();t=build_snar_action_targets(g,CharTokenizer())
    assert t.benchmark=='summit_snar'
    assert ''.join(t.token_texts)=='1.02.00.360.0'
    assert set(t.semantic_paths)=={'/'+k for k in ACTION}
    assert t.prediction_positions==tuple(p-1 for p in t.token_positions) and 1 not in t.token_ids
    assert t.source_prefix_hash==tensor_state_hash({'input_ids':g.input_ids_with_completion,'attention_mask':torch.ones_like(g.input_ids_with_completion)})
    raw=torch.linspace(-2,2,256).repeat(1,len(t.token_ids),1)
    expected=torch.stack([raw[0,i].log_softmax(-1)[token] for i,token in enumerate(t.token_ids)]).mean()
    torch.testing.assert_close(t.mean_logprob(raw),expected)
    assert not torch.isclose(t.mean_logprob(raw),t.mean_logprob(raw/.6))
    raw[:,:,0]=-float('inf')
    with pytest.raises(ActionTargetError):t.mean_logprob(raw)
    with pytest.raises(ActionTargetError):t.validate(g.input_ids_with_completion[:,:-1],torch.ones_like(g.input_ids_with_completion[:,:-1]))
    with pytest.raises(ActionTargetError):build_snar_action_targets(replace(g,success=False),CharTokenizer())


def test_mixed_field_name_number_token_is_rejected():
    class Whole:
        all_special_ids=[1]
        def decode(self,tokens,**kwargs):
            return ''.join({2:json.dumps(ACTION),1:'<eos>',3:'P'}[t] for t in tokens)
    g=replace(generated(),input_ids_with_completion=torch.tensor([[3,2,1]]),prompt_token_count=1)
    with pytest.raises(ActionTargetError,match='inseparably'):build_snar_action_targets(g,Whole())


class Tiny(torch.nn.Module):
    """Two-dimensional causal CPU unit model; no scientific fidelity claim."""
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(_attn_implementation='synthetic-unit-scan')
        self.embedding=torch.nn.Embedding(256,2);self.mlp=torch.nn.Linear(2,2,bias=False)
        self.head=torch.nn.Linear(2,256,bias=False)
        with torch.no_grad():
            n=torch.arange(256).float();self.embedding.weight.copy_(torch.stack([.2+n/1000,.3+(n%7)/100],-1))
            self.mlp.weight.copy_(torch.eye(2)*.5)
            # Unit model has a resolvable direct sink derivative at the unchanged
            # FP32 epsilon=.001 / rtol=.05 / atol=.0001 production gate.
            self.head.weight.copy_(torch.randn(256,2,generator=torch.Generator().manual_seed(12)))
        self.eval()
    def get_input_embeddings(self):return self.embedding
    def forward(self,input_ids,attention_mask=None,use_cache=False,logits_to_keep=0):
        assert not use_cache
        x=self.embedding(input_ids);x=x+.03*x.cumsum(1);x=x+self.mlp(x)
        selection=slice(-logits_to_keep,None) if isinstance(logits_to_keep,int) else logits_to_keep
        return SimpleNamespace(logits=self.head(x[:,selection]))


def exact_tc():
    tc=TopKTranscoder(TranscoderConfig(2,2,2,2))
    with torch.no_grad():
        tc.encoder.weight.copy_(torch.eye(2)*.5);tc.encoder.bias.zero_()
        tc.decoder.weight.copy_(torch.eye(2));tc.decoder.bias.zero_()
    return tc


def test_actual_native_full_prefix_parity_future_gradient_and_fd():
    model=Tiny();stamp=PolicyStamp('unit-native',tensor_state_hash(dict(model.state_dict())),'synthetic-unit-only',0)
    tc=exact_tc();binding=TranscoderBinding('mlp',tc,{'fidelity_gate_passed':True,'max_dev_fvu':.5,'provenance':{'layer_path':'mlp'}})
    native=NativeAttributor(model,[binding],state_id_getter=lambda:stamp.state_id,
        architecture_review={'verified':True,'source':'synthetic CPU unit contract only'},
        max_nodes=512,max_feature_nodes=4,max_backward_targets=6,node_influence_mass=1,edge_row_mass=1)
    g=generated(stamp=stamp);t=build_snar_action_targets(g,CharTokenizer())
    trace=native.capture(g.input_ids_with_completion,stamp,action_targets=t)
    assert trace.parity_max_abs==0
    for index,position in enumerate(t.token_positions):
        torch.testing.assert_close(trace.reference_logits[0,index],model(g.input_ids_with_completion[:,:position],logits_to_keep=1).logits[0,-1])
    grads=torch.autograd.grad(t.mean_logprob(trace.logits),list(trace.embedding_leaves.values())+list(trace.mlp_leaves.values()),retain_graph=True)
    for grad in grads:assert torch.count_nonzero(grad[:,t.causal_horizon+1:])==0
    later=generated(suffix='   ',stamp=stamp);lt=build_snar_action_targets(later,CharTokenizer())
    assert lt.target_spec_hash==t.target_spec_hash and lt.source_prefix_hash!=t.source_prefix_hash
    later_logits=model(later.input_ids_with_completion,logits_to_keep=torch.tensor(lt.prediction_positions)).logits
    torch.testing.assert_close(lt.mean_logprob(later_logits),t.mean_logprob(trace.reference_logits))
    from .features import validate_snar_backend,attribute_features
    from matdiscovery.representation_training import GraphStageConfig
    config=GraphStageConfig(max_nodes=512,max_feature_nodes=4,max_backward_targets=6,dtype='float32')
    adapter=SimpleNamespace(tokenizer=CharTokenizer(),prepare_trace_inputs=lambda ids,stamp:(ids,{'use_cache':False}))
    with pytest.raises(ValueError,match='exact policy state'):
        attribute_features(native,adapter,g,config=config)
    report=validate_snar_backend(native,adapter,g,config=config)
    assert report['passed'] and len(report['checks'])>=2
    features,meta,_=attribute_features(native,adapter,g,config=config)
    assert features['graph.score_node_count']==1 and meta['target']['benchmark']=='summit_snar'
    assert not any('logprob' in k for k in features)
    stale=replace(g,model_stamp=replace(stamp,state_id='different-session'))
    with pytest.raises(ValueError,match='exact policy state'):
        attribute_features(native,adapter,stale,config=config)


def test_real_reconstruction_transfer_gate_pass_failure_unknown_and_test_rejection(tmp_path):
    tc=exact_tc();binding=TranscoderBinding('mlp',tc,{'original_pretraining_only':True})
    candidate=CandidateBank((binding,),{'unit_test_only':True},'unit-bank','unit-policy')
    def shard(split,y):
        x=torch.tensor([[.2,.4],[.4,.6],[.8,.9],[1.,1.1]])
        p=tmp_path/(split+'.pt')
        write_activation_shard(p,x,y(x),group_ids=[split]*4,prefix_hashes=[split+'-prefix']*4,
            split=split,policy_fingerprint='unit-policy',layer_path='mlp')
        return p
    train=shard('train',lambda x:x*.5);dev=shard('dev',lambda x:x*.5)
    proof=validate_transfer_on_shards(candidate,{'mlp':[train]},{'mlp':[dev]})
    assert proof['passed'] and proof['layers'][0]['dev']['output_fvu']==pytest.approx(0)
    assert not proof['test_data_used'] and not proof['transcoder_training_performed']
    shard('dev',lambda x:x*.5+10)
    assert not validate_transfer_on_shards(candidate,{'mlp':[train]},{'mlp':[dev]})['passed']
    shard('dev',lambda x:torch.ones_like(x))
    assert not validate_transfer_on_shards(candidate,{'mlp':[train]},{'mlp':[dev]})['passed']
    with pytest.raises(ValueError,match='split'):
        validate_transfer_on_shards(candidate,{'mlp':[train]},{'mlp':[train]})
    with pytest.raises(ValueError,match='0.5'):
        validate_transfer_on_shards(candidate,{'mlp':[train]},{'mlp':[dev]},max_dev_fvu=2)


def test_activation_rows_are_actual_uniform_rows_not_hidden_state_graph_proxy():
    capture=SimpleNamespace(input_ids=torch.ones(1,20,dtype=torch.long),prefix_hash='actual-unit-prefix',
        mlp_inputs={'mlp':torch.arange(40).reshape(1,20,2)},mlp_outputs={'mlp':torch.arange(40).reshape(1,20,2)*2})
    result=activation_rows(capture,token_rows=4)['mlp']
    assert result['token_positions']==[0,6,12,19]
    torch.testing.assert_close(result['outputs'],capture.mlp_outputs['mlp'][0,[0,6,12,19]].float())


def test_original_cpu_normalized64_fresh_bank_all32_and_no_retraining_resume(tmp_path, monkeypatch):
    """Real tiny64-epoch fits, never Qwen/chemical data or physical results."""
    import hashlib
    from matdiscovery.representation_training import QWEN_MLP_PATHS, validate_transcoder_bank
    from .features import train_snar_bank,load_candidate_bank
    configuration={'policy_runtime':{'synthetic_cpu_test':True},'test_configuration':True}
    config_fp=hashlib.sha256(json.dumps(configuration,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    policy=SimpleNamespace(model_stamp=PolicyStamp('unit-base','unit-checkpoint','unit-model',0),
        checkpoint_hash='unit-checkpoint',model_id='unit-model',mlp_paths=QWEN_MLP_PATHS,
        configuration_fingerprint=config_fp,runtime_precision_record=lambda:configuration['policy_runtime'],
        _configuration=lambda:configuration)
    tc=TopKTranscoder(TranscoderConfig(2,2,8,8))
    template=CandidateBank(tuple(TranscoderBinding(p,tc,{}) for p in QWEN_MLP_PATHS),{'unit_test_only':True},'unit-bank','unit-checkpoint')
    x=torch.tensor([[.2,.4],[.4,.6],[.8,.9],[1.,1.1]])
    train,dev={},{}
    for i,path in enumerate(QWEN_MLP_PATHS):
        for split,store in [('train',train),('dev',dev)]:
            p=tmp_path/f'{split}_{i}.pt';store[path]=[p]
            write_activation_shard(p,x,x*.5,group_ids=[split]*4,prefix_hashes=[split+'-prefix']*4,
                split=split,policy_fingerprint='unit-checkpoint',layer_path=path)
    output=tmp_path/'independent_bank'
    manifest=train_snar_bank(policy,train,dev,output_directory=output,registration={'unit_test_only':True},
                            template=template,learning_rate=.04,batch_size=4)
    bank=validate_transcoder_bank(manifest,model_key='qwen35_4b',checkpoint_hash='unit-checkpoint')
    assert bank['complete'] and bank['passed_layers']==32 and bank['snar_training_performed']
    assert bank['scientific_oracle_calls']==0
    assert all(len(r['metadata']['history'])==64 for r in bank['layers'])
    loaded=load_candidate_bank(manifest,policy)
    assert loaded.source['source_is_snar_fit'] and loaded.transfer_report is None
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in output.iterdir()}
    import matdiscovery.normalized_transcoders as original
    monkeypatch.setattr(original,'train_normalized_layer',lambda *a,**k:pytest.fail('Completed bank must not retrain'))
    assert train_snar_bank(policy,train,dev,output_directory=output,registration={'unit_test_only':True},
                          template=template,learning_rate=.04,batch_size=4)==manifest
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in output.iterdir()}
    partial=tmp_path/'partial';partial.mkdir();(partial/'training_claim.json').write_text('{}')
    with pytest.raises(ValueError,match='Partial'):
        train_snar_bank(policy,train,dev,output_directory=partial,registration={'unit_test_only':True},
                        template=template,learning_rate=.04,batch_size=4)
