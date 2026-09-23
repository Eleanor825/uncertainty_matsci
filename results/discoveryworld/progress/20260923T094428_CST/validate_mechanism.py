from pathlib import Path
import json
p=Path(__file__).resolve().parent
d=json.loads((p/'mechanism_audit.json').read_bytes());s=json.loads((p/'snapshot.json').read_bytes())
assert len(d['pairs'])==4 and all(r['matching_executed_action_prompt_tokens_sampling_seeds_and_logprobs_prefix_length']==13 and r['first_public_and_prompt_difference_attempt']==14 and not r['same_input_seed_runtime_but_different_completion_attempts']and not r['causal_controller_effect_identified']for r in d['pairs'])
c={r['condition']:r for r in d['conditions']};assert c['NoGraphRisk']['summary']['nonfirst_selections']==c['ExplicitRepeatRisk']['summary']['nonfirst_selections']==0
assert c['ExplicitRepeatCommon2']['executed_packet_change_attempts']==[17,19,24,25]and c['ExplicitRepeatInternal2']['executed_packet_change_attempts']==[21,23,25,28]
assert [r['task_score']for r in s['p335']['rows']]==[0,0,.125,0,.25]and [r['task_score_difference']for r in s['p335']['contrasts']]==[.125,.125,.25]
assert not d['python_hash_bug_established']and not d['controller_causal_benefit_established']and all(x==0 for x in d['new_scientific_calls'].values())
print(json.dumps({'mechanism_qualification_passed':True,'pairs_with_public_divergence_at14':4,'conditional_second_selections':0,'all_original_scores_preserved':True,'new_scientific_calls':0},sort_keys=True))
