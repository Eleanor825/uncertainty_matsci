"""Validate this complete public export without model, environment or large-artifact reads."""
from pathlib import Path
import hashlib, importlib.util, json, math
P = Path(__file__).resolve().parent
def module(name):
    spec = importlib.util.spec_from_file_location(name, P / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
m = json.loads((P / 'manifest.json').read_text())
for name, r in m['files'].items():
    b = (P / name).read_bytes()
    assert len(b) == r['bytes'] and hashlib.sha256(b).hexdigest() == r['sha256'], name
d = json.loads((P / 'snapshot.json').read_text())
assert d['all_ten_complete'] is True and d['publication_ready'] is True
assert m['publication_kind'] == 'complete_seeded_world4_diagnostic'
assert m['complete_results_released'] is True and m['new_scientific_calls'] == 0
assert all(v == 0 for v in d['scientific_calls'].values())
assert not d['old_unseeded_p335_pooling'] and not d['original_Full_followup_included']
assert len(d['arms']) == 10 and len(d['groups']) == 2
assert all(g['all_five_arms_closed'] and g['group_completion_validated'] and g['resource_closed'] and not g['failure_present'] for g in d['groups'])
arms = {(a['policy_seed'], a['condition']): a for a in d['arms']}
expected = {336: [(0,29), (0,29), (0,29), (.125,9), (.25,10)], 337: [(0,2), (0,2), (0,2), (0,21), (0,4)]}
for seed in (336,337):
    for name, target in zip(module('summarize').ARMS, expected[seed]):
        a = arms[(seed,name)]; q = a['metrics']
        assert a['status'] == 'complete' and a['world_seed'] == 4 and not a['audit_errors']
        assert (q['scoreNormalized'],q['failure_count']) == target and q['completedSuccessfully'] is False
        assert q['agent_attempts'] == len(a['action_rows']) == 30
        assert [r['attempt'] for r in a['action_rows']] == list(range(1,31))
        assert q['failure_count'] == sum(r['official_success'] is False for r in a['action_rows'])
        assert math.isclose(q['failure_fraction'],q['failure_count']/30,abs_tol=1e-15)
        assert math.isclose(q['F'],q['scoreNormalized']-.1*q['failure_fraction'],abs_tol=1e-15)
        assert a['RPC_closure'] == {'child_still_running':False,'closed_response':True,'returncode':0,'unknown_connection':False}
        assert a['seed_sidecar']['closed_phase_recorded'] and a['seed_sidecar']['recorded_states'] == 31
        assert [r['after_attempt'] for r in a['seed_sidecar']['states']] == list(range(31))
        if name in ('NoGraphRisk','ExplicitRepeatRisk'):
            native=arms[(seed,'Native1')]
            assert [r['executed_action_sha256'] for r in a['action_rows']] == [r['executed_action_sha256'] for r in native['action_rows']]
            assert q == native['metrics']
        assert a['candidate_costs']['candidate_assemble_calls'] == a['candidate_costs']['candidate_backward_calls'] == 0
        assert a['candidate_costs']['NoGraph_forward_counts']['intents'] == a['candidate_costs']['NoGraph_forward_counts']['returns']
assert len(d['paired_contrasts']) == 8 and all(r['complete_pair'] for r in d['paired_contrasts'])
for r in d['paired_contrasts']:
    for k,v in r['right_minus_left'].items():
        assert v == arms[(r['policy_seed'],r['right'])]['metrics'][k] - arms[(r['policy_seed'],r['left'])]['metrics'][k]
assert len(d['same_seed_prefix_audits']) == 8
aligned = 0
for p in d['same_seed_prefix_audits']:
    left,right = (arms[(p['policy_seed'],p[k])] for k in ('left','right'))
    length = 0
    for l,r in zip(left['action_rows'],right['action_rows']):
        if l['executed_action_sha256'] != r['executed_action_sha256']: break
        length += 1
    assert length == p['same_action_prefix_length']
    assert length == (19 if p['policy_seed']==336 else 3) if p['left']=='ExplicitRepeatCommon2' else length==30
    snapshots = p['aligned_same_action_prefix_snapshots']
    assert [s['after_attempt'] for s in snapshots] == list(range(length+1))
    assert not p['public_mismatch_after_same_prefix'] and not p['global_rng_mismatch_after_same_prefix']
    assert not p['complete_hidden_state_equivalence_claimed'] and not p['causal_benefit_established']
    aligned += len(snapshots)
    for s in snapshots:
        n=s['after_attempt']
        assert all(s[k] is True for k in ('executed_action_prefix_equal','global_rng_equal','object_construction_summary_equal','public_UI_available','public_UI_equal','seed_state_available'))
        ls,rs=left['seed_sidecar']['states'][n],right['seed_sidecar']['states'][n]
        assert all(ls[k]==rs[k] for k in ('current_global_random_state_sha256','objects_seeded','ordered_object_identity_sha256'))
        if s['next_proposal0_available']:
            assert all(s[k] is True for k in ('next_prompt_messages_equal','next_prompt_tokens_equal','next_sampling_seed_equal'))
            lp,rp=left['action_rows'][n]['proposal0'],right['action_rows'][n]['proposal0']
            assert all(lp[k]==rp[k] for k in ('prompt_messages_hash','prompt_token_hash','sampling_seed'))
        else: assert n==30
assert aligned == 210
renderer=module('render'); report,stats=renderer.render(d)
assert report == (P/'report.md').read_text()
assert json.loads(json.dumps(stats)) == json.loads((P/'complete_two_seed_statistics.json').read_text())
assert hashlib.sha256((P/'render.py').read_bytes()).hexdigest()==m['renderer_source_sha256']
summary=json.loads((P/'descriptive_statistics.json').read_text())
assert summary==module('summarize').summarize(d)
assert len(stats)==16 and len(summary['arms'])==20 and len(summary['paired_contrasts'])==24
for record in summary['arms']+summary['paired_contrasts']:
    x,y=(record['values_by_seed'][str(s)] for s in (336,337))
    assert record['n']==2
    assert math.isclose(record['mean'],(x+y)/2,abs_tol=1e-15)
    assert math.isclose(record['sample_variance'],(x-y)**2/2,abs_tol=1e-15)
    assert math.isclose(record['sample_SD'],math.sqrt(record['sample_variance']),abs_tol=1e-15)
assert sum(a['candidate_costs']['generated_proposals'] for a in d['arms']) == 476
assert sum(a['candidate_costs']['NoGraph_forward_counts']['returns'] for a in d['arms']) == 1127
print(json.dumps({'complete_results_ready':True,'closed_episodes':10,'closed_resource_groups':2,'complete_pairs':8,'aligned_same_prefix_snapshots':aligned,'renderer_statistics':16,'descriptive_statistics':44,'recorded_agent_actions':300,'recorded_generated_proposals':476,'new_scientific_calls':0},sort_keys=True))
