"""Validate this scientific-field projection and report without scientific calls."""
from pathlib import Path
import hashlib
import json
import math
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate(data):
    assert data['schema'] == 'discoveryworld_public_progress_v2'
    assert data['public_export_fingerprint'] == digest({k: v for k, v in data.items() if k != 'public_export_fingerprint'})
    assert data['scientific_calls_for_export'] == 0
    assert data['effectiveness_demonstrated'] is False
    counts = data['accepted_stage_completions']
    assert counts == {'held_out_Baseline': 1, 'held_out_Full': 0, 'completed_Baseline_Full_pairs': 0,
                      'risk_NN_fit_and_calibration': 0, 'Full_ES_episodes': 0, 'formally_qualified_graph_rows': 0}
    row = data['held_out_Baseline']
    assert row['accepted'] is True
    assert row['job'] == {'arm': 'Baseline', 'stage': 'test', 'model_key': 'qwen35_4b', 'scenario': 'Proteomics',
                          'difficulty': 'Normal', 'world_seed': 3, 'policy_seed': 401,
                          'max_agent_attempts': 30, 'max_proposals_per_attempt': 1}
    metrics = row['metrics']
    assert metrics['agent_attempts'] == 30 and metrics['failure_count'] == 29
    assert metrics['scoreNormalized'] == 0 and metrics['completedSuccessfully'] is False
    assert math.isclose(metrics['failure_fraction'], 29 / 30, abs_tol=1e-12)
    assert math.isclose(metrics['F'], -.1 * 29 / 30, abs_tol=1e-12)
    assert metrics['J'] == metrics['F']
    bank = data['DW_domain_bank']
    assert bank['status'] == 'complete' and bank['passed_layers'] == bank['total_layers'] == 32
    assert bank['max_dev_fvu'] == .5 and bank['test_used'] is False
    assert bank['original8prefix_qualification_still_required'] is True
    fd = data['finite_difference_diagnostic']
    assert fd['qualified'] is False and fd['original_qualification_remains_failed'] is True
    assert fd['prefix_count'] == 1 and fd['required_formal_prefix_count'] == 8
    assert fd['rtol'] == .05 and fd['atol'] == .0001
    assert fd['test_used'] is False and fd['training_updates'] == fd['new_policy_generations'] == fd['environment_calls'] == 0
    assert fd['total_policy_forwards'] == 36 and fd['capture_policy_forwards'] == 2 and fd['intervention_policy_forwards'] == 34
    assert len(fd['checks']) == 16
    assert {(r['epsilon'], r['ordinal']) for r in fd['checks']} == {(e, n) for e in (.03, .01, .003, .001) for n in range(4)}
    totals = {}
    for row in fd['checks']:
        expected = abs(row['finite_difference'] - row['analytical'])
        tolerance = fd['atol'] + fd['rtol'] * abs(row['analytical'])
        assert math.isclose(row['absolute_error'], expected, abs_tol=1e-12)
        assert math.isclose(row['original_tolerance'], tolerance, abs_tol=1e-12)
        assert row['passes_original_numeric_tolerance'] == (expected <= tolerance)
        totals[row['epsilon']] = totals.get(row['epsilon'], 0) + row['passes_original_numeric_tolerance']
    assert totals == {.03: 4, .01: 4, .003: 4, .001: 3}
    assert data['historical_MADE_bank_transfer']['layer0_fvu'] > .5
    assert data['historical_preparation']['observed_CST'].startswith('2026-09-22T05:24:01')
    text = json.dumps(data, allow_nan=False)
    assert not re.search(r'/Users/|/mnt/|/private/|/tmp/|GPU-|(?i:hostname|gpu_uuid|password|passwd|bearer|private.key)|"pid"|"prompt"', text)
    def refs(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('path', 'repo_path'):
                    p = Path(item)
                    assert not p.is_absolute() and '..' not in p.parts
                if key.endswith('sha256'):
                    assert re.fullmatch('[0-9a-f]{64}', item)
                refs(item)
        elif isinstance(value, list):
            for item in value:
                refs(item)
    refs(data)


def report(data):
    validate(data)
    metrics = data['held_out_Baseline']['metrics']
    fd = data['finite_difference_diagnostic']
    lines = ['# DiscoveryWorld: one Baseline result; Full comparison incomplete', '',
        f"Observed **{data['observed_CST']}**. Model: **Qwen3.5-4B**; scenario: **Proteomics Normal**.", '',
        '**The fixed Baseline trajectory is complete. Full has no accepted held-out result, so there is no completed pair or demonstrated method improvement.**', '',
        '| Stage | Accepted completion | Interpretation |', '|---|---:|---|',
        '| Held-out Baseline | 1/1 trajectory | World seed 3, policy seed 401, B30 |',
        '| Held-out Full | 0/1 trajectory | No result available |',
        '| Baseline–Full matched pairs | 0/1 pair | No paired effect can be estimated |',
        '| DW-domain transcoder bank | 32/32 layers pass development fidelity | Does not establish full attribution qualification |',
        '| Formally qualified graph rows | 0/160 | Required graph dataset unavailable |',
        '| Fitted and calibrated risk NN | 0 | No accepted fit/calibration completion |',
        '| Full parameter-updating ES | 0/7 episodes | No accepted Full ES trajectory |', '',
        '## Fixed held-out Baseline', '',
        '| Attempts | Normalized score | Task success | Failed attempts | Failure fraction | Fitness F = J |',
        '|---:|---:|---|---:|---:|---:|',
        f"| 30 | {metrics['scoreNormalized']:.1f} | False | 29/30 | {metrics['failure_fraction']:.6f} | {metrics['F']:.6f} |", '',
        'The original completion record is accepted, but the environment task was not solved. Failed action attempts are an episode metric, not 29 separate evaluation runs or a technical failure of the exported result. This single Baseline trajectory is insufficient to establish a Full-method effect.', '',
        '## Bank fidelity and finite-difference diagnosis', '',
        'The new DW-domain bank was trained on world 0 preparatory sources and checked for development fidelity on world 1. Its completion receipt records 32/32 layers passing the unchanged development FVU limit of 0.5. The test world was not used. The original eight-prefix attribution qualification remains required.', '',
        'The original attribution check at epsilon 0.001 failed on the first fixed TRAIN prefix. A separate diagnostic then measured the same four selected edges at four fixed step sizes. All diagnostic outcomes are retained below; this does not turn the diagnostic into a formal qualification.', '',
        '| Finite-difference epsilon | Passed checks | Scope |', '|---:|---:|---|']
    for epsilon in fd['epsilon_order']:
        rows = [r for r in fd['checks'] if r['epsilon'] == epsilon]
        lines.append(f"| {epsilon} | {sum(r['passes_original_numeric_tolerance'] for r in rows)}/4 | Same single TRAIN prefix |")
    lines += ['',
        'At epsilon 0.01, all four numerical checks pass the original rtol=0.05 and atol=0.0001. At epsilon 0.001, the token-to-action-score edge still fails (analytic 0.01641509; finite difference 0.01943111). The diagnostic reports `qualified=false` and preserves the original failed qualification. Agreement at larger steps is consistent with a small-step numerical issue; its exact cause has not been established.', '',
        'The diagnostic used 36 policy forwards (2 capture and 34 intervention/control forwards), zero new policy generations, zero environment calls and zero training updates. It used no held-out test information. Formal adoption of a changed step requires a separately registered prospective protocol, all eight fixed prefixes and qualification for each policy state used by the Full method. No later V3 execution is included in this snapshot.', '',
        '## Retained earlier results', '',
        'The MADE-trained bank previously failed cross-domain transfer at layer 0: FVU 0.843559980392456 exceeded 0.5. That negative result remains valid and is not replaced by the newly trained DW bank.', '',
        'The [05:24 historical report](../20260922T052401_CST/report.md) records eight completed preparatory Native B100 trajectories (800 attempts) and six of seven completed episodes in the earlier, separate ES-only B100 branch. Those counts retain their original observation time here. They are training/development evidence, not additional held-out Baseline or Full tests.', '',
        'The planned Full method still requires graph features, a fitted/calibrated risk NN, bounded proposal revisions and full-parameter Agentic ESOpt. The seven planned B10 ES episodes comprise four candidate trajectories and three development trajectories including G0. The fixed held-out Baseline and Full tests use the same world seed 3, policy seed 401 and B30. No effectiveness claim is made before this chain completes.', '',
        '## Export validation', '',
        '`snapshot.json` is an explicit scientific-field projection with source hashes and project-relative artifact references. Operational records and raw prompts are excluded. Original result acceptance and bank receipts are read as recorded; raw simulator acceptance and raw bank tensors were not rerun for this export. Source observations were collected sequentially and retain their own timestamps.', '',
        'Run `python3 -B validate.py` to check the whitelist, identities, metric arithmetic, all 16 diagnostic checks, report rendering and bundle hashes. Export generation makes zero model, simulator, graph, NN or ES calls.', '']
    return '\n'.join(lines)


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / 'snapshot.json').read_text())
    assert (root / 'report.md').read_text() == report(data)
    for ref in json.loads((root / 'manifest.json').read_text())['files']:
        raw = (root / ref['path']).read_bytes()
        assert len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256']
    print(json.dumps({'passed': True, 'Baseline_tests': 1, 'Full_tests': 0, 'paired_comparisons': 0,
                      'bank_fidelity_layers': 32, 'FD_diagnostic_checks': 16,
                      'formal_FD_qualification': False, 'risk_NN_completions': 0, 'Full_ES_episodes': 0,
                      'scientific_calls_for_export': 0}))


if __name__ == '__main__':
    main()
