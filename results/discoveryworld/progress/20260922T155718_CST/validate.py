"""Validate this scientific projection; no model, simulator, fit or network calls."""
from pathlib import Path
import hashlib
import json
import math
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate(data):
    assert data['schema'] == 'discoveryworld_public_progress_v3'
    assert data['public_export_fingerprint'] == digest({k: v for k, v in data.items() if k != 'public_export_fingerprint'})
    assert data['scientific_calls_for_export'] == 0 and data['effectiveness_demonstrated'] is False
    graph = data['graph']
    assert graph['rows'] == graph['expected_rows'] == 160
    assert graph['indices'] == [1 + i * 99 // 19 for i in range(20)]
    assert graph['qualification_semantics'] == 'declared_V4_local_Jacobian_mathematical_reference'
    assert graph['native_FP32_flags_preserved'] is True and graph['all_native_FP32_checks_passed_claimed'] is False
    assert len(graph['groups']) == 4
    episodes = []
    for group in graph['groups']:
        assert group['status'] == 'subset_graph_group_complete'
        assert group['qualification_prefixes_returned'] == group['qualification_prefixes_intended'] == 8
        assert group['qualification_prefixes_failed'] == 0
        assert group['new_environment_calls'] == group['new_policy_generations'] == 0
        assert len(group['shards']) == 2
        episodes.extend(Path(r['path']).parent.name for r in group['shards'])
    assert len(set(episodes)) == 8
    assert sum('-collection_train-' in x for x in episodes) == 6
    assert sum('-collection_nn_dev-' in x for x in episodes) == 2
    risk = data['neural_risk']
    assert risk['admitted_for_frozen_controller'] is True and risk['test_used'] is False
    assert risk['fit_dev_or_test_data_read'] is False and risk['total_optimizer_updates'] == 400
    head = risk['heads']['next_action_failure']
    assert head['admitted'] is True and all(head['gates'].values())
    assert head['support']['positive_rows'] == 10 and head['support']['negative_rows'] == 30
    assert head['support']['episodes'] == 2
    for metrics in head['metrics'].values():
        assert (metrics['rows'], metrics['episodes'], metrics['worlds']) == (40, 2, 1)
        assert 0 <= metrics['auroc'] <= 1 and 0 <= metrics['brier'] <= 1 and metrics['nll'] >= 0
        assert all(math.isfinite(metrics[k]) for k in ('auroc', 'brier', 'nll'))
    internal = head['metrics']['Internal']
    control = head['metrics']['OutputAction']
    assert math.isclose(internal['auroc'], .97) and math.isclose(control['auroc'], .96)
    assert internal['brier'] < control['brier'] and internal['nll'] < control['nll']
    for name in ('generation_invalid', 'terminal_noncompletion'):
        assert risk['heads'][name]['admitted'] is False and risk['models'][name]['available'] is False
    models = risk['models']['next_action_failure']['models']
    assert set(models) == {'Internal', 'OutputAction'}
    assert all(m['updates'] == 200 and m['tensor_changed'] is True for m in models.values())
    g0 = data['Full_G0_development']
    assert g0['accepted'] is True and g0['generation'] == 0
    assert (g0['job']['world_seed'], g0['job']['policy_seed'], g0['job']['max_agent_attempts']) == (2, 303, 10)
    assert g0['job']['stage'] == 'es_dev' and g0['job']['project_split'] == 'dev'
    metrics = g0['metrics']
    assert (metrics['agent_attempts'], metrics['failure_count'], metrics['scoreNormalized']) == (10, 7, 0)
    assert metrics['completedSuccessfully'] is False
    assert math.isclose(metrics['failure_fraction'], .7) and math.isclose(metrics['F'], -.07) and metrics['J'] == metrics['F']
    rows = g0['attempts']
    assert [r['attempt_index'] for r in rows] == list(range(1, 11))
    assert all(r['candidate_count'] == 1 and r['supported'] is True for r in rows)
    assert all(not r['neural_revision'] and not r['neural_rank_change'] for r in rows)
    assert g0['controller_counts'] == {'supported_neural_revisions': 0, 'supported_rank_changes': 0}
    threshold = models['Internal']['threshold']
    assert threshold['source_partition'] == 'world1_calibration_only' and threshold['n'] == 60
    assert threshold['quantile'] == .75 and threshold['rank_one_based'] == 45
    assert g0['frozen_threshold'] == threshold['threshold']
    assert max(r['risk'] for r in rows) == g0['max_risk'] < g0['frozen_threshold']
    assert data['Full_ES']['status'] == 'stopped_at_G0_controller_activity_gate'
    assert data['Full_ES']['accepted_G0_development_episodes'] == 1
    assert data['Full_ES']['accepted_mutated_policy_episodes'] == data['Full_ES']['policy_parameter_perturbations'] == 0
    assert data['Full_ES']['failure']['message'] == 'Registered Full G0 controller-activity gate failed'
    baseline = data['held_out_Baseline']
    assert baseline['accepted'] is True
    assert (baseline['job']['world_seed'], baseline['job']['policy_seed'], baseline['job']['max_agent_attempts']) == (3, 401, 30)
    assert baseline['metrics']['scoreNormalized'] == 0 and baseline['metrics']['failure_count'] == 29
    assert data['held_out_Full_accepted'] == data['completed_matched_pairs'] == 0
    assert data['G0_vs_Baseline_comparison_valid'] is False
    assert data['MADE_context'] == {'planned': 1080, 'valid': 1079, 'technical_failures': 1, 'pending': 0,
                                    'unchanged_export': 'results/made_components/incremental/20260922T101850_CST/report.md'}
    def fields(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in {'hostname', 'pid', 'gpu_uuid', 'password', 'credentials', 'prompt', 'raw_prompt', 'traceback'}
                if key in ('path', 'repo_path'):
                    assert not Path(item).is_absolute() and '..' not in Path(item).parts
                if key.endswith('sha256'):
                    assert re.fullmatch('[0-9a-f]{64}', item)
                fields(item)
        elif isinstance(value, list):
            for item in value:
                fields(item)
        elif isinstance(value, str):
            assert not re.search(r'/Users/|/mnt/|/private/|GPU-|tj-3041039|de-41039|-----BEGIN .*PRIVATE KEY', value)
    fields(data)


def report(data):
    validate(data)
    risk = data['neural_risk'];head = risk['heads']['next_action_failure'];g0 = data['Full_G0_development']
    lines = ['# DiscoveryWorld: V4 graphs complete; Full search stopped at G0', '',
        f"Observed **{data['observed_CST']}**. Model: **Qwen3.5-4B**; scenario: **Proteomics Normal**.", '',
        '**All 160 selected graphs and the supported next-action-failure risk models are complete. The Full branch stopped after its G0 development trajectory, before the first ES parameter perturbation. Full held-out tests and completed Baseline–Full pairs remain zero.**', '',
        '| Stage | Result | Scientific scope |', '|---|---|---|',
        '| Selected graphs | 160/160; 4 groups; 8 source episodes | Fixed 20 attempts per original preparatory episode |',
        '| V4 qualification | Each group returned 8/8 fixed prefixes, 0 qualification failures | Declared local-Jacobian mathematical reference |',
        '| Primary risk head | Internal and OutputAction models fitted and development-admitted | 40 development rows, 2 episodes, 1 world |',
        '| Two auxiliary risk heads | Not fitted/admitted | Fixed support gates failed |',
        '| Full G0 development | 1 accepted B10 trajectory | No controller revision or rank change |',
        '| Full ES parameter perturbations | 0 | Stopped at the registered G0 activity gate |',
        '| Held-out results | Baseline 1; Full 0; complete pairs 0 | No Full-effect estimate |', '',
        '## V4 graph qualification', '',
        'The original eight preparatory episodes supplied 160 graphs at the fixed 20 attempt indices. Six episodes supply fitting/calibration data (worlds 0 and 1), and two supply development data (world 2). Graph extraction generated no new actions and made no new environment calls.', '',
        'Each of four fresh graph processes passed the same eight-prefix suite under the declared V4 local-Jacobian mathematical-reference procedure. These are repeated checks of eight fixed prefixes, not 32 independent qualification examples. For eligible final-layer feature-to-action-score edges, V4 uses an independently checked FP64 mathematical tail reference. Original native FP32 finite-amplitude FD results and failure flags remain recorded. This completion does not claim that every original native FP32 FD check passed, or establish causality for environment failures.', '',
        'The full graph computation retains the original 32-feature / 42-backward-target limits. The DW bank remains the separately trained 32-layer bank that passed development fidelity; the earlier MADE-to-DW transfer failure and prior numerical failures remain in historical snapshots.', '',
        '## Risk-model development results', '',
        'The next-action-failure head has 10 positive and 30 negative development rows across two episodes in world 2. Internal features and output/action features were evaluated on the same rows. The frozen admission record passed its numerical and support gates.', '',
        '| Model/control | AUROC | Brier score | NLL |', '|---|---:|---:|---:|']
    for name in ('Internal', 'OutputAction', 'calibration', 'gradient', 'half'):
        m = head['metrics'][name];label = {'calibration': 'Calibration-partition prior', 'gradient': 'Fitting-partition prior', 'half': 'Constant 0.5'}.get(name, name)
        lines.append(f"| {label} | {m['auroc']:.6f} | {m['brier']:.6f} | {m['nll']:.6f} |")
    lines += ['',
        'AUROC is higher-is-better; Brier score and NLL are lower-is-better. These are development/admission statistics, not held-out task-performance estimates. Forty rows within two episodes are not 40 independent trials. No significance claim or demonstrated online improvement follows from the small Internal–OutputAction difference.', '',
        'Both next-action-failure models have changed NN parameters after 200 optimizer updates each (400 total). Fitting used world 0; temperature and threshold calibration used world 1. The fit seal records no world-2 development or held-out test read during fitting. The generation-invalid and terminal-noncompletion heads were not fitted because their fitting/calibration labels lacked both classes. Their unavailable outputs are not reported as trained predictions.', '',
        '## G0 activity gate and stopped Full search', '',
        '| World seed | Policy seed | Budget | Normalized score | Failed attempts | Fitness F = J |', '|---:|---:|---:|---:|---:|---:|',
        '| 2 | 303 | 10 | 0.0 | 7/10 | -0.070000 |', '',
        f"The accepted G0 development episode did not solve the task. Its largest supported risk was **{g0['max_risk']:.9f}**, below the frozen threshold **{g0['frozen_threshold']:.9f}** (the 45th of 60 world-1 calibration risks, quantile 0.75). All ten attempts retained one candidate; supported neural revisions and rank changes were both zero.", '',
        'The registered controller-activity gate failed and stopped the V4 branch before the first ES parameter perturbation. The risk NNs were trained, but the language model had not undergone an ES search update. The accepted G0 record and gate failure are both retained. There is no completed parameter-updating Full search or Full held-out trajectory in this snapshot.', '',
        'The previously accepted Baseline test used world 3, policy seed 401 and B30, with normalized score 0 and 29/30 failed attempts. It cannot be paired with G0: the world, policy seed, budget and development/test role differ. Baseline and G0 scores therefore do not provide a Full-versus-Baseline effect estimate.', '',
        '## History and validation', '',
        '[The 10:44 snapshot](../20260922T104452_CST/report.md) preserves the earlier single-prefix diagnosis and zero-completion state at that time. [The 05:24 snapshot](../20260922T052401_CST/report.md) preserves the preparatory Native episodes, earlier ES-only branch and failed bank transfer. Later snapshots overlap these records; counts must not be added together.', '',
        'MADE remains unchanged at 1079 valid evaluations, one retained technical failure and zero pending. Its Full-minus-Native mean SUN and AUDC differences remain negative at B10, B30 and B50.', '',
        '`snapshot.json` contains only explicitly selected scientific fields, relative artifact references and source hashes. Run `python3 -B validate.py` to verify the projection seal, counts, identities, G0 risk/threshold arithmetic, report rendering, privacy field constraints and bundle hashes. Development AUROC/Brier/NLL values are original reported metrics; this export does not reconstruct them from raw predictions or repeat original simulator acceptance. No model, graph extraction, simulator, NN fitting or ES work was run to create the export.', '']
    return '\n'.join(lines)


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / 'snapshot.json').read_text())
    assert (root / 'report.md').read_text() == report(data)
    for ref in json.loads((root / 'manifest.json').read_text())['files']:
        raw = (root / ref['path']).read_bytes()
        assert len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256']
    print(json.dumps({'passed': True, 'graph_rows': 160, 'graph_groups': 4, 'qualification_prefixes_per_group': 8,
                      'development_rows': 40, 'development_episodes': 2, 'development_worlds': 1,
                      'G0_accepted': 1, 'policy_parameter_perturbations': 0, 'Full_held_out_tests': 0,
                      'completed_matched_pairs': 0, 'scientific_calls_for_export': 0}))


if __name__ == '__main__':
    main()
