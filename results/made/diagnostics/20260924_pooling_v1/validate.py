"""Validate the fixed public aggregate; no network, fitting or private source files."""
from pathlib import Path
import hashlib, json, math

P = Path(__file__).resolve().parent
def read(name):
    return json.loads((P / name).read_text())
def equal(a, b):
    return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)

m = read('manifest.json')
assert set(m['files']) == {p.name for p in P.iterdir() if p.is_file() and p.name != 'manifest.json'}
for name, pin in m['files'].items():
    value = (P / name).read_bytes()
    assert len(value) == pin['bytes'] and hashlib.sha256(value).hexdigest() == pin['sha256'], name
v = read('MADE_pooling_diagnostic_results_20260924.json')
s = v['scope']
assert v['benchmark'] == 'MADE' and v['status'] == 'complete_accepted_fixed_offline_development_diagnostic'
assert s['source_rows'] == s['supported_rows'] == 350
assert s['train_rows'] == 250 and s['dev_rows'] == 100
assert s['actual_shards'] == 2 and s['rows_per_shard'] == [175, 175]
assert s['train_chemistries'] == 3 and s['dev_chemistries'] == 2
assert s['train_episodes'] == 5 and s['dev_episodes'] == 2
assert s['public_missing_prompt_rows'] == s['unsupported_rows'] == s['new_online_trajectories'] == 0
assert s['same_support_for_all_six_arms'] is True
names = ('A', 'B', 'C', 'P', 'P+B-hidden', 'P+C-hidden')
assert set(v['models']) == set(names)
for name in names:
    model = v['models'][name]
    assert model['effective_temperature'] == 1 and model['selected_regularization_C'] in (.001, .01, .1, 1.)
    dev = model['dev']
    assert dev['rows'] == 100 and dev['row_ids_hash'] == s['dev_row_ids_hash']
    assert set(dev['by_group']) == {'Al-Pd-Sm', 'Dy-In-Pd'}
    assert all(g['rows'] == 50 for g in dev['by_group'].values())
    assert dev['by_group']['Al-Pd-Sm']['failures'] == 25
    assert dev['by_group']['Dy-In-Pd']['failures'] == 32
    for metric in ('Brier', 'NLL'):
        assert equal(dev['macro_' + metric], sum(g[metric] for g in dev['by_group'].values()) / 2)
        assert equal(dev['micro_' + metric], dev['macro_' + metric])
    assert not dev['row_level_significance_tested']
    assert 0 <= dev['pooled_AUROC'] <= 1
for contrast, value in v['comparisons'].items():
    left, right = contrast.split('_minus_')
    a, b = (v['models'][k]['dev'] for k in (left, right))
    assert value['prediction_row_ids_hash'] == s['dev_row_ids_hash']
    assert not value['statistical_significance_claimed']
    for metric in ('Brier', 'NLL'):
        assert equal(value['dev_macro_' + metric + '_difference'], a['macro_' + metric] - b['macro_' + metric])
        for group in a['by_group']:
            assert equal(value['by_group'][group][metric], a['by_group'][group][metric] - b['by_group'][group][metric])
cost = v['costs']
assert cost['extraction_forward_intents'] == cost['extraction_forward_returns'] == 700
assert cost['extraction_environment_calls'] == 0
cpu = cost['CPU_fit']
assert cpu['CV_fit_intents'] == cpu['CV_fit_returns'] == 72
assert cpu['final_fit_intents'] == cpu['final_fit_returns'] == 6
assert all(cpu[k] == 0 for k in ('CV_fit_failures', 'LLM_forwards', 'environment_calls', 'graph_calls', 'temperature_fit_intents', 'temperature_fit_returns'))
assert not v['conclusions']['planner_improvement_demonstrated']
assert not v['conclusions']['unseen_task_generalization_demonstrated']
assert not v['summary_generation']['source_metric_numbers_recomputed_from_raw_predictions']
for ref in v['sources'].values():
    assert set(ref) <= {'sha256', 'bytes', 'commit'} and len(ref['sha256']) == 64
md = (P / 'MADE_pooling_diagnostic_results_20260924.md').read_text()
for name in names:
    model = v['models'][name]; dev = model['dev']
    display = model['display_name']
    assert f"| {display} | {dev['pooled_AUROC']:.6f} | {dev['macro_Brier']:.6f} | {dev['macro_NLL']:.6f} | {model['selected_regularization_C']:g} |" in md
assert m['new_scientific_calls'] == v['public_export']['new_scientific_calls'] == 0
print(json.dumps({'status': 'passed', 'rows': 350, 'train_rows': 250, 'development_rows': 100,
                  'models': 6, 'development_chemistries': 2, 'contrasts': len(v['comparisons']),
                  'extraction_forwards': 700, 'CV_fits': 72, 'final_fits': 6,
                  'raw_prediction_recomputation': False, 'new_scientific_calls': 0}, sort_keys=True))
