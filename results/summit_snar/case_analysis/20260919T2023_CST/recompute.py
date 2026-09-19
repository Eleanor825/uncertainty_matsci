"""Local, stdlib-only SnAr case diagnostics. No model/oracle/remote interfaces."""
from pathlib import Path
import csv, hashlib, json, math, statistics

D = Path(__file__).resolve().parent
I = D / 'inputs'
TOL = 1e-12

def read(name):
    return json.loads((I / name).read_text())

def write(name, value):
    (D / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def table(name, rows):
    with (D / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def stat(x):
    return {'n': len(x), 'mean': statistics.mean(x), 'sample_variance': statistics.variance(x),
            'sample_SD': statistics.stdev(x)}

for item in json.loads((D / 'input_manifest.json').read_text())['files']:
    assert hashlib.sha256((I / item['input']).read_bytes()).hexdigest() == item['sha256']
source = {x['summary']['name']: x['summary'] for x in read('episode_summaries.json')}
coverage = {x['name']: x['valid_proposal_graph_fraction'] for x in read('heldout_graph_coverage.json')}
assert len(source) == 35
assert {(s['arm'], s['seed']) for s in source.values()} == {
    (arm, seed) for arm in read('protocol.json')['evaluation']['arms'] for seed in range(5101, 5106)}
assert all(s['queries'] == 50 and len(s['hv_curve']) == 50 and s['prior_rows'] == 550 for s in source.values())
prior = read('prior_and_selection.json')['prior_HV']
assert all(s['prior_hv'] == prior for s in source.values())

rows, events, fronts = [], [], []
reference = {(x['objectives']['sty'], x['objectives']['e_factor']) for x in source['test_qwen_base_5101']['raw_pareto_front']}
for name, s in sorted(source.items()):
    prev = prior; positive = []; stagnant = max_stagnant = 0
    for q, hv in enumerate(s['hv_curve'], 1):
        delta = hv - prev
        assert delta >= -1e-14
        if delta > TOL:
            positive.append(q); stagnant = 0
            events.append({'arm': s['arm'], 'seed': s['seed'], 'query_1based': q, 'HV_increment': delta,
                           'HV_increment_micro': delta * 1e6, 'gain_from_prior_after': hv - prior})
        else:
            stagnant += 1; max_stagnant = max(max_stagnant, stagnant)
        prev = hv
    assert abs(s['final_hv_gain'] - (s['hv_curve'][-1] - prior)) < 1e-15
    assert abs(s['mean_querywise_hv_gain'] - statistics.mean(h - prior for h in s['hv_curve'])) < 1e-15
    llm = s['arm'] not in ('random', 'gp_ei_scalarized')
    guided = s['arm'] in ('uq_esopt', 'uq_only')
    second = s['proposal_count'] - 45 if llm else None
    if llm: assert 0 <= second <= 45
    graph_fraction = coverage[name]
    missing = None
    if guided:
        valid = s['proposal_count'] - s['invalid_proposals']
        missing_float = valid * (1 - graph_fraction)
        missing = round(missing_float)
        assert abs(missing_float - missing) < 1e-9
    nonlhs_hits = sum(q > 5 for q in positive)
    early = 45 - second if guided else None
    rows.append({'name': name, 'arm': s['arm'], 'seed': s['seed'], 'final_HV_gain': s['final_hv_gain'],
        'mean_querywise_HV_gain': s['mean_querywise_hv_gain'], 'gain_B10_prefix': s['hv_curve'][9] - prior,
        'gain_B30_prefix': s['hv_curve'][29] - prior, 'gain_B50': s['hv_curve'][49] - prior,
        'positive_HVI_queries': len(positive), 'initial_LHS_positive_HVI_queries': sum(q <= 5 for q in positive),
        'first_improvement_query': positive[0] if positive else None,
        'last_improvement_query': positive[-1] if positive else None,
        'last_improvement_to_end_queries': 50 - positive[-1] if positive else 50,
        'longest_no_HVI_run_queries': max_stagnant, 'proposal_attempts': s['proposal_count'],
        'second_proposal_queries': second, 'invalid_proposals': s['invalid_proposals'],
        'risk_observations': s['risk_observations'], 'Brier': s['brier'],
        'valid_proposal_graph_fraction': graph_fraction, 'missing_graph_valid_proposals': missing,
        'high_risk_first_proposal_trigger_lower_bound': max(0, second - s['invalid_proposals'] - missing) if guided else None,
        'high_risk_first_proposal_trigger_upper_bound': second if guided else None,
        'accepted_first_proposal_with_risk_below_0_5': early,
        'low_risk_no_HVI_executions_lower_bound': max(0, early - nonlhs_hits) if guided else None,
        'raw_final_front_points': len(s['raw_pareto_front']),
        'prior_share_of_absolute_final_HV': prior / s['final_hv']})
    for j, p in enumerate(s['raw_pareto_front']):
        o = p['objectives']; x = p['parameters']
        fronts.append({'arm': s['arm'], 'seed': s['seed'], 'front_index': j,
            'matches_zero_gain_base5101_reference_objective': (o['sty'], o['e_factor']) in reference,
            **x, **o})
by = {(r['arm'], r['seed']): r for r in rows}
table('per_seed_mechanisms.csv', rows); table('positive_HVI_events.csv', events); table('terminal_pareto_points.csv', fronts)

pairs = []
for arm in ('uq_esopt', 'uq_only'):
    for comp in ('qwen_base', 'gp_ei_scalarized', 'random_controller'):
        for seed in range(5101, 5106):
            a, b = by[arm, seed], by[comp, seed]
            delta = a['final_HV_gain'] - b['final_HV_gain']
            pairs.append({'arm': arm, 'comparator': comp, 'seed': seed,
                'final_gain_delta': delta, 'final_gain_delta_micro': delta * 1e6,
                'querywise_gain_delta': a['mean_querywise_HV_gain'] - b['mean_querywise_HV_gain'],
                'sign': 'positive' if delta > TOL else 'negative' if delta < -TOL else 'tie',
                'first_improvement_arm': a['first_improvement_query'], 'first_improvement_comparator': b['first_improvement_query']})
table('paired_case_differences.csv', pairs)

full = read('full_development_episodes.json')
es = read('es_only_development.json')
selection_full = read('prior_and_selection.json')['full_checkpoint_selection']
selection_es = es['selection']['value']
dev_rows = []
for branch, episodes, selected in [('full', full, selection_full), ('es_only', es['episodes'], selection_es)]:
    group = {}
    for e in episodes:
        s = e['summary']['value'] if 'value' in e['summary'] else e['summary']
        g = int(e['name'].split('_G')[1].split('_')[0])
        score = s['mean_querywise_hv'] - .1 * s['invalid_proposal_rate'] - (.1 * s['brier'] if branch == 'full' else 0)
        group.setdefault(g, []).append(s)
        expected = selected['development'][g]['scores'][[4101, 4102].index(s['seed'])]
        assert abs(score - expected) < 1e-14
    means = {}
    for g, ep in sorted(group.items()):
        assert len(ep) == 2
        means[g] = {'mean_HV': statistics.mean(s['mean_querywise_hv'] for s in ep),
            'mean_Brier': statistics.mean(s['brier'] for s in ep) if branch == 'full' else None,
            'mean_invalid_rate': statistics.mean(s['invalid_proposal_rate'] for s in ep)}
        m = means[g]; baseline = means[0]
        dhv = m['mean_HV'] - baseline['mean_HV']
        dbrier = -.1 * (m['mean_Brier'] - baseline['mean_Brier']) if branch == 'full' else 0
        dinvalid = -.1 * (m['mean_invalid_rate'] - baseline['mean_invalid_rate'])
        fitness = selected['development'][g]['fitness']
        delta = fitness - selected['development'][0]['fitness']
        assert abs(dhv + dbrier + dinvalid - delta) < 1e-14
        dev_rows.append({'branch': branch, 'generation': g, **m, 'fitness': fitness,
            'fitness_delta_from_G0': delta, 'HV_contribution_to_delta': dhv,
            'Brier_penalty_contribution_to_delta': dbrier, 'invalid_penalty_contribution_to_delta': dinvalid,
            'selected': selected['selected']['generation'] == g,
            'weight_hash': selected['development'][g]['weight_hash']})
table('development_selection_decomposition.csv', dev_rows)

uq = [by['uq_esopt', s] for s in range(5101, 5106)]
risk = read('risk_development.json')
n = sum(r['risk_observations'] for r in uq)
hits = sum(r['positive_HVI_queries'] for r in uq)
assert all(r['initial_LHS_positive_HVI_queries'] == 0 for r in rows)
assert n == 224 and hits == 6 and by['uq_esopt', 5105]['risk_observations'] == 44
assert by['uq_esopt', 5105]['positive_HVI_queries'] == 0
assert all(by['uq_esopt', s]['risk_observations'] == 45 for s in range(5101, 5105))
# Therefore all six HVI-positive actions are scored and the sole missing score is no-HVI.
scored_no_hvi = n - hits
summary = {'scope': {'episodes': 35, 'seeds': 5, 'reaction_functions': 1, 'shared_prior_rows': 550,
    'shared_prior_HV': prior, 'adaptation_calls': 550, 'heldout_calls': 1750, 'this_analysis_new_scientific_calls': 0},
    'arm_gains': {}, 'comparisons': {}, 'full_UQ': {
        'non_LHS_queries': 225, 'proposals': sum(r['proposal_attempts'] for r in uq),
        'second_proposal_queries': sum(r['second_proposal_queries'] for r in uq),
        'extra_proposal_query_fraction': sum(r['second_proposal_queries'] for r in uq) / 225,
        'invalid_proposals': sum(r['invalid_proposals'] for r in uq),
        'missing_valid_proposal_graphs': sum(r['missing_graph_valid_proposals'] for r in uq),
        'high_risk_first_trigger_lower_bound': sum(r['high_risk_first_proposal_trigger_lower_bound'] for r in uq),
        'high_risk_first_trigger_upper_bound': sum(r['high_risk_first_proposal_trigger_upper_bound'] for r in uq),
        'early_first_low_risk_executions': sum(r['accepted_first_proposal_with_risk_below_0_5'] for r in uq),
        'low_risk_no_HVI_executions_lower_bound': sum(r['low_risk_no_HVI_executions_lower_bound'] for r in uq),
        'positive_HVI_queries': hits, 'scored_queries': n, 'scored_no_HVI_queries': scored_no_hvi,
        'weighted_test_Brier': sum(r['Brier'] * r['risk_observations'] for r in uq) / n,
        'constant_dev_prevalence_0_9_Brier_descriptive': (scored_no_hvi * .01 + hits * .81) / n,
        'constant_one_Brier_descriptive': hits / n,
        'development_metrics': risk['report']['development_metrics'],
        'development_retry_probability': risk['report']['random_controller_retry_probability']},
    'identity_checks': {}}
for arm in read('protocol.json')['evaluation']['arms']:
    ar = [by[arm, seed] for seed in range(5101, 5106)]
    summary['arm_gains'][arm] = {'final': stat([r['final_HV_gain'] for r in ar]),
        'querywise': stat([r['mean_querywise_HV_gain'] for r in ar]),
        'HVI_positive_queries_total': sum(r['positive_HVI_queries'] for r in ar)}
for comp in ('qwen_base', 'gp_ei_scalarized', 'random_controller'):
    pp = [r for r in pairs if r['arm'] == 'uq_esopt' and r['comparator'] == comp]
    summary['comparisons']['full_minus_' + comp] = {'final_delta': stat([r['final_gain_delta'] for r in pp]),
        'querywise_delta': stat([r['querywise_gain_delta'] for r in pp]),
        'positive': sum(r['sign'] == 'positive' for r in pp), 'negative': sum(r['sign'] == 'negative' for r in pp),
        'ties': sum(r['sign'] == 'tie' for r in pp)}
for a, b in [('uq_esopt', 'uq_only'), ('es_only', 'qwen_base')]:
    summary['identity_checks'][a + '_vs_' + b] = {
        'all_5_HV_curves_equal': all(source[f'test_{a}_{s}']['hv_curve'] == source[f'test_{b}_{s}']['hv_curve'] for s in range(5101, 5106)),
        'all_5_weight_hashes_equal': all(source[f'test_{a}_{s}']['weight_hash'] == source[f'test_{b}_{s}']['weight_hash'] for s in range(5101, 5106)),
        'raw_proposal_sequence_equality_verified': False}
write('case_summary.json', summary)
write('recomputation_receipt.json', {'passed': True, 'source_rows': 35, 'curve_points': 1750,
    'case_comparisons': len(pairs), 'development_generations': len(dev_rows),
    'scope': 'case statistics only; existing full scientific acceptance not rerun', 'scientific_calls': 0})
print(json.dumps({'passed': True, 'rows': len(rows), 'positive_HVI_events': len(events),
                  'scored_Brier': summary['full_UQ']['weighted_test_Brier'],
                  'low_risk_no_HVI_lower_bound': summary['full_UQ']['low_risk_no_HVI_executions_lower_bound']}))
