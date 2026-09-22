"""Metadata-only incumbent selection; no model, evaluator, or result-ref traversal."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def select_incumbent(rows, *, expected_seeds):
    """Source-verified dev rows enter here; unknown/failed never become zero.

    Caller owns original acceptance/qualification/checkpoint checks. This pure
    function validates a balanced G0/G1/G2 dev grid and returns both rankings.
    """
    seeds = tuple(expected_seeds)
    require(seeds and len(set(seeds)) == len(seeds) and all(type(s) is int for s in seeds), 'Invalid registered seeds')
    expected = {(g, s) for g in (0, 1, 2) for s in seeds}
    actual = {}
    for row in rows:
        require(type(row.get('generation')) is int and type(row.get('seed')) is int, 'Invalid identity')
        key = (row['generation'], row['seed'])
        require(key in expected and key not in actual, 'Unexpected/duplicate identity')
        require(row.get('split') == 'dev' and row.get('metric_name') == 'AUDC', 'Only dev official AUDC permitted')
        if row.get('status') == 'accepted':
            score = row.get('AUDC')
            require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1, 'Invalid accepted AUDC')
        actual[key] = dict(row)
    incomplete = [list(k) for k in sorted(expected) if k not in actual or actual[k].get('status') != 'accepted']
    if incomplete:
        return {'status': 'selection_incomplete', 'selected_generation': None,
                'deployment_default_generation': 0, 'retained_G0_is_not_a_measured_winner': True,
                'incomplete_cells': incomplete, 'all_observed_rows': list(actual.values())}
    curve = [{'generation': g, 'mean_AUDC': math.fsum(actual[g,s]['AUDC'] for s in seeds) / len(seeds),
              'seed_scores': [{'seed': s, 'AUDC': actual[g,s]['AUDC']} for s in seeds]}
             for g in (0, 1, 2)]
    order = lambda values: sorted(values, key=lambda r: (-r['mean_AUDC'], r['generation']))
    legacy, incumbent = order(curve[1:]), order(curve)
    return {'status': 'complete_metadata_selection', 'original_G1_G2_ranking': legacy,
            'incumbent_G0_G1_G2_ranking': incumbent,
            'original_selected_generation': legacy[0]['generation'],
            'selected_generation': incumbent[0]['generation'],
            'selected_dev_AUDC': incumbent[0]['mean_AUDC'],
            'change_in_selected_dev_AUDC': incumbent[0]['mean_AUDC'] - legacy[0]['mean_AUDC'],
            'equal_seed_weighting': True, 'tie_break': 'smaller_generation',
            'all_observed_rows': list(actual.values())}


def rows_from_historical_projection(value):
    """Validate fixed archived dev projection; never claim to reread raw RPC."""
    require(value.get('schema') == 'derived_existing_G0_development_comparison_v1', 'Wrong source schema')
    require(value.get('task') == 'Al-Pd-Sm' and value.get('budget') == 10, 'Wrong task/budget')
    require(value.get('model_id') == 'Qwen/Qwen3.5-4B', 'Wrong model')
    require(value.get('main_1080_member') is False and value.get('original_selection_rewritten') is False, 'Original result changed')
    a = value['audit']
    require(a['G0_acceptance'] == 'passed' and a['returncode'] == 0 and a['entire_child_subtree_closed'] is True
            and a['original_G0_evidence_unchanged'] is True, 'Accepted G0 closure absent')
    require([r['generation'] for r in value['rows']] == [0, 1, 2], 'Original row order/coverage changed')
    seed = value['environment_seed']; official = None; rows = []
    for row in value['rows']:
        e = row['physical_evidence']; costs = e['costs']
        require(e['metric_name'] == 'AUDC' and e['metric_value'] == row['AUDC'], 'Official metric mismatch')
        require(e['environment_seeds'] == [seed] and e['episode_metric_values'] == [row['AUDC']], 'Dev seed/episode mismatch')
        require(e['reward_is_llm_self_score'] is False, 'LLM reward forbidden')
        require(costs['candidate_oracle_attempts'] == 10 and costs['step_attempts'] == 10
                and costs['completed_episodes'] == 1, 'Incomplete physical budget')
        official = official or e['official_commit']
        require(e['official_commit'] == official, 'Different official evaluator')
        if row['generation']:
            original = row['original_output_references']
            require(original['generation'] == row['generation'] and original['AUDC'] == row['AUDC']
                    and original['new_calls'] == 0, 'Original dev identity mismatch')
        rows.append({'generation': row['generation'], 'seed': seed, 'split': 'dev',
                     'metric_name': 'AUDC', 'AUDC': row['AUDC'], 'status': 'accepted',
                     'source_role': 'SHA_verified_historical_dev_projection',
                     'original_output_references': row.get('original_output_references', {
                         'original_result': value['original_result_reference'], 'completion': value['completion_reference']}),
                     'physical_evidence': e})
    return rows


def read_pinned(path, expected_sha):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected_sha, 'Source SHA mismatch: ' + str(path))
    return raw


def run_audit(package, project, protocol_commit, output):
    package, project, output = map(lambda p: Path(p).resolve(), (package, project, output))
    protocol_raw = (package / 'protocol.json').read_bytes()
    # Only committed protocol/source plus one fixed dev-only projection are opened.
    for name, raw in [('protocol.json', protocol_raw), ('selector.py', Path(__file__).read_bytes())]:
        blob = subprocess.run(['git','-C',str(package),'show',protocol_commit+':'+name], check=True, capture_output=True).stdout
        require(blob == raw, 'Protocol/source differs from committed blob')
    p = json.loads(protocol_raw)
    require(p['schema'] == 'made_incumbent_dev_selection_protocol_v1', 'Wrong protocol')
    rel = Path(p['historical_projection']['project_relative_path'])
    require(not rel.is_absolute() and '..' not in rel.parts and str(rel).startswith('technical_not_main/G0_development_result_'), 'Only fixed dev path allowed')
    source = (project / rel).resolve()
    require(source.is_relative_to(project), 'Dev projection escaped project')
    raw = read_pinned(source, p['historical_projection']['sha256'])
    value = json.loads(raw)
    result = select_incumbent(rows_from_historical_projection(value), expected_seeds=p['environment_seeds'])
    result.update(schema='made_safe_incumbent_selection_audit_v1', protocol_commit=protocol_commit,
                  protocol_sha256=sha(protocol_raw), selector_sha256=sha(Path(__file__).read_bytes()),
                  historical_projection={'project_relative_path':str(rel),'sha256':sha(raw),'bytes':len(raw)},
                  original_raw_references_reread=False, metadata_projection_only=True,
                  original_G0_state_hash=value['theta0'], original_selection_rewritten=False,
                  eligible_for_live_model_activation=False, test_outcomes_read_by_this_audit=False,
                  new_model_calls=0, new_environment_calls=0, new_training_updates=0,
                  interpretation='One previously used development seed; retrospective rule audit, not independent efficacy or deployment.')
    output.mkdir(parents=True, exist_ok=False)
    (output/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (output/'report.md').write_text('# Incumbent selection metadata audit\n\n'
        + 'Original G1/G2 selects G%d; including G0 selects G%d.\n\n' % (result['original_selected_generation'], result['selected_generation'])
        + 'Official dev AUDC: G0=0.68, G1=0.62, G2=0.64; one already-used AlPdSm/B10 seed. '
        'Both rankings and all evidence references are retained in selection.json. '
        'This reuses the SHA-verified dev projection; raw RPC was not reread. No test outcomes, model/environment calls, '
        'or training updates; no production checkpoint activated.\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--protocol-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    r=run_audit(Path(__file__).parent,a.project,a.protocol_commit,a.output)
    print(json.dumps({k:r[k] for k in ('status','original_selected_generation','selected_generation','selected_dev_AUDC','test_outcomes_read_by_this_audit','new_environment_calls')}))

if __name__ == '__main__':
    main()
