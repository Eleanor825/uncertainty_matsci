"""Validate derived public result files; stdlib only, no experiment/network calls."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
METHODS = {'baseline', 'esopt_graph_risk'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def near(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def main():
    manifest = json.loads((RESULTS / 'manifest.json').read_text())
    require(manifest['schema'] == 'derived_public_results_manifest_v1', 'Unknown manifest')
    names = [entry['path'] for entry in manifest['files']]
    require(len(names) == len(set(names)), 'Duplicate exported file')
    for entry in manifest['files']:
        path = ROOT / entry['path']
        require(path.is_file() and not path.is_symlink() and ROOT in path.resolve().parents, 'Invalid export path')
        data = path.read_bytes()
        require(len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['export_sha256'], 'Export SHA changed: ' + entry['path'])
    index = json.loads((RESULTS / 'index.json').read_text())
    historical = {}
    trajectories_seen = 0
    curve_points = 0
    for item in index['snapshots']:
        folder = ROOT / item['summary']; summary = json.loads(folder.read_text()); folder = folder.parent
        require(summary['snapshot_id'] == item['snapshot_id'] and summary['budgets'] == item['budgets'], 'Index differs from snapshot')
        trajectories = rows(folder / 'trajectories.csv')
        require(len({r['job_id'] for r in trajectories}) == len(trajectories), 'Duplicate trajectory')
        curves = defaultdict(list)
        for point in rows(folder / 'curves.csv'):
            curves[point['job_id']].append((int(point['candidate_attempt']), int(point['SUN'])))
            curve_points += 1
        require(set(curves) == {r['job_id'] for r in trajectories}, 'Curve/trajectory inventory differs')
        by_budget = defaultdict(dict)
        for row in trajectories:
            budget = int(row['budget']); points = curves[row['job_id']]
            require(row['benchmark'] == 'made' and row['model_key'] == 'qwen35_4b' and row['method'] in METHODS
                    and row['seed'] == '1' and row['complete'] == 'True', 'Wrong scope/identity')
            require([x for x, y in points] == list(range(budget + 1)) and points[0] == (0, 0), 'Incomplete attempt curve')
            require(all(0 <= y <= x for x, y in points), 'Invalid SUN count')
            sun = points[-1][1]
            area = sum((x1-x0)*(y1+y0)/2 for (x0,y0),(x1,y1) in zip(points, points[1:]))
            require(int(row['candidate_oracle_attempts']) == budget and int(row['SUN']) == sun
                    and near(row['mSUN'], sun/budget) and near(row['AUDC'], 2*area/budget**2), 'Curve/metric/cost mismatch')
            signature = (row['task_id'], row['method'], budget, row['seed'], row['raw_result_sha256'], tuple(points))
            require(row['job_id'] not in historical or historical[row['job_id']] == signature, 'Historical result was changed')
            historical[row['job_id']] = signature
            key = (row['task_id'], row['method'])
            require(key not in by_budget[budget], 'Duplicate system/method')
            by_budget[budget][key] = row
            trajectories_seen += 1
        pair_rows = rows(folder / 'paired_systems.csv')
        require(len({(r['budget'],r['task_id'],r['seed']) for r in pair_rows}) == len(pair_rows), 'Duplicate pair')
        for budget_text, stated in summary['budgets'].items():
            budget = int(budget_text); members = by_budget[budget]
            systems = {key[0] for key in members}
            paired = {t for t in systems if all((t, m) in members for m in METHODS)}
            pairs = [r for r in pair_rows if int(r['budget']) == budget]
            require(stated['completed_trajectories'] == len(members) and stated['complete_pairs'] == len(paired)
                    and {r['task_id'] for r in pairs} == paired, 'Wrong completed denominator/pair inventory')
            require(stated['data_computation_complete'] == (len(members) == stated['expected_trajectories']), 'Wrong completion claim')
            require(sum(int(r['candidate_oracle_attempts']) for r in members.values()) == stated['completed_candidate_attempts'], 'Wrong completed cost sum')
            for metric in ('SUN', 'AUDC', 'mSUN'):
                left = [float(members[(t, 'baseline')][metric]) for t in sorted(paired)]
                right = [float(members[(t, 'esopt_graph_risk')][metric]) for t in sorted(paired)]
                expected = stated[metric]
                if not left:
                    require(expected['baseline_mean'] is None and expected['full_mean'] is None, 'Missing data became zero')
                    continue
                require(near(expected['baseline_mean'], sum(left)/len(left)) and near(expected['full_mean'], sum(right)/len(right))
                        and near(expected['full_minus_baseline'], sum(y-x for x,y in zip(left,right))/len(left)), 'Aggregate mismatch')
                require(expected['wins'] == sum(y>x+1e-12 for x,y in zip(left,right))
                        and expected['losses'] == sum(y<x-1e-12 for x,y in zip(left,right))
                        and expected['ties'] == sum(abs(y-x)<=1e-12 for x,y in zip(left,right)), 'Win/loss/tie mismatch')
            for row in pairs:
                left = members[(row['task_id'], 'baseline')]; right = members[(row['task_id'], 'esopt_graph_risk')]
                for metric in ('SUN','AUDC','mSUN'):
                    require(near(row['baseline_'+metric], left[metric]) and near(row['full_'+metric], right[metric]), 'Pair metric differs')
                require(row['baseline_raw_result_sha256'] == left['raw_result_sha256'] and row['full_raw_result_sha256'] == right['raw_result_sha256'], 'Pair hash differs')
    print(json.dumps({'passed': True, 'snapshots': len(index['snapshots']), 'trajectory_rows_across_overlapping_snapshots': trajectories_seen,
                      'distinct_physical_trajectories': len(historical), 'curve_points': curve_points,
                      'exported_files_hash_checked': len(manifest['files']), 'new_scientific_oracle_calls': 0, 'new_model_calls': 0}))


if __name__ == '__main__':
    main()
