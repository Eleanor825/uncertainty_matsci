"""Check published curves, scope, paired statistics and file hashes; stdlib only."""
from collections import defaultdict, Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
METHODS = ('baseline', 'esopt_graph_risk')

def read_csv(name):
    with (ROOT / name).open(newline='') as handle:
        return list(csv.DictReader(handle))

def near(actual, expected):
    if expected is None:
        return actual in (None, '')
    return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)

def desc(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'sample_variance': statistics.variance(values) if len(values) > 1 else None,
            'SD': statistics.stdev(values) if len(values) > 1 else None}

def main():
    summary = json.loads((ROOT / 'summary.json').read_text())
    records = read_csv('trajectories.csv')
    matrix = read_csv('all_system_results.csv')
    assert len(records) == len({r['job_id'] for r in records}) == 154
    assert len(matrix) == len({(r['budget'], r['task_id'], r['method'], r['seed']) for r in matrix}) == 180
    assert Counter(r['status'] for r in matrix) == {'complete': 154, 'active': 10, 'pending': 16}
    for row in matrix:
        if row['status'] != 'complete':
            assert all(row[k] == '' for k in ('SUN', 'AUDC', 'mSUN', 'raw_result_sha256'))
    curves = defaultdict(list)
    for row in read_csv('curves.csv'):
        curves[row['job_id']].append((int(row['candidate_attempt']), int(row['SUN'])))
    assert set(curves) == {r['job_id'] for r in records}
    assert sum(map(len, curves.values())) == 4254
    by = {}
    for row in records:
        b = int(row['budget']); points = curves[row['job_id']]
        assert row['benchmark'] == 'made' and row['model_key'] == 'qwen35_4b'
        assert row['seed'] == '1' and row['complete'] == 'True' and row['method'] in METHODS
        assert [x for x,y in points] == list(range(b+1)) and points[0] == (0,0)
        assert all(0 <= y <= x for x,y in points)
        audc = sum((x1-x0)*(y1+y0) for (x0,y0),(x1,y1) in zip(points,points[1:])) / b**2
        assert int(row['SUN']) == points[-1][1] and int(row['candidate_oracle_attempts']) == b
        assert near(row['AUDC'], audc) and near(row['mSUN'], points[-1][1]/b)
        for key in ('raw_result_sha256','raw_RPC_sha256','scope_registration_sha256'):
            assert len(row[key]) == 64 and all(c in '0123456789abcdef' for c in row[key])
        key=(b,row['task_id'],row['method']);assert key not in by;by[key]=row
    expected_stats = {}
    expected_pairs = {}
    for b in (10,30,50):
        tasks=sorted({t for bb,t,m in by if bb==b})
        paired=[t for t in tasks if all((b,t,m) in by for m in METHODS)]
        stated=summary['budgets'][str(b)]
        assert len(paired)==stated['complete_pairs'] and paired==stated['paired_system_ids']
        for metric in ('SUN','mSUN','AUDC'):
            left=[float(by[b,t,'baseline'][metric]) for t in paired]
            right=[float(by[b,t,'esopt_graph_risk'][metric]) for t in paired]
            delta=[y-x for x,y in zip(left,right)]
            for cohort in ('all_completed','matched_systems'):
                for method in METHODS:
                    values=([float(r[metric]) for (bb,t,m),r in by.items() if bb==b and m==method]
                            if cohort=='all_completed' else left if method=='baseline' else right)
                    expected_stats[b,cohort,method,metric]=desc(values)
            expected_stats[b,'paired_full_minus_baseline','paired_delta',metric]=desc(delta)
            expected=stated[metric]
            assert near(expected['baseline_mean'],statistics.mean(left)) and near(expected['full_mean'],statistics.mean(right))
            assert near(expected['full_minus_baseline'],statistics.mean(delta))
            assert (expected['wins'],expected['ties'],expected['losses']) == (sum(d>1e-12 for d in delta),sum(abs(d)<=1e-12 for d in delta),sum(d< -1e-12 for d in delta))
        for t in paired:expected_pairs[b,t]=True
    stats=read_csv('statistics.csv');assert len(stats)==len(expected_stats)
    for row in stats:
        key=int(row['budget']),row['cohort'],row['method'],row['metric']
        assert row['variance_ddof']=='1'
        for k,v in expected_stats.pop(key).items():assert near(row[k],v),(key,k,row[k],v)
    assert not expected_stats
    pair_rows=read_csv('paired_systems.csv');assert len(pair_rows)==len(expected_pairs)==64
    for row in pair_rows:
        b,t=int(row['budget']),row['task_id'];expected_pairs.pop((b,t))
        for method,prefix in [('baseline','baseline'),('esopt_graph_risk','full')]:
            source=by[b,t,method]
            for metric in ('SUN','mSUN','AUDC'):assert near(row[prefix+'_'+metric],float(source[metric]))
            assert row[prefix+'_raw_result_sha256']==source['raw_result_sha256']
    assert not expected_pairs
    candidate_calls=sum(int(r['candidate_oracle_attempts']) for r in records)
    assert candidate_calls==4100
    manifest_path=ROOT/'manifest.json'
    verified=0
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for item in manifest['files']:
            p=ROOT/item['path'];assert p.parent==ROOT and p.is_file() and not p.is_symlink()
            data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==item['export_sha256'] and len(data)==item['bytes']
            verified+=1
    print(json.dumps({'passed':True,'complete_trajectories':len(records),'registered_matrix_rows':len(matrix),
                     'curve_points':sum(map(len,curves.values())),'complete_pairs_by_budget':{'10':30,'30':30,'50':4},
                     'completed_trajectory_candidate_attempts':candidate_calls,'export_file_hashes_verified':verified,
                     'new_scientific_oracle_calls_by_validation':0,'new_model_calls_by_validation':0}))

if __name__=='__main__':main()
