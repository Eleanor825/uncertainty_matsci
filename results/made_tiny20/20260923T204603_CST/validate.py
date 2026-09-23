"""Validate this public projection; no experiment-server or model access."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
evidence = json.loads((ROOT / 'evidence.json').read_text())
manifest = json.loads((ROOT / 'manifest.json').read_text())
for item in manifest['files']:
    data = (ROOT / item['path']).read_bytes()
    assert len(data) == item['bytes'], item['path']
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
assert manifest['accepted_jobs'] == 3 and manifest['planned_canonical_jobs'] == 20
assert evidence['scope']['accepted_jobs_in_snapshot'] == 3
assert evidence['scope']['canonical_jobs'] == 20
assert evidence['scope']['common2_present'] is False
assert evidence['publication']['scientific_calls'] == 0
rows = list(csv.DictReader((ROOT / '3rows.csv').open()))
curves = list(csv.DictReader((ROOT / 'curves.csv').open()))
assert len(rows) == 3 and len(curves) == 18
for source, row in zip(evidence['accepted_rows'], rows):
    assert source['job_id'] == row['job_id']
    for key in ['arm', 'task', 'status', 'result_sha256', 'completion_sha256', 'observed_UTC']:
        assert row[key] == source[key], key
    for key in ['budget', 'seed', 'SUN', 'mSUN', 'AUDC', 'selected_generation']:
        assert math.isclose(float(row[key]), source[key], rel_tol=0, abs_tol=1e-14), key
    if 'failure_rate' in source:
        assert math.isclose(float(row['failure_rate']), source['failure_rate'], rel_tol=0, abs_tol=1e-14)
    else:
        assert row['failure_rate'] == ''
    for key, val in source['costs'].items():
        assert math.isclose(float(row[key]), val, rel_tol=1e-14, abs_tol=1e-14), key
    for key in ['surrogate_oracle_attempts', 'score_sentinels']:
        if key not in source['costs']:
            assert row[key] == ''
    points = [r for r in curves if r['job_id'] == row['job_id']]
    curve = [[int(r['candidate_oracle_attempt']), int(r['SUN'])] for r in points]
    assert curve == source['curve']
    assert [p[0] for p in curve] == list(range(int(row['budget']) + 1))
    assert all(p['result_sha256'] == row['result_sha256'] for p in points)
    budget = int(row['budget'])
    area = sum((right[0] - left[0]) * (right[1] + left[1]) / 2 for left, right in zip(curve, curve[1:]))
    assert source['SUN'] == curve[-1][1]
    assert source['mSUN'] == source['SUN'] / budget
    assert source['AUDC'] == 2 * area / budget**2
    assert source['costs']['candidate_oracle_attempts'] == budget
failures = evidence['original_initialization_failures']
assert len(failures) == 8 and len({r['job_id'] for r in failures}) == 8
for row in failures:
    assert row['recorded_request_ops'] == ['init']
    assert row['candidate_oracle_attempts'] == 0
    assert row['score'] is None
    assert row['decision_file_exists'] is False
    assert row['error_code'] == 'missing_asset'
sources = json.loads((ROOT / 'scientific_sha256.json').read_text())['files']
raw_sources = {r['job_id']: r for r in sources if r['role'] == 'accepted_raw_result'}
assert len(raw_sources) == 3
completion_sources = {r['job_id']: r for r in sources if r['role'] == 'accepted_completion_receipt'}
assert len(completion_sources) == 3
for row in rows:
    assert row['result_sha256'] == raw_sources[row['job_id']]['sha256']
    assert row['completion_sha256'] == completion_sources[row['job_id']]['sha256']
for filename in [r['path'] for r in manifest['files'] if r['path'] != 'validate.py']:
    data = (ROOT / filename).read_text()
    assert not any(s in data for s in ['/Users/', '/root/', '/mnt/', 'hfeno-master', 'GPU-', 'password', 'cookie', 'authorization']), filename
print('PASS: bundle hashes; 3 accepted rows; 18 curve points; SUN/mSUN/AUDC and costs; 8 unscored infrastructure failures.')
