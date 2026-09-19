"""Validate the accepted SnAr export. Standard library; no network or science."""
from collections import defaultdict
from pathlib import Path
from statistics import mean, variance, stdev
import csv
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def rows(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))
def close(a,b,*,variance_value=False):
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-22 if variance_value else 1e-12)

def main():
    p=read('protocol.json');summary=read('summary.json');accepted=read('acceptance_evidence.json')
    fingerprint=hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert fingerprint==summary['protocol_fingerprint']
    assert len(p['evaluation']['arms'])==7 and p['evaluation']['seeds']==list(range(5101,5106))
    assert accepted['actual_completion']['complete'] and accepted['actual_completion']['passed']
    assert accepted['actual_completion']['original_report_remains_failed'] and accepted['original_failure_preserved']
    assert accepted['actual_completion']['new_calls_performed_by_recovery']==accepted['actual_completion']['new_model_or_training_calls']==0
    audit=accepted['actual_acceptance']['summary']
    assert audit['complete'] and audit['passed'] and audit['unique_physical_attempts']==2300
    assert (audit['episodes'],audit['adaptation_episodes'],audit['test_episodes'])==(60,25,35)
    assert (audit['adaptation_oracle_calls'],audit['test_oracle_calls'],audit['failed_or_unknown_physical_attempts'])==(550,1750,0)
    refs={r['path']:r['sha256'] for r in accepted['actual_completion']['outputs']}
    for r in [accepted['actual_acceptance']['source'],accepted['actual_report_reference']]:assert refs[r['path']]==r['sha256']
    assert read('original_report_failure.json')['original_reference']['sha256']==accepted['original_failure']['sha256']
    assert read('original_report_failure.json')['observed_payload']['error_type']=='ModuleNotFoundError'

    episodes={x['summary']['name']:x for x in read('episode_summaries.json')}
    matrix=rows('per_seed.csv');assert len(matrix)==len(episodes)==35
    expected={(a,s) for a in p['evaluation']['arms'] for s in p['evaluation']['seeds']}
    assert {(r['arm'],int(r['seed'])) for r in matrix}==expected
    curve=defaultdict(list)
    for point in rows('curves.csv'):curve[point['name']].append(point)
    assert set(curve)==set(episodes) and sum(map(len,curve.values()))==1750
    by=defaultdict(list)
    for row in matrix:
        v=episodes[row['name']]['summary'];points=curve[row['name']]
        assert row['original_summary_sha256']==episodes[row['name']]['original_summary']['sha256']
        assert row['acceptance']=='separate_derived_acceptance_passed'
        assert int(row['queries'])==int(row['budget'])==50 and int(row['prior_rows'])==550
        assert row['protocol_fingerprint']==fingerprint
        assert [int(x['query_index']) for x in points]==list(range(1,51))
        hv=[float(x['HV']) for x in points]
        assert hv==v['hv_curve'] and all(math.isfinite(x) for x in hv)
        assert all(b>=a-1e-12 for a,b in zip(hv,hv[1:]))
        assert close(row['final_hv'],hv[-1]) and close(row['mean_querywise_hv'],mean(hv))
        assert close(row['final_hv_gain'],hv[-1]-float(row['prior_hv']))
        assert close(row['mean_querywise_hv_gain'],mean(hv)-float(row['prior_hv']))
        for point in points:assert close(point['HV_gain_from_shared_prior'],float(point['HV'])-float(row['prior_hv']))
        by[row['arm']].append(v)
    assert len({r['prior_fingerprint'] for r in matrix})==1 and len({r['prior_hv'] for r in matrix})==1
    for values in by.values():values.sort(key=lambda x:x['seed'])
    accepted_report=read('accepted_report_fields.json')
    checked=0
    for arm,stats in accepted_report['arms'].items():
        for metric,st in stats.items():
            values=[x[metric] for x in by[arm]];assert st['per_seed']==values and st['n']==5
            assert close(st['mean'],mean(values)) and close(st['sample_variance'],variance(values),variance_value=True)
            assert close(st['sample_std'],stdev(values));checked+=1
    for stat in rows('statistics.csv'):
        values=[r[stat['metric']] for r in by[stat['arm']] if r[stat['metric']] is not None]
        assert int(stat['n'])==len(values) and json.loads(stat['raw_values_seed_order'])==values
        if not values:assert stat['mean']==stat['sample_variance_ddof1']==stat['SD']==''
        else:
            assert close(stat['mean'],mean(values)) and close(stat['sample_variance_ddof1'],variance(values),variance_value=True)
            assert close(stat['SD'],stdev(values))
    for row in rows('ablation_deltas.csv'):
        left=next(r for r in by[row['reference_arm']] if r['seed']==int(row['seed']))
        right=next(r for r in by[row['arm']] if r['seed']==int(row['seed']))
        assert close(row['paired_delta'],right[row['metric']]-left[row['metric']])
    for metric,st in accepted_report['paired_full_minus_base'].items():
        delta=[b[metric]-a[metric] for a,b in zip(by['qwen_base'],by['uq_esopt'])]
        assert st['per_seed_delta']==delta and close(st['mean_delta'],mean(delta))
        assert close(st['sample_variance'],variance(delta),variance_value=True)
        assert [st['wins'],st['ties'],st['losses']]==[sum(x>1e-12 for x in delta),sum(abs(x)<=1e-12 for x in delta),sum(x< -1e-12 for x in delta)]
        assert st==summary['full_minus_base'][metric]
    selection=read('prior_and_selection.json')
    assert selection['full_checkpoint_selection']['selected']['generation']==0
    assert all(a['hv_curve']==b['hv_curve'] for a,b in zip(by['uq_esopt'],by['uq_only']))
    assert all(a['hv_curve']==b['hv_curve'] for a,b in zip(by['es_only'],by['qwen_base']))
    count=0
    if (ROOT/'manifest.json').exists():
        for item in read('manifest.json')['files']:
            path=ROOT/item['path'];assert path.parent==ROOT and not path.is_symlink()
            data=path.read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['export_sha256'];count+=1
    print(json.dumps({'passed':True,'test_episodes':35,'arms':7,'seeds_per_arm':5,'test_curve_points':1750,
        'accepted_distinct_physical_calls':2300,'actual_accepted_report_metric_sets_checked':checked,
        'manifest_files_verified':count,'full_minus_base_negative_mean_preserved':True,
        'original_report_failure_preserved':True,'new_science_or_model_calls':0}))

if __name__=='__main__':main()
