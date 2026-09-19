"""Portable CSV-only checks for the frozen17-case audit and dated progress."""
from collections import defaultdict
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import math

D=Path(__file__).resolve().parent
def read(n):return json.loads((D/n).read_text())
def rows(n):
    with (D/n).open(newline='') as f:return list(csv.DictReader(f))
def near(a,b):return math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12)

def main():
    cases=rows('all17_cases.csv');episodes=rows('all68_episodes_and_costs.csv');points=rows('all68_discovery_curves.csv')
    summary=read('summary.json');progress=read('progress_summary.json');prov=read('provenance.json')
    pin='6b6cab9beeff97ef12d65246a7cf5d034821b27cf746ba881c7a6afa276491cd'
    assert summary['snapshot_sha256']==progress['snapshot_sha256']==prov['source_snapshot_sha256']==pin
    assert (len(cases),len(episodes),len(points))==(17,68,748)
    assert summary['repeated_seed_variance_estimated'] is False
    arms=tuple(summary['arms']);assert len(arms)==4
    curves=defaultdict(list)
    for r in points:
        assert (r['budget'],r['seed'])==('10','2')
        curves[r['arm'],r['task_id']].append((int(r['query']),int(r['SUN'])))
    indexed={}
    for r in episodes:
        key=r['arm'],r['task_id'];assert key not in indexed;indexed[key]=r
        c=curves[key];assert [x for x,y in c]==list(range(11)) and c[0]==(0,0)
        assert all(0<=y<=x for x,y in c) and int(r['candidate_oracle_attempts'])==10
        area=sum((x1-x0)*(y1+y0)/2 for (x0,y0),(x1,y1) in zip(c,c[1:]))
        assert near(r['AUDC'],2*area/100) and int(r['SUN'])==c[-1][1]
        positive=[x for x,y in c if y>0]
        assert r['first_positive_query']==(str(min(positive)) if positive else '')
        assert int(r['SUN_at_query5'])==c[5][1] and int(r['SUN_change_queries5_to10'])==c[-1][1]-c[5][1]
        assert len(r['result_sha256'])==len(r['receipt_sha256'])==64
    assert {r['task_id'] for r in cases}=={k[1] for k in indexed}
    for r in cases:
        task=r['task_id'];assert all((a,task) in indexed for a in arms)
        for arm in arms:
            for metric in ('SUN','AUDC'):assert near(r[arm+'_'+metric],indexed[arm,task][metric])
        for metric in ('SUN','AUDC'):
            full=float(indexed['full_support_aware',task][metric])
            assert near(r['full_minus_baseline_'+metric],full-float(indexed['baseline_reference',task][metric]))
            assert near(r['full_minus_UQ_'+metric],full-float(indexed['uq_only_support_aware',task][metric]))
        assert (r['full_and_ES_only_identical_SUN_curve']=='True')==(curves['full_support_aware',task]==curves['es_only_independent',task])
    for arm,totals in summary['arms'].items():
        rs=[r for r in episodes if r['arm']==arm]
        assert totals['n']==len(rs)==17 and totals['SUN_sum']==sum(int(r['SUN']) for r in rs)
        assert near(totals['AUDC_mean'],mean(float(r['AUDC']) for r in rs))
        for cost,st in totals['cost_fields'].items():
            known=[float(r[cost]) for r in rs if r[cost]!='']
            assert st['missing_rows']==len(rs)-len(known) and near(st['reported_sum'],sum(known))
    for reference,prefix in [('baseline_reference','full_minus_baseline'),('uq_only_support_aware','full_minus_UQ')]:
        for metric in ('SUN','AUDC'):
            ds=[float(indexed['full_support_aware',r['task_id']][metric])-float(indexed[reference,r['task_id']][metric]) for r in cases]
            st=summary[prefix][metric];assert near(st['mean'],mean(ds))
            assert [st['wins'],st['ties'],st['losses']]==[sum(x>1e-12 for x in ds),sum(abs(x)<=1e-12 for x in ds),sum(x< -1e-12 for x in ds)]
    assert sum(c['complete'] for c in progress['evaluation_cohorts'].values())==progress['completed_evaluations']==172
    assert sum(c['expected'] for c in progress['evaluation_cohorts'].values())==progress['registered_evaluations']==1080
    for c in progress['evaluation_cohorts'].values():assert c['complete']+c['failed']+c['claimed_no_result']+c['pending']==c['expected']
    pair_rows=rows('progress_pairs.csv')
    for arm,groups in progress['matched_baseline_comparisons_by_budget'].items():
        for budget,st in groups.items():
            selected=[r for r in pair_rows if r['arm']==arm and r['budget']==budget]
            assert len(selected)==st['n']
            assert sum(int(r['baseline_SUN']) for r in selected)==st['baseline_SUN_sum']
            assert sum(int(r['arm_SUN']) for r in selected)==st['arm_SUN_sum']
            assert near(mean(float(r['baseline_AUDC']) for r in selected),st['baseline_AUDC_mean'])
            assert near(mean(float(r['arm_AUDC']) for r in selected),st['arm_AUDC_mean'])
    core=rows('core5_matrix.csv');assert len(core)==180 and sum(r['status']=='accepted' for r in core)==35
    for c in rows('core5_seed_coverage.csv'):
        selected=[r for r in core if all(r[k]==c[k] for k in ('arm','budget','seed'))]
        assert len(selected)==int(c['expected'])==5 and sum(r['status']=='accepted' for r in selected)==int(c['accepted'])
    for name,expected in prov['byte_preserved_case_files'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==expected
    n=0
    if (D/'manifest.json').exists():
        for r in read('manifest.json')['files']:
            p=D/r['path'];assert p.parent==D and not p.is_symlink();b=p.read_bytes()
            assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['export_sha256'];n+=1
    print(json.dumps({'passed':True,'cases':17,'episodes':68,'curve_points_including_initial':748,
        'candidate_ORB_calls_in_case_cohort':680,'new_study_accepted_evaluations':172,
        'core5_accepted_of_180':35,'missing_cost_fields_preserved':True,
        'repeated_seed_variance_estimated':False,'manifest_files_checked':n,'new_scientific_calls':0}))

if __name__=='__main__':main()
