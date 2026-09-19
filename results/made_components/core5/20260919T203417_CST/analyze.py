"""Fixed-five MADE seed statistics; local cached data only, no scientific calls."""
from pathlib import Path
from fractions import Fraction
import csv, hashlib, json, math, statistics

D = Path(__file__).resolve().parent
P = json.loads((D / 'provenance.json').read_text())
assert hashlib.sha256((D / 'study_inputs.json').read_bytes()).hexdigest() == P['scientific_projection_sha256']
V = json.loads((D / 'study_inputs.json').read_text())
SYSTEMS = V['core_systems']; SEEDS = V['evaluation_seeds']; BUDGET = V['budget']
ARMS = [a['arm'] for a in V['arms']]
BASE = 'baseline_reference'; FULL = 'full_support_aware'; ES = 'es_only_independent'; UQ = 'uq_only_support_aware'
METRICS = ['SUN', 'AUDC', 'mSUN']

def table(name, rows):
    with (D / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)

def js(name, value):
    (D / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def stats(values):
    n = len(values)
    return {'mean': statistics.mean(values) if n else None,
            'sample_variance': statistics.variance(values) if n > 1 else None,
            'sample_SD': statistics.stdev(values) if n > 1 else None}

def sign(x): return 'positive' if x > 1e-12 else 'negative' if x < -1e-12 else 'tie'

def ref(r, key, field): return (r.get(key) or {}).get(field)

assert SYSTEMS == ['Al-Li-V', 'Al-V-Zn', 'Au-K-Tb', 'Co-Dy-W', 'Co-Mg-Na'] and SEEDS == [2, 3, 4] and BUDGET == 10
observed = {r['job_id']: r for r in V['observed_core_rows']}
assert len(observed) == len(V['observed_core_rows'])
assignments = {(j['arm'], j['task_id'], j['seed']): j for j in V['registered_core_jobs']}
assert len(assignments) == 60
runs, curves, costs, complete = [], [], [], {}
all_cost_names = sorted({k for r in observed.values() for k in (r.get('costs') or {})})
for arm in ARMS:
    for task in SYSTEMS:
        for seed in SEEDS:
            j = assignments[arm, task, seed]; r = observed.get(j['job_id'], {})
            if r: assert all(r[k] == j[k] for k in ('arm', 'task_id', 'seed', 'budget', 'job_id'))
            done = bool(r.get('receipt') and r.get('result') and not r.get('failure'))
            status = 'complete' if done else 'failed' if r.get('failure') else 'claimed_without_result' if r.get('claimed') else 'pending_no_claim_in_snapshot'
            row = dict(j, status=status, SUN=r['SUN'] if done else None, AUDC=r['AUDC'] if done else None,
                mSUN=r['SUN'] / BUDGET if done else None,
                registration_fingerprint=next(a['registration_fingerprint'] for a in V['arms'] if a['arm'] == arm),
                receipt_artifact=ref(r,'receipt','path'), receipt_sha256=ref(r,'receipt','sha256'),
                result_artifact=ref(r,'result','path'), result_sha256=ref(r,'result','sha256'))
            runs.append(row)
            if not done: continue
            curve = r['curve']; assert curve[0] == [0, 0] and [x[0] for x in curve] == list(range(11))
            assert all(isinstance(y,int) and 0 <= y <= x for x,y in curve)
            area_twice = sum((b[0]-a[0])*(a[1]+b[1]) for a,b in zip(curve, curve[1:]))
            expected = Fraction(area_twice, BUDGET**2)
            assert math.isclose(float(expected), r['AUDC'], abs_tol=1e-12) and r['SUN'] == curve[-1][1]
            assert r['costs']['candidate_oracle_attempts'] == BUDGET
            complete[arm,task,seed] = row
            for q,sun in curve: curves.append({'job_id':j['job_id'],'arm':arm,'task_id':task,'seed':seed,'budget':10,'candidate_attempt':q,'official_SUN':sun})
            costs.append(dict(job_id=j['job_id'],arm=arm,task_id=task,seed=seed,**{k:r['costs'].get(k) for k in all_cost_names}))
assert [sum(k[0]==a for k in complete) for a in (BASE,FULL,ES,UQ)] == [15,15,15,11]
table('runs.csv', runs); table('completed_curves.csv', curves); table('timings_and_costs.csv', costs)

system_stats=[]
for task in SYSTEMS:
    for arm in ARMS:
        present=[s for s in SEEDS if (arm,task,s) in complete]
        row={'task_id':task,'budget':10,'arm':arm,'n':len(present),'planned_n':3,
            'available_seeds':';'.join(map(str,present)),'missing_seeds':';'.join(str(s) for s in SEEDS if s not in present),
            'all_planned_seeds_complete':len(present)==3,'variance_axis':'evaluation_seed_within_fixed_system_budget_arm','ddof':1}
        for m in METRICS:
            values=[complete[arm,task,s][m] for s in present]
            row[m+'_values']=json.dumps(values)
            row.update({m+'_'+k:v for k,v in stats(values).items()})
        system_stats.append(row)
table('per_system_seed_statistics.csv',system_stats)

pairs=[];pair_stats=[]
for arm in (FULL,ES):
    for task in SYSTEMS:
        rr=[]
        for seed in SEEDS:
            a,b=complete[BASE,task,seed],complete[arm,task,seed]
            r={'comparison':arm+'_minus_'+BASE,'arm':arm,'baseline':BASE,'task_id':task,'budget':10,'seed':seed,
                'baseline_job_id':a['job_id'],'arm_job_id':b['job_id'],'baseline_result_sha256':a['result_sha256'],'arm_result_sha256':b['result_sha256']}
            for m in METRICS:r.update({m+'_baseline':a[m],m+'_arm':b[m],m+'_delta':b[m]-a[m],m+'_sign':sign(b[m]-a[m])})
            pairs.append(r);rr.append(r)
        sr={'comparison':arm+'_minus_'+BASE,'task_id':task,'n_paired_seeds':3,'paired_seeds':'2;3;4','ddof':1}
        for m in METRICS:sr.update({m+'_delta_'+k:v for k,v in stats([r[m+'_delta'] for r in rr]).items()})
        pair_stats.append(sr)
table('paired_run_differences.csv',pairs);table('paired_system_seed_statistics.csv',pair_stats)

aggregates=[];aggregate_stats=[]
for arm in ARMS:
    for seed in SEEDS:
        present=[task for task in SYSTEMS if (arm,task,seed) in complete]
        full=len(present)==5
        r={'arm':arm,'seed':seed,'budget':10,'n_systems_complete':len(present),'expected_systems':5,
            'missing_systems':';'.join(t for t in SYSTEMS if t not in present),'same_five_system_seed_cohort_complete':full,
            'SUN_total':sum(complete[arm,t,seed]['SUN'] for t in present) if full else None,
            'SUN_macro_mean':statistics.mean(complete[arm,t,seed]['SUN'] for t in present) if full else None,
            'AUDC_macro_mean':statistics.mean(complete[arm,t,seed]['AUDC'] for t in present) if full else None,
            'mSUN_macro_mean':statistics.mean(complete[arm,t,seed]['mSUN'] for t in present) if full else None}
        aggregates.append(r)
    ar=[r for r in aggregates if r['arm']==arm and r['same_five_system_seed_cohort_complete']]
    full=len(ar)==3
    sr={'arm':arm,'budget':10,'n_complete_seed_cohorts':len(ar),'planned_seed_cohorts':3,
        'complete_15_run_cohort':full,'variance_axis':'three_seed_five_system_aggregate','ddof':1,
        'incomplete_aggregate_policy':'no_UQ_cross_seed_aggregate_until_all_15_complete'}
    for m in ('SUN_total','SUN_macro_mean','AUDC_macro_mean','mSUN_macro_mean'):
        sr.update({m+'_'+k:v for k,v in (stats([r[m] for r in ar]) if full else {'mean':None,'sample_variance':None,'sample_SD':None}).items()})
    aggregate_stats.append(sr)
table('five_system_aggregates_by_seed.csv',aggregates);table('aggregate_seed_statistics.csv',aggregate_stats)

agg_by={(r['arm'],r['seed']):r for r in aggregates};agg_pairs=[];agg_pair_stats=[]
for arm in (FULL,ES):
    ar=[]
    for seed in SEEDS:
        a,b=agg_by[BASE,seed],agg_by[arm,seed]
        r={'comparison':arm+'_minus_'+BASE,'seed':seed,'n_fixed_systems':5}
        for m in ('SUN_total','SUN_macro_mean','AUDC_macro_mean','mSUN_macro_mean'):
            r.update({m+'_baseline':a[m],m+'_arm':b[m],m+'_delta':b[m]-a[m]})
        agg_pairs.append(r);ar.append(r)
    sr={'comparison':arm+'_minus_'+BASE,'n_seed_aggregates':3,'ddof':1,'variance_axis':'paired_three_evaluation_seed_aggregates'}
    for m in ('SUN_total','SUN_macro_mean','AUDC_macro_mean','mSUN_macro_mean'):
        sr.update({m+'_delta_'+k:v for k,v in stats([r[m+'_delta'] for r in ar]).items()})
    agg_pair_stats.append(sr)
table('paired_aggregates_by_seed.csv',agg_pairs);table('paired_aggregate_seed_statistics.csv',agg_pair_stats)

summary={'schema':'fixed_core5_B10_three_evaluation_seed_summary_v1','observed_CST':V['observation_finished_CST'],
 'model':V['model'],'systems':SYSTEMS,'seeds':SEEDS,'budget':10,'all_new_study':{},'core':{},'baseline_full':{},
 'all_four_arm_main_ablation_complete':False,'UQ_full15_aggregate_comparison_performed':False,
 'variance_is_training_replication':False,'variance_is_between_systems':False,'new_scientific_calls':0}
for k in ('expected','complete','failed','claimed_no_result','pending'):summary['all_new_study'][k]=sum(a['all1080_arm_progress'][k] for a in V['arms'])
assert summary['all_new_study']=={'expected':1080,'complete':254,'failed':0,'claimed_no_result':10,'pending':816}
for arm in ARMS:
 ar=[r for r in runs if r['arm']==arm]
 summary['core'][arm]={'expected':15,**{state:sum(r['status']==state for r in ar) for state in ['complete','claimed_without_result','pending_no_claim_in_snapshot','failed']},
 'missing_run_keys':[{'task_id':r['task_id'],'seed':r['seed'],'status':r['status'],'job_id':r['job_id']} for r in ar if r['status']!='complete']}
fullpairs=[r for r in pairs if r['arm']==FULL]
for m in ('SUN','AUDC'):
 ps=[r for r in pair_stats if r['comparison'].startswith(FULL)]
 vals=[r[m+'_delta_mean'] for r in ps]
 summary['baseline_full'][m]={'baseline_total_across_15_runs':sum(complete[BASE,t,s][m] for t in SYSTEMS for s in SEEDS),
  'full_total_across_15_runs':sum(complete[FULL,t,s][m] for t in SYSTEMS for s in SEEDS),
  'baseline_mean_per_run':statistics.mean(complete[BASE,t,s][m] for t in SYSTEMS for s in SEEDS),
  'full_mean_per_run':statistics.mean(complete[FULL,t,s][m] for t in SYSTEMS for s in SEEDS),
  'mean_paired_delta':statistics.mean(r[m+'_delta'] for r in fullpairs),
  'system_mean_signs':{x:sum(sign(v)==x for v in vals) for x in ('positive','tie','negative')},
  'run_pair_signs':{x:sum(r[m+'_sign']==x for r in fullpairs) for x in ('positive','tie','negative')}}
summary['completed_core_candidate_calls']=sum(r['candidate_oracle_attempts'] for r in costs)
summary['completed_curve_points']=len(curves)
summary['core_expected_runs']=60;summary['core_completed_runs']=len(complete)
summary['all_new_complete_are_B10_in_source_snapshot']=True
assert summary['baseline_full']['SUN']['baseline_total_across_15_runs']==22 and summary['baseline_full']['SUN']['full_total_across_15_runs']==20
assert math.isclose(summary['baseline_full']['AUDC']['baseline_mean_per_run'],.156) and math.isclose(summary['baseline_full']['AUDC']['full_mean_per_run'],.144)
js('summary.json',summary)
# Independent algebraic variance formula checks the axis and ddof.
for row in system_stats:
 for m in METRICS:
  x=json.loads(row[m+'_values'])
  if len(x)>1:
   v=(sum(z*z for z in x)-sum(x)**2/len(x))/(len(x)-1)
   assert math.isclose(v,row[m+'_sample_variance'],rel_tol=1e-10,abs_tol=1e-12)
for arm in (BASE,FULL,ES):
 a=next(x for x in aggregate_stats if x['arm']==arm)
 assert math.isclose(a['SUN_macro_mean_mean'],statistics.mean(complete[arm,t,s]['SUN'] for t in SYSTEMS for s in SEEDS))
 assert math.isclose(a['AUDC_macro_mean_mean'],statistics.mean(complete[arm,t,s]['AUDC'] for t in SYSTEMS for s in SEEDS))
js('validation.json',{'passed':True,'registered_cells':60,'completed_runs':56,'curves_recomputed':56,'curve_points':616,
 'SUN_and_AUDC_recomputed_from_official_curves':True,'per_system_variance_independently_checked':True,
 'complete_15_run_arms':[BASE,FULL,ES],'incomplete_UQ_aggregate_suppressed':True,'all_negative_and_tied_cases_retained':True,
 'source_projection_sha256':hashlib.sha256((D/'study_inputs.json').read_bytes()).hexdigest(),'scientific_calls':0,'remote_calls':0})
print(json.dumps({'passed':True,'completed_core_runs':len(complete),'SUN':summary['baseline_full']['SUN'],'AUDC':summary['baseline_full']['AUDC']},sort_keys=True))
