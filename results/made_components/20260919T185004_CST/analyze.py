"""Fixed-configuration evaluation-seed statistics from a dated cached snapshot.

Default: verify all derived tables. --write: render them deterministically.
No network, model, scientific source imports, fitting or oracle calls.
"""
from pathlib import Path
from collections import defaultdict
import argparse,csv,hashlib,io,json,math

ROOT=Path(__file__).resolve().parent
METRICS=('SUN','mSUN','AUDC')
BASE='baseline_reference'
def average(x):return math.fsum(x)/len(x) if x else None
def stats(x):
    mean=average(x);variance=math.fsum((v-mean)**2 for v in x)/(len(x)-1) if len(x)>1 else None
    return {'n':len(x),'mean':mean,'sample_variance_ddof1':variance,'SD':math.sqrt(variance) if variance is not None else None}
def field_seeds(seeds):return ';'.join(map(str,sorted(seeds)))
def csv_text(rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);return stream.getvalue()
def wtl(values):return [sum(v>1e-12 for v in values),sum(abs(v)<=1e-12 for v in values),sum(v < -1e-12 for v in values)]
def tag(n):return 'complete_three_evaluation_seeds' if n==3 else 'partial_fewer_than_three_seeds'
def build(data):
    assert data['source_snapshot_sha256']=='11c6430d9a3803c636f191b87b91ab892221168ac4c06863b0d889626ce1ea59'
    tasks=data['task_ids'];arms=data['arms'];budgets=data['budgets'];seeds=data['evaluation_seeds'];core=data['core5_task_ids']
    assert len(tasks)==len(set(tasks))==30 and budgets==[10,30,50] and seeds==[2,3,4] and len(core)==5
    observed={};registrations={};cohort_counts={}
    for cohort in data['cohorts']:
        arm=cohort['arm'];registrations[arm]=cohort['registration'];cohort_counts[arm]=cohort['counts']
        for row in cohort['rows']:
            key=(arm,row['budget'],row['seed'],row['task_id']);assert key not in observed
            assert arm in arms and row['budget'] in budgets and row['seed'] in seeds and row['task_id'] in tasks
            observed[key]=row
            if row['status']=='accepted':
                assert row['result'] and row['receipt'] and not row['failure']
                curve=row['curve'];budget=row['budget'];assert [p[0] for p in curve]==list(range(budget+1))
                assert all(0<=y<=x for x,y in curve)
                audc=math.fsum((y0+y1)*(x1-x0) for (x0,y0),(x1,y1) in zip(curve,curve[1:]))/budget**2
                assert math.isclose(audc,row['AUDC'],abs_tol=1e-12) and curve[-1][1]==row['SUN']
                assert row['costs']['candidate_oracle_attempts']==budget
                row['mSUN']=row['SUN']/budget
    def get(arm,budget,seed,task):return observed.get((arm,budget,seed,task))
    def complete(arm,budget,seed,task):
        row=get(arm,budget,seed,task);return row if row and row['status']=='accepted' else None
    matrix=[];curves=[];costs=[];seed_stats=[];core_matrix=[]
    cost_keys=sorted(set().union(*(set(r['costs']) for r in observed.values() if r['status']=='accepted')))
    for budget in budgets:
        for task in tasks:
            for arm in arms:
                available={seed:complete(arm,budget,seed,task) for seed in seeds};available={s:r for s,r in available.items() if r}
                for metric in METRICS:
                    seed_stats.append({'task_id':task,'budget':budget,'arm':arm,'metric':metric,
                        'observed_seeds':field_seeds(available),'missing_seeds':field_seeds(set(seeds)-set(available)),
                        'planned_seeds':field_seeds(seeds),'coverage':tag(len(available)),**stats([available[s][metric] for s in sorted(available)])})
                for seed in seeds:
                    row=get(arm,budget,seed,task);accepted=row and row['status']=='accepted'
                    value={'task_id':task,'budget':budget,'seed':seed,'arm':arm,'status':row['status'] if row else 'unclaimed',
                        'job_id':row['job_id'] if row else '',**{m:row[m] if accepted else None for m in METRICS},
                        'result_path':row['result']['path'] if row and row.get('result') else '',
                        'result_sha256':row['result']['sha256'] if row and row.get('result') else '',
                        'receipt_sha256':row['receipt']['sha256'] if row and row.get('receipt') else '',
                        'registration_sha256':registrations[arm]['sha256']}
                    matrix.append(value)
                    if task in core:core_matrix.append(value)
                    if accepted:
                        costs.append({k:value[k] for k in ('task_id','budget','seed','arm','job_id')}|{k:row['costs'].get(k) for k in cost_keys})
                        for query,sun in row['curve']:curves.append({k:value[k] for k in ('task_id','budget','seed','arm','job_id')}|{'query':query,'SUN':sun})
    paired=[];paired_seed_stats=[];paired_summary={};common_rows=[];common_summary={}
    others=[a for a in arms if a!=BASE]
    for arm in others:
        selected=[]
        for budget in budgets:
            for task in tasks:
                per_seed={}
                for seed in seeds:
                    a,b=complete(BASE,budget,seed,task),complete(arm,budget,seed,task)
                    if not (a and b):continue
                    row={'task_id':task,'budget':budget,'seed':seed,'arm':arm}
                    for m in METRICS:row.update({f'baseline_{m}':a[m],f'arm_{m}':b[m],f'delta_{m}':b[m]-a[m]})
                    row.update(baseline_result_sha256=a['result']['sha256'],arm_result_sha256=b['result']['sha256'])
                    paired.append(row);selected.append(row);per_seed[seed]=row
                for metric in METRICS:
                    paired_seed_stats.append({'task_id':task,'budget':budget,'arm_minus_baseline':arm,'metric':metric,
                        'paired_seeds':field_seeds(per_seed),'missing_paired_seeds':field_seeds(set(seeds)-set(per_seed)),
                        'coverage':tag(len(per_seed)),**stats([per_seed[s]['delta_'+metric] for s in sorted(per_seed)])})
        paired_summary[arm]={}
        for budget in budgets:
            rr=[r for r in selected if r['budget']==budget]
            entry={'complete_task_seed_pairs':len(rr),'seeds':sorted({r['seed'] for r in rr})}
            for m in METRICS:
                entry[m]={'baseline_sum':math.fsum(r['baseline_'+m] for r in rr),'arm_sum':math.fsum(r['arm_'+m] for r in rr),
                    'baseline_mean':average([r['baseline_'+m] for r in rr]),'arm_mean':average([r['arm_'+m] for r in rr]),
                    'paired_delta_mean':average([r['delta_'+m] for r in rr]),'wins_ties_losses':wtl([r['delta_'+m] for r in rr])}
            paired_summary[arm][str(budget)]=entry
    for budget in budgets:
        for task in tasks:
            for seed in seeds:
                rr={arm:complete(arm,budget,seed,task) for arm in arms}
                if not all(rr.values()):continue
                value={'task_id':task,'budget':budget,'seed':seed}
                for arm,r in rr.items():
                    for m in METRICS:value[arm+'_'+m]=r[m]
                common_rows.append(value)
        selected=[r for r in common_rows if r['budget']==budget]
        common_summary[str(budget)]={'complete_task_seed_sets':len(selected),'arms':{}}
        for arm in arms:
            common_summary[str(budget)]['arms'][arm]={'SUN_sum':math.fsum(r[arm+'_SUN'] for r in selected),
                'AUDC_mean':average([r[arm+'_AUDC'] for r in selected]),
                'SUN_WTL_vs_baseline':wtl([r[arm+'_SUN']-r[BASE+'_SUN'] for r in selected]),
                'AUDC_WTL_vs_baseline':wtl([r[arm+'_AUDC']-r[BASE+'_AUDC'] for r in selected])}
    # Aggregate each complete seed over the same predeclared systems FIRST.
    # Only then measure dispersion across seeds; partial-system means stay NA.
    seed_aggregates=[];aggregate_stats=[];aggregate_pairs=[];agg_index={}
    for label,group_tasks in [('all30',tasks),('core5',core)]:
        for budget in budgets:
            for arm in arms:
                available={}
                for seed in seeds:
                    rr=[complete(arm,budget,seed,task) for task in group_tasks];n=sum(r is not None for r in rr)
                    full=n==len(group_tasks);metric_values={m:average([r[m] for r in rr]) if full else None for m in METRICS}
                    entry={'cohort':label,'budget':budget,'seed':seed,'arm':arm,'completed_systems':n,'expected_systems':len(group_tasks),
                        'status':'complete_seed_cohort' if full else 'partial_seed_cohort',**{m+'_mean':v for m,v in metric_values.items()}}
                    seed_aggregates.append(entry);agg_index[label,budget,arm,seed]=entry
                    if full:available[seed]=metric_values
                for metric in METRICS:
                    aggregate_stats.append({'cohort':label,'budget':budget,'arm':arm,'metric':metric,
                        'complete_cohort_seeds':field_seeds(available),'missing_complete_cohort_seeds':field_seeds(set(seeds)-set(available)),
                        'coverage':tag(len(available)),**stats([available[s][metric] for s in sorted(available)])})
            for arm in others:
                available={}
                for seed in seeds:
                    a,b=agg_index[label,budget,BASE,seed],agg_index[label,budget,arm,seed]
                    if a['status']==b['status']=='complete_seed_cohort':available[seed]={m:b[m+'_mean']-a[m+'_mean'] for m in METRICS}
                for metric in METRICS:
                    aggregate_pairs.append({'cohort':label,'budget':budget,'arm_minus_baseline':arm,'metric':metric,
                        'complete_paired_cohort_seeds':field_seeds(available),'missing_complete_paired_cohort_seeds':field_seeds(set(seeds)-set(available)),
                        'coverage':tag(len(available)),**stats([available[s][metric] for s in sorted(available)])})
    counts={};cost_totals={}
    for arm in arms:
        rr=[r for r in matrix if r['arm']==arm];accepted=[r for r in rr if r['status']=='accepted']
        counts[arm]={state:sum(r['status']==state for r in rr) for state in ('accepted','claimed_pending','failed','reserved_unknown','unclaimed')}
        assert counts[arm]['accepted']==cohort_counts[arm]['complete'] and counts[arm]['failed']==cohort_counts[arm]['failed']
        assert counts[arm]['claimed_pending']==cohort_counts[arm]['claimed_no_result'] and counts[arm]['unclaimed']==cohort_counts[arm]['pending']
        rc=[r for r in costs if r['arm']==arm]
        cost_totals[arm]={k:{'reported_sum':math.fsum(r[k] for r in rc if r[k] is not None),'missing_accepted_rows':sum(r[k] is None for r in rc)} for k in cost_keys}
    core_coverage={arm:{str(b):{str(s):sum(complete(arm,b,s,t) is not None for t in core) for s in seeds} for b in budgets} for arm in arms}
    repeated={arm:sum(r['arm']==arm and r['metric']=='SUN' and r['n']==3 for r in seed_stats) for arm in arms}
    summary={'schema':'fixed_configuration_evaluation_seed_statistics_v1','observed_CST':data['observed_CST'],'observed_UTC':data['observed_UTC'],
        'source_snapshot_sha256':data['source_snapshot_sha256'],'planned_evaluations':1080,'accepted_evaluations':sum(c['accepted'] for c in counts.values()),
        'counts':counts,'paired_comparisons':paired_summary,'common_four_arm':common_summary,'core5_coverage':core_coverage,
        'configurations_with_all_three_evaluation_seeds':repeated,'cost_totals_accepted_only':cost_totals,
        'variance_definition':'Across evaluation seeds2/3/4 within a fixed system,budget,arm and frozen trained state. ddof=1; n<2 variance/SD is NA; n<3 remains partial.',
        'aggregate_variance_definition':'First average over every system in an entire all30/core5 cohort for each seed; then calculate across-seed variance, using only complete seed cohorts.',
        'training_randomness_repeated':False,'cross_system_variance_used_as_seed_variance':False,
        'comparisons_use_matching_system_budget_seed':True,'partial_pair_totals_are_not_full_matrix_effects':True,'new_scientific_calls_for_export':0}
    outputs={'all_system_results.csv':matrix,'completed_curves.csv':curves,'timings_and_costs.csv':costs,'seed_statistics.csv':seed_stats,
        'paired_results.csv':paired,'paired_seed_statistics.csv':paired_seed_stats,'common_four_arm.csv':common_rows,
        'aggregate_seed_means.csv':seed_aggregates,'aggregate_seed_statistics.csv':aggregate_stats,
        'paired_aggregate_seed_statistics.csv':aggregate_pairs,'core5_matrix.csv':core_matrix}
    assert len(matrix)==1080 and len(costs)==205 and len(curves)==2255 and len(core_matrix)==180
    outputs={name:csv_text(rr) for name,rr in outputs.items()};outputs['summary.json']=json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    return outputs,summary
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');args=ap.parse_args()
    inputs=json.loads((ROOT/'study_inputs.json').read_text());outputs,summary=build(inputs)
    for name,value in outputs.items():
        if args.write:(ROOT/name).write_bytes(value.encode())
        else:assert (ROOT/name).read_bytes()==value.encode(),name
    checked=0
    if not args.write and (ROOT/'manifest.json').exists():
        for row in json.loads((ROOT/'manifest.json').read_text())['files']:
            data=(ROOT/row['path']).read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['export_sha256'];checked+=1
    print(json.dumps({'passed':True,'accepted':summary['accepted_evaluations'],'planned':1080,'completed_curves':2255,
        'same_configuration_seed_statistics':True,'cross_system_variance_substituted':False,'all3_configurations':summary['configurations_with_all_three_evaluation_seeds'],
        'manifest_files_checked':checked,'scientific_calls_for_recomputation':0},sort_keys=True))
if __name__=='__main__':main()
