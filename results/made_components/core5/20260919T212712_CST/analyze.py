"""Recompute the fixed60-cell MADE ablation from cached scientific data only."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import argparse
import csv
import hashlib
import io
import json
import math
import statistics as st

D=Path(__file__).resolve().parent
BASE='baseline_reference';UQ='uq_only_support_aware';ES='es_only_independent';FULL='full_support_aware'
ARMS=[BASE,UQ,ES,FULL]
SYSTEMS=['Al-Li-V','Al-V-Zn','Au-K-Tb','Co-Dy-W','Co-Mg-Na'];SEEDS=[2,3,4]
METRICS=['SUN','AUDC','mSUN']
CONTRASTS=[
 ('UQ_minus_base','pairwise',{UQ:1,BASE:-1},'Enable the entire UQ/support controller at theta0; does not isolate the NN'),
 ('ES_minus_base','pairwise',{ES:1,BASE:-1},'Independent no-UQ ES policy versus theta0 without controller'),
 ('full_minus_UQ','pairwise',{FULL:1,UQ:-1},'UQ-guided ES policy versus theta0 with support controller'),
 ('full_minus_ES','pairwise',{FULL:1,ES:-1},'UQ-trained-and-controlled algorithm versus independently trained ES; weights differ'),
 ('full_minus_base','pairwise',{FULL:1,BASE:-1},'Complete algorithm versus baseline'),
 ('UQ_minus_ES','pairwise',{UQ:1,ES:-1},'Two different component algorithms; not an isolated factorial effect'),
 ('average_UQ_algorithm_contrast','algorithm_factorial',{UQ:.5,BASE:-.5,FULL:.5,ES:-.5},'Half of (UQ-base)+(full-ES); algorithm-level, not fixed-weight causal effect'),
 ('average_ES_algorithm_contrast','algorithm_factorial',{ES:.5,BASE:-.5,FULL:.5,UQ:-.5},'Half of (ES-base)+(full-UQ); no independent training replication'),
 ('algorithm_interaction','algorithm_factorial',{FULL:1,UQ:-1,ES:-1,BASE:1},'(full-UQ)-(ES-base); different learned ES policies, not a pure neural interaction'),
]

def stats(x):
    assert len(x)==3
    mean=st.mean(x);var=st.variance(x)
    alternative=sum((a-b)**2 for i,a in enumerate(x) for b in x[i+1:])/(len(x)*(len(x)-1))
    assert math.isclose(var,alternative,abs_tol=1e-12,rel_tol=1e-10)
    return dict(mean=mean,sample_variance=var,sample_SD=st.stdev(x))

def sign(v):return 'positive' if v>1e-12 else 'negative' if v< -1e-12 else 'tie'

def signs(x):return {s:sum(sign(v)==s for v in x)for s in ['positive','tie','negative']}

def csv_bytes(rows):
    stream=io.StringIO(newline='');w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return stream.getvalue().encode()

def json_bytes(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()

def compute(v):
    assert v['core_systems']==SYSTEMS and v['evaluation_seeds']==SEEDS and v['budget']==10
    cells={(r['arm'],r['task_id'],r['seed']):r for r in v['observed_core_rows']}
    registered={(r['arm'],r['task_id'],r['seed']):r for r in v['registered_core_jobs']}
    assert len(cells)==len(registered)==len(v['observed_core_rows'])==60
    assert set(cells)==set(registered)
    runs=[];curves=[];costs=[];metrics={}
    costkeys=sorted({k for r in cells.values()for k in r['costs']})
    for arm in ARMS:
        for task in SYSTEMS:
            for seed in SEEDS:
                r=cells[arm,task,seed];job=registered[arm,task,seed]
                assert all(r[k]==job[k]for k in ['arm','task_id','seed','budget','job_id'])
                assert r['receipt'] and r['result'] and not r['failure']
                curve=r['curve'];assert [a for a,b in curve]==list(range(11)) and curve[0]==[0,0]
                assert all(isinstance(b,int)and 0<=b<=a for a,b in curve)
                audc=Fraction(sum(a[1]+b[1] for a,b in zip(curve,curve[1:])),100)
                assert math.isclose(float(audc),r['AUDC'],abs_tol=1e-12) and curve[-1][1]==r['SUN']
                assert r['costs']['candidate_oracle_attempts']==10
                metrics[arm,task,seed]=dict(SUN=r['SUN'],AUDC=r['AUDC'],mSUN=r['SUN']/10)
                run=dict(arm=arm,task_id=task,seed=seed,budget=10,job_id=r['job_id'],status='complete',**metrics[arm,task,seed],
                    receipt_artifact=r['receipt']['path'],receipt_sha256=r['receipt']['sha256'],result_artifact=r['result']['path'],result_sha256=r['result']['sha256'])
                runs.append(run)
                curves.extend(dict(job_id=r['job_id'],arm=arm,task_id=task,seed=seed,candidate_attempt=i,official_SUN=s)for i,s in curve)
                costs.append(dict(job_id=r['job_id'],arm=arm,task_id=task,seed=seed,**{k:r['costs'].get(k)for k in costkeys}))
    cost_summary=[]
    for arm in ARMS:
        rr=[r for r in costs if r['arm']==arm]
        out=dict(arm=arm,complete_runs=15)
        for name in ['candidate_oracle_attempts','initialization_oracle_attempts','surrogate_oracle_attempts','wall_seconds','graph_seconds']:
            known=[r[name]for r in rr if r[name] is not None]
            out[name+'_observed_n']=len(known)
            out[name+'_sum']=sum(known)if len(known)==len(rr)else None
        cost_summary.append(out)
    systems=[]
    for task in SYSTEMS:
        for arm in ARMS:
            row=dict(task_id=task,arm=arm,budget=10,seeds='2;3;4',n=3,ddof=1,variance_axis='evaluation_seed_within_fixed_system_and_arm')
            for m in METRICS:
                x=[metrics[arm,task,s][m]for s in SEEDS];row[m+'_values']=json.dumps(x);row.update({m+'_'+k:z for k,z in stats(x).items()})
            systems.append(row)
    aggregate=[];aggregate_stats=[]
    for arm in ARMS:
        rr=[]
        for seed in SEEDS:
            r=dict(arm=arm,seed=seed,budget=10,n_fixed_systems=5,SUN_total=sum(metrics[arm,t,seed]['SUN']for t in SYSTEMS))
            r.update({m+'_macro_mean':st.mean(metrics[arm,t,seed][m]for t in SYSTEMS)for m in METRICS})
            aggregate.append(r);rr.append(r)
        row=dict(arm=arm,n_seed_aggregates=3,n_systems_per_seed=5,complete_runs=15,ddof=1,variance_axis='evaluation_seed_of_fixed_five_system_aggregate')
        for m in ['SUN_total']+[m+'_macro_mean'for m in METRICS]:row.update({m+'_'+k:z for k,z in stats([r[m]for r in rr]).items()})
        aggregate_stats.append(row)
    agg={(r['arm'],r['seed']):r for r in aggregate}
    contrast_rows=[];contrast_sys=[];contrast_seed=[];contrast_stats=[];contrast_summary={}
    for name,kind,weights,interpretation in CONTRASTS:
        for task in SYSTEMS:
            rr=[]
            for seed in SEEDS:
                row=dict(contrast=name,kind=kind,task_id=task,seed=seed,budget=10)
                for m in METRICS:
                    delta=sum(w*metrics[a,task,seed][m]for a,w in weights.items());row[m+'_delta']=delta;row[m+'_sign']=sign(delta)
                contrast_rows.append(row);rr.append(row)
            row=dict(contrast=name,kind=kind,task_id=task,n_paired_seeds=3,ddof=1)
            for m in METRICS:row.update({m+'_delta_'+k:z for k,z in stats([r[m+'_delta']for r in rr]).items()})
            contrast_sys.append(row)
        rr=[]
        for seed in SEEDS:
            row=dict(contrast=name,kind=kind,seed=seed,n_fixed_systems=5)
            for m in ['SUN_total']+[m+'_macro_mean'for m in METRICS]:row[m+'_delta']=sum(w*agg[a,seed][m]for a,w in weights.items())
            contrast_seed.append(row);rr.append(row)
        row=dict(contrast=name,kind=kind,n_seed_aggregates=3,ddof=1,interpretation=interpretation)
        for m in ['SUN_total']+[m+'_macro_mean'for m in METRICS]:row.update({m+'_delta_'+k:z for k,z in stats([r[m+'_delta']for r in rr]).items()})
        contrast_stats.append(row)
        within=[r for r in contrast_rows if r['contrast']==name];sys=[r for r in contrast_sys if r['contrast']==name]
        contrast_summary[name]=dict(kind=kind,weights=weights,interpretation=interpretation,
            signs_by_system_mean={m:signs([r[m+'_delta_mean']for r in sys])for m in METRICS},
            signs_by_run_pair={m:signs([r[m+'_delta']for r in within])for m in METRICS},
            signs_by_seed_macro={m:signs([r[m+'_macro_mean_delta']for r in rr])for m in METRICS},
            per_seed_SUN_total_delta=[r['SUN_total_delta']for r in rr],
            per_seed_AUDC_macro_delta=[r['AUDC_macro_mean_delta']for r in rr],
            mean_per_run_delta={m:st.mean(r[m+'_delta']for r in within)for m in METRICS},
            all15_SUN_total_delta=sum(r['SUN_delta']for r in within))
    # Weighted2x2 descriptive contrasts must sum exactly to total algorithm contrast.
    for task in SYSTEMS:
        for seed in SEEDS:
            rr={r['contrast']:r for r in contrast_rows if r['task_id']==task and r['seed']==seed}
            for m in METRICS:
                assert math.isclose(rr['average_UQ_algorithm_contrast'][m+'_delta']+rr['average_ES_algorithm_contrast'][m+'_delta'],rr['full_minus_base'][m+'_delta'],abs_tol=1e-12)
    progress={k:sum(a['all1080_arm_progress'][k]for a in v['arms'])for k in ['expected','complete','failed','claimed_no_result','pending']}
    assert progress==dict(expected=1080,complete=279,failed=0,claimed_no_result=10,pending=791)
    arms={}
    for arm in ARMS:
        rr=[r for r in runs if r['arm']==arm]
        arms[arm]=dict(complete_runs=len(rr),SUN_total=sum(r['SUN']for r in rr),mean_per_run={m:st.mean(r[m]for r in rr)for m in METRICS},
            per_seed_SUN_total=[agg[arm,s]['SUN_total']for s in SEEDS],per_seed_AUDC_macro=[agg[arm,s]['AUDC_macro_mean']for s in SEEDS])
    assert [arms[a]['SUN_total']for a in ARMS]==[22,25,17,20]
    assert arms[UQ]['per_seed_SUN_total']==[10,11,4]
    for x,y in zip(arms[UQ]['per_seed_AUDC_macro'],[.224,.238,.116]):assert math.isclose(x,y,abs_tol=1e-12)
    summary=dict(schema='complete_core5_four_arm_three_seed_ablation_summary_v1',observed_CST=v['observation_finished_CST'],
        model=v['model'],systems=SYSTEMS,seeds=SEEDS,budget=10,complete_runs=60,planned_core_runs=60,
        all_new_study=progress,core_is_subset_of_1080=True,core4arm_complete=True,full1080_complete=False,
        completed_core_candidate_calls=sum(r['candidate_oracle_attempts']for r in costs),curves=len(curves),
        arms=arms,contrasts=contrast_summary,variance_ddof=1,
        variance_is_evaluation_seed_variance_not_training_replication=True,contrast_is_algorithm_level_not_fixed_weight_causal_effect=True,
        NN_contribution_separately_isolated=False,new_model_or_oracle_calls=0)
    validation=dict(passed=True,complete_cells=60,all_cells_have_original_receipt_and_result_hashes=True,
        curves_recomputed=60,curve_points=660,variance_independent_pairwise_identity_passed=True,
        algorithm_contrast_sum_identity_passed=True,prior_56_completed_rows_unchanged=v['previous_56_completed_metrics_curves_costs_and_hashes_unchanged'],
        full_remote_acceptance_reexecuted=False,new_model_or_oracle_calls=0)
    return {'runs.csv':csv_bytes(runs),'curves.csv':csv_bytes(curves),'timings_and_costs.csv':csv_bytes(costs),'cost_summary.csv':csv_bytes(cost_summary),
        'per_system_seed_statistics.csv':csv_bytes(systems),'five_system_aggregates_by_seed.csv':csv_bytes(aggregate),
        'aggregate_seed_statistics.csv':csv_bytes(aggregate_stats),'contrasts_by_run.csv':csv_bytes(contrast_rows),
        'contrast_system_statistics.csv':csv_bytes(contrast_sys),'contrast_aggregates_by_seed.csv':csv_bytes(contrast_seed),
        'contrast_aggregate_statistics.csv':csv_bytes(contrast_stats),'summary.json':json_bytes(summary),'validation.json':json_bytes(validation)}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    provenance=json.loads((D/'provenance.json').read_text());raw=(D/'study_inputs.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==provenance['scientific_projection_sha256']
    outputs=compute(json.loads(raw))
    for name,raw in outputs.items():
        if args.write:(D/name).write_bytes(raw)
        else:assert (D/name).read_bytes()==raw,'Derived output mismatch: '+name
    if not args.write:
        manifest=json.loads((D/'manifest.json').read_text())
        for item in manifest['files']:
            p=D/item['path'];assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    summary=json.loads(outputs['summary.json'])
    print(json.dumps(dict(passed=True,core_complete=60,total_study_complete=279,SUN={a:x['SUN_total']for a,x in summary['arms'].items()},AUDC={a:x['mean_per_run']['AUDC']for a,x in summary['arms'].items()},new_scientific_calls=0),sort_keys=True))

if __name__=='__main__':main()
