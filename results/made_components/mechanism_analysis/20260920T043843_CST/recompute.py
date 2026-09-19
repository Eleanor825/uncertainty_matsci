"""Recompute a fixed posthoc held-out mechanism report; standard library only."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, gzip, hashlib, io, json, math, statistics

HERE=Path(__file__).resolve().parent
INPUT_SHA='13f29ed763b8814fc588d5bc005408ed81e991c83a2a34f4958dd6284570c8e7'
JSON_SHA='0b58bd2507d536c5a6a20d713bfa87da915f4fa55bbd1acf629bb33c269380ec'
COHORT_SHA='b33fecfdbc1559378ff70b4ac796e92a59b4f790cff60c2fe5824918a42cfbe5'
ARMS=('baseline_reference','full_support_aware');SEEDS=(2,3,4)

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def near(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
def stats(v):
    return dict(n=len(v),mean=statistics.mean(v) if v else None,
        sample_variance=statistics.variance(v) if len(v)>1 else None,
        SD=statistics.stdev(v) if len(v)>1 else None)
def sign(x):return 'positive' if x>1e-12 else 'negative' if x< -1e-12 else 'tie'
def num(x):return type(x) in (int,float) and math.isfinite(x)
def isprob(x):return num(x) and 0<=x<=1
def islabel(x):return type(x) in (int,bool) and x in (0,1)
def csvbytes(rows):
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
    w.writerows({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    return out.getvalue().encode()
def jsonbytes(x):return (json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()

def scores(points,equal=False):
    # Independent pairwise weighted AUROC implementation for numerical checking.
    # All points are observed predictions/labels, never a fit or imputation.
    counts=Counter(k for p,y,k in points)
    x=[(float(p),int(y),1/counts[k] if equal else 1.) for p,y,k in points]
    n=math.fsum(w for p,y,w in x);pos=[(p,w) for p,y,w in x if y];neg=[(p,w) for p,y,w in x if not y]
    wp=math.fsum(w for p,w in pos);wn=math.fsum(w for p,w in neg)
    auc=math.fsum(a*b*(int(p>q)+.5*int(p==q)) for p,a in pos for q,b in neg)/(wp*wn) if wp and wn else None
    bins=[]
    for i in range(10):
        z=[(p,y,w) for p,y,w in x if min(int(p*10),9)==i];mass=math.fsum(w for p,y,w in z)
        bins.append(dict(bin=i,n=len(z),weight=mass,prediction=math.fsum(p*w for p,y,w in z)/mass if mass else None,
            observed_failure=math.fsum(y*w for p,y,w in z)/mass if mass else None))
    metrics=dict(n=len(x),unique_outcomes=len(counts),weight=n,
        minimum=min((p for p,y,w in x),default=None),maximum=max((p for p,y,w in x),default=None),
        mean_prediction=math.fsum(p*w for p,y,w in x)/n if n else None,failure_fraction=wp/n if n else None,
        Brier=math.fsum(w*(p-y)**2 for p,y,w in x)/n if n else None,AUROC=auc,
        ECE10=math.fsum(b['weight']*abs(b['prediction']-b['observed_failure']) for b in bins if b['weight'])/n if n else None)
    return metrics,bins

def build():
    raw=(HERE/'observations.json.gz').read_bytes();assert sha(raw)==INPUT_SHA
    unpacked=gzip.decompress(raw);assert sha(unpacked)==JSON_SHA
    for private in (b'/mnt/',b'/root/',b'/Users/',b'zhanghuan',b'GPU-',b'"hostname"',b'"argv"',b'"password"'):
        assert private not in unpacked
    v=json.loads(unpacked);craw=(HERE/'cohort.json').read_bytes();assert sha(craw)==COHORT_SHA;c=json.loads(craw)
    assert v['complete'] and v['captured_runs']==144 and v['captured_steps']==1440 and not v['gaps']
    assert v['new_LLM_graph_oracle_calls']==v['new_fits']==0 and not v['original_files_modified']
    assert {(r['job']['task_id'],r['job']['arm'],r['job']['seed']) for r in v['trajectories']}=={(t,a,s) for t in c['systems'] for a in ARMS for s in SEEDS}
    assert len(c['systems'])==24 and len(c['coverage_all_30'])==30
    cohort={r['job_id']:r for r in c['jobs']};lookup={};tr=[];steps=[];sequences=[];calgroups=defaultdict(list);checks=0
    for r in v['trajectories']:
        j=r['job'];jid=j['job_id'];source=cohort[jid];lookup[j['task_id'],j['arm'],j['seed']]=r
        for k in ('SUN','AUDC','seed','task_id','arm','budget'):assert j[k]==source[k];checks+=1
        assert j['budget']==len(r['step_outcomes'])==10
        curve=[(0,0)]+[(s['step'],s['SUN_after']) for s in r['step_outcomes']]
        area=math.fsum((x1-x0)*(y1+y0) for (x0,y0),(x1,y1) in zip(curve,curve[1:]))/100
        assert curve[-1][1]==j['SUN'] and near(area,j['AUDC']);checks+=2
        for s in r['step_outcomes']:
            assert s['official_num_newly_discovered_stable']==s['SUN_after'] and s['official_queries_used']==s['step'];checks+=2
            assert s['SUN_delta']==s['SUN_after']-s['SUN_before'];checks+=1
            if s['ok'] and type(s['event_stable_and_new']) is bool:
                assert s['event_stable_new_minus_SUN_delta']==int(s['event_stable_and_new'])-s['SUN_delta'];checks+=1
            steps.append(dict(job_id=jid,task_id=j['task_id'],arm=j['arm'],seed=j['seed'],**s))
        points=[dict(zip(v['risk_point_columns'],a)) for a in r['risk_rows']]
        assert len(points)==r['counts']['proposals'];checks+=1
        assert sum(d['disposition']=='executed' for d in points)==r['counts']['executed_policy_tools'];checks+=1
        assert sum(not d['generation_success'] for d in points)==r['counts']['generation_invalid'];checks+=1
        assert sum(d['supported'] is True for d in points)==r['counts']['support_true'];checks+=1
        assert sum(d['supported'] is False for d in points)==r['counts']['support_false'];checks+=1
        grouped=defaultdict(list)
        for d in points:
            grouped[d['sequence']].append(d)
            if d['disposition']!='executed':assert d['label_future_failure'] is None and d['next_scientific_rpc_id'] is None;checks+=1
            sup='supported' if d['supported'] is True else 'unsupported' if d['supported'] is False else 'support_missing'
            if d['disposition']=='executed' and d['generation_success'] and isprob(d['raw_scalar']) and islabel(d['label_future_failure']) and d['next_scientific_rpc_id'] is not None:
                point=(d['raw_scalar'],d['label_future_failure'],(jid,d['next_scientific_rpc_id']))
                key=(j['arm'],'scalar',d['scalar_origin'],sup,'all_executed_next_outcome_associations')
                calgroups[key].append(point)
                if d['direct_material_step_rpc_id'] is not None:calgroups[(j['arm'],'scalar',d['scalar_origin'],sup,'direct_selected_material')].append(point)
            if d['disposition']=='executed' and d['generation_success']:
                for i,h in enumerate(v['type_head_order']):
                    p=d['diagnostic_types'][i];y=d['observed_types'][i];ids=d['type_observation_rpc_ids'][i]
                    if isprob(p) and islabel(y):
                        calgroups[(j['arm'],'type',h,sup,'original_type_label_horizon')].append((p,y,(jid,h,tuple(ids) if ids else d['local_decision_id'])))
        assert sum(max(0,len(g)-1) for g in grouped.values())==r['counts']['actual_extra_proposals'];checks+=1
        nn=[d['raw_scalar'] for d in points if d['generation_success'] and isprob(d['raw_scalar'])]
        cstats={k:r['counts'][k] for k in r['counts']}
        tr.append(dict(j,**cstats,scalar_n=len(nn),scalar_min=min(nn,default=None),scalar_max=max(nn,default=None),
            event_minus_final=r['event_stable_new_minus_final_SUN'],formula_top_fraction=r['evaluated_formulas']['top_fraction'],
            unique_evaluated_formulas=r['evaluated_formulas']['unique'],
            wall_seconds=r['costs'].get('wall_seconds'),graph_seconds=r['costs'].get('graph_seconds'),
            initialization_calls=r['costs'].get('initialization_oracle_attempts'),
            MACE_events=sum(e['recorded_events'] for k,e in r['oracle_events'].items() if k.startswith('mace|')),
            MACE_cost_field=r['costs'].get('surrogate_oracle_attempts'),tool_counts=r['tool_counts'],
            actual_model_state_hash=r['profile']['actual_model_state_hash']))
        for seq in r['intervention_sequences']:
            assert len(seq['candidates'])==len(grouped[seq['sequence']]);checks+=1
            for d in seq['candidates']:
                if d['disposition']!='executed':assert d['future_failure'] is None;checks+=1
            sequences.append(dict(job_id=jid,task_id=j['task_id'],arm=j['arm'],seed=j['seed'],**seq))
    chemistry=[];paired=[];system_means=[];macros=[]
    for t in c['systems']:
        means={}
        for arm in ARMS:
            rr=[lookup[t,arm,s] for s in SEEDS]
            for metric in ('SUN','AUDC'):
                a=stats([r['job'][metric] for r in rr]);means[arm,metric]=a['mean'];chemistry.append(dict(task_id=t,arm=arm,metric=metric,**a))
        diffs={}
        for s in SEEDS:
            b,f=[lookup[t,a,s] for a in ARMS]
            row=dict(task_id=t,seed=s,baseline_SUN=b['job']['SUN'],full_SUN=f['job']['SUN'],baseline_AUDC=b['job']['AUDC'],full_AUDC=f['job']['AUDC'],
                SUN_difference=f['job']['SUN']-b['job']['SUN'],AUDC_difference=f['job']['AUDC']-b['job']['AUDC'])
            paired.append(row)
        for metric in ('SUN','AUDC'):diffs[metric]=stats([r[metric+'_difference'] for r in paired if r['task_id']==t])
        system_means.append(dict(task_id=t,baseline_SUN=means[ARMS[0],'SUN'],full_SUN=means[ARMS[1],'SUN'],
            baseline_AUDC=means[ARMS[0],'AUDC'],full_AUDC=means[ARMS[1],'AUDC'],
            SUN_difference=diffs['SUN']['mean'],AUDC_difference=diffs['AUDC']['mean'],
            SUN_difference_variance=diffs['SUN']['sample_variance'],AUDC_difference_variance=diffs['AUDC']['sample_variance']))
    for seed in SEEDS:
        for arm in ARMS:
            rr=[lookup[t,arm,seed] for t in c['systems']]
            macros.append(dict(arm=arm,seed=seed,systems=24,SUN_total=sum(r['job']['SUN'] for r in rr),
                SUN_macro=statistics.mean(r['job']['SUN'] for r in rr),AUDC_macro=statistics.mean(r['job']['AUDC'] for r in rr)))
    perarm=[];toolrows=[];errorrows=[]
    for arm in ARMS:
        rr=[r for r in v['trajectories'] if r['job']['arm']==arm];tool=Counter();errs=Counter();counts=Counter()
        for r in rr:
            tool.update(r['tool_counts']);counts.update(r['counts']);errs.update(x['error']['message_prefix'] for x in r['tool_failures'] if x['error'])
        mr=[r for r in macros if r['arm']==arm]
        perarm.append(dict(arm=arm,runs=72,SUN_sum=sum(r['job']['SUN'] for r in rr),
            AUDC_macro=statistics.mean(r['job']['AUDC'] for r in rr),
            seed_SUN_total_statistics=stats([r['SUN_total'] for r in mr]),seed_AUDC_macro_statistics=stats([r['AUDC_macro'] for r in mr]),
            counts=dict(counts),wall_seconds_sum=sum(r['costs']['wall_seconds'] for r in rr),
            graph_seconds_sum=sum(r['costs']['graph_seconds'] for r in rr),
            initialized_ORB=sum(r['costs']['initialization_oracle_attempts'] for r in rr),
            MACE_events=sum(e['recorded_events'] for r in rr for k,e in r['oracle_events'].items() if k.startswith('mace|')),
            MACE_cost_field_missing=sum('surrogate_oracle_attempts' not in r['costs'] for r in rr)))
        toolrows.extend(dict(arm=arm,tool=k,calls=n) for k,n in sorted(tool.items()))
        errorrows.extend(dict(arm=arm,message=k,count=n) for k,n in sorted(errs.items()))
    calibration=[];bins=[]
    for key,points in sorted(calgroups.items()):
        for eq in (False,True):
            metrics,bb=scores(points,eq);ident=dict(zip(('arm','signal','origin_or_head','support','horizon'),key));ident['weighting']='equal_observed_outcomes' if eq else 'per_decision'
            calibration.append(dict(ident,**metrics));bins.extend(dict(ident,**b) for b in bb)
    outcome_groups=[]
    for outcome_sign in ('positive','tie','negative'):
        ts={r['task_id'] for r in system_means if sign(r['AUDC_difference'])==outcome_sign}
        for arm in ARMS:
            rr=[r for r in tr if r['arm']==arm and r['task_id'] in ts]
            outcome_groups.append(dict(AUDC_system_sign=outcome_sign,arm=arm,systems=len(ts),runs=len(rr),
                SUN_sum=sum(r['SUN'] for r in rr),AUDC_mean=statistics.mean(r['AUDC'] for r in rr),
                support_true=sum(r['support_true'] for r in rr),proposals=sum(r['proposals'] for r in rr),
                scalar_crossings=sum(r['supported_scalar_threshold_crossings'] for r in rr),
                type_extra_proposals=sum(r['type_extra_proposals'] for r in rr),schema_extra_proposals=sum(r['schema_extra_proposals'] for r in rr),
                risk_selection_changes=sum(r['supported_valid_pair_selection_changes'] for r in rr),
                failed_selects=sum(r['failed_selects'] for r in rr)))
    worst=min(system_means,key=lambda r:(r['AUDC_difference'],r['SUN_difference'],r['task_id']))['task_id']
    illustrations=[r for r in v['trajectories'] if r['job']['task_id'] in ('Ag-Nd-Pd-Pt-Tb',worst)]
    signs={m:dict(Counter(sign(r[m+'_difference']) for r in system_means)) for m in ('SUN','AUDC')}
    assert signs==v['comparisons']['system_mean_sign_counts'];checks+=1
    summary=dict(schema='MADE_144_complete_B10_posthoc_mechanism_report_v1',runs=144,steps=1440,systems=24,seeds=list(SEEDS),
        source_snapshot=v['source_snapshot'],posthoc=True,scope_is_not_all_1080=True,
        original_raw_JSON_SHA=JSON_SHA,original_gzip_SHA=INPUT_SHA,derived_metrics_verified=checks,
        per_arm=perarm,system_sign_counts=signs,illustration_tasks=['Ag-Nd-Pd-Pt-Tb',worst],
        new_fits=0,new_model_graph_oracle_calls=0,unexecuted_outcomes_imputed=False,
        case_differences_are_not_causal_effects=True,prospective_new_NN_training_inputs_changed=False)
    outputs={name:csvbytes(rows) for name,rows in [('trajectories.csv',tr),('chemistry_statistics.csv',chemistry),
        ('system_means.csv',system_means),('matched_seed_differences.csv',paired),('per_seed_macros.csv',macros),
        ('calibration.csv',calibration),('calibration_bins.csv',bins),('mechanism_by_outcome_sign.csv',outcome_groups),
        ('tools.csv',toolrows),('tool_errors.csv',errorrows),('step_outcomes.csv',steps)]}
    outputs.update({'summary.json':jsonbytes(summary),'intervention_sequences.json':jsonbytes(sequences),
        'illustrations.json':jsonbytes([dict(job=r['job'],counts=r['counts'],evaluated_formulas=r['evaluated_formulas'],
            step_outcomes=r['step_outcomes'],intervention_sequences=r['intervention_sequences'],tool_failures=r['tool_failures']) for r in illustrations])})
    return outputs,summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    outputs,summary=build()
    for name,raw in outputs.items():
        path=HERE/name
        if args.write:path.write_bytes(raw)
        else:assert path.read_bytes()==raw,name
    if not args.write:
        for item in json.loads((HERE/'manifest.json').read_text())['files']:
            raw=(HERE/item['path']).read_bytes();assert len(raw)==item['bytes'] and sha(raw)==item['sha256'],item['path']
    print(canonical({'passed':True,'runs':144,'steps':1440,'checks':summary['derived_metrics_verified'],
        'new_scientific_calls':0,'per_arm':[(r['arm'],r['SUN_sum'],r['AUDC_macro']) for r in summary['per_arm']]}))

if __name__=='__main__':main()
