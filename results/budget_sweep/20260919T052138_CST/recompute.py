"""Recompute the exported curves and matched descriptive statistics; stdlib only."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,math,statistics
ROOT=Path(__file__).resolve().parent
METHODS=('baseline','esopt_graph_risk')
def read(name): return json.loads((ROOT/name).read_text())
def table(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))
def near(a,b):
    return a in ('',None) if b is None else math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)
def desc(xs):
    return {'n':len(xs),'mean':statistics.mean(xs) if xs else None,'sample_variance':statistics.variance(xs) if len(xs)>1 else None,'SD':statistics.stdev(xs) if len(xs)>1 else None}
def main():
    s=read('summary.json');records=table('trajectories.csv');matrix=table('all_system_results.csv');curves=defaultdict(list)
    assert len(records)==len({r['job_id'] for r in records})==s['complete_trajectories']
    assert len(matrix)==len({(r['budget'],r['task_id'],r['method'],r['seed']) for r in matrix})==180
    assert s['available_evaluation_seeds']==[1] and s['run_to_run_sample_variance'] is None and s['run_to_run_SD'] is None
    for r in table('curves.csv'):curves[r['job_id']].append((int(r['candidate_attempt']),int(r['SUN'])))
    assert set(curves)=={r['job_id'] for r in records}
    by={}
    for r in records:
        b=int(r['budget']);p=curves[r['job_id']];assert r['seed']=='1' and r['method'] in METHODS
        assert r['benchmark']=='made' and r['model_key']=='qwen35_4b' and r['complete']=='True'
        assert [x for x,y in p]==list(range(b+1)) and p[0]==(0,0) and all(0<=y<=x for x,y in p)
        area=sum((x1-x0)*(y1+y0) for (x0,y0),(x1,y1) in zip(p,p[1:]))/b**2
        assert near(r['AUDC'],area) and near(r['SUN'],p[-1][1]) and near(r['mSUN'],p[-1][1]/b)
        assert int(r['candidate_oracle_attempts'])==b
        for k in ('raw_result_sha256','raw_RPC_sha256','scope_registration_sha256','acceptance_evidence_sha256'):
            assert len(r[k])==64 and all(c in '0123456789abcdef' for c in r[k])
        key=(b,r['task_id'],r['method']);assert key not in by;by[key]=r
    for r in matrix:
        key=(int(r['budget']),r['task_id'],r['method'])
        if r['status'] in ('complete','complete_derived_acceptance'):
            a=by[key];assert r['job_id']==a['job_id']
            for k in ('SUN','mSUN','AUDC'):assert near(r[k],float(a[k]))
        else:assert key not in by and all(r[k]=='' for k in ('SUN','mSUN','AUDC','raw_result_sha256'))
    expected_stats={};expected_pairs={}
    for b in (10,30,50):
        tasks=sorted({t for bb,t,m in by if bb==b});paired=[t for t in tasks if all((b,t,m) in by for m in METHODS)]
        stated=s['budgets'][str(b)];assert len(paired)==stated['complete_pairs'] and paired==stated['paired_system_ids']
        assert sum(bb==b for bb,t,m in by)==stated['completed_trajectories']
        for metric in ('SUN','mSUN','AUDC'):
            a=[float(by[b,t,'baseline'][metric]) for t in paired];z=[float(by[b,t,'esopt_graph_risk'][metric]) for t in paired];delta=[y-x for x,y in zip(a,z)]
            x=stated[metric];assert near(x['baseline_mean'],statistics.mean(a)) and near(x['full_mean'],statistics.mean(z)) and near(x['full_minus_baseline'],statistics.mean(delta))
            assert (x['wins'],x['ties'],x['losses'])==(sum(d>1e-12 for d in delta),sum(abs(d)<=1e-12 for d in delta),sum(d< -1e-12 for d in delta))
            for cohort in ('all_completed','matched_systems'):
                for method in METHODS:
                    vals=[float(r[metric]) for (bb,t,m),r in by.items() if bb==b and m==method] if cohort=='all_completed' else a if method=='baseline' else z
                    expected_stats[b,cohort,method,metric]=desc(vals)
            expected_stats[b,'paired_full_minus_baseline','paired_delta',metric]=desc(delta)
        for t in paired:expected_pairs[b,t]=True
    stats=table('statistics.csv');assert len(stats)==len(expected_stats)
    for r in stats:
        key=(int(r['budget']),r['cohort'],r['method'],r['metric']);assert r['variance_ddof']=='1'
        assert r['dispersion_axis']=='chemical_system_at_seed1_not_repeated_seeds'
        for k,v in expected_stats.pop(key).items():assert near(r[k],v),(key,k)
    assert not expected_stats
    pairs=table('paired_systems.csv');assert len(pairs)==len(expected_pairs)
    for r in pairs:
        b,t=int(r['budget']),r['task_id'];expected_pairs.pop((b,t))
        for metric in ('SUN','mSUN','AUDC'):
            a=float(by[b,t,'baseline'][metric]);z=float(by[b,t,'esopt_graph_risk'][metric])
            assert near(r['baseline_'+metric],a) and near(r['full_'+metric],z) and near(r['delta_'+metric],z-a)
    assert not expected_pairs
    ga=read('ga_pt_tm_reconciliation.json');r=by[50,'Ga-Pt-Tm','baseline']
    assert ga['originally_accepted'] is False and ga['accepted_for_combined_reporting'] is True
    assert r['derived_acceptance']=='True' and r['original_runner_accepted']=='False' and r['acceptance_evidence_kind']=='independent_derived_acceptance'
    assert r['raw_result_sha256']==ga['derived_envelope']['sha256'] and r['receipt_sha256']==ga['derived_receipt']['sha256'] and r['root_adoption_sha256']==ga['root_adoption']['sha256']
    assert curves[r['job_id']][25:27]==[(25,9),(26,7)] and near(r['AUDC'],.3664) and near(r['SUN'],18)
    assert sum(r['derived_acceptance']=='True' for r in records)==1
    coverage=table('seed_coverage.csv');assert len(coverage)==180
    assert all(r['available_seeds']=='1' and r['run_to_run_sample_variance']==r['run_to_run_SD']=='' for r in coverage)
    costs=table('timings_and_costs.csv');assert len(costs)==len(records)
    candidate=sum(int(r['candidate_oracle_attempts']) for r in costs)
    assert candidate==sum(int(r['candidate_oracle_attempts']) for r in records)==s['completed_trajectory_candidate_attempts']
    expected_cost={}
    for r in table('cost_summary.csv'):
        rows=[v for v in costs if v['budget']==r['budget'] and v['method']==r['method']]
        assert int(r['complete_trajectories'])==len(rows)
        for k,v in r.items():
            if k in ('budget','method','complete_trajectories'):continue
            expected=sum(float(x[k]) for x in rows) if all(x[k]!='' for x in rows) else None
            assert near(v,expected),(r['budget'],r['method'],k)
    checked=0
    if (ROOT/'manifest.json').exists():
        for r in read('manifest.json')['files']:
            p=ROOT/r['path'];assert p.parent==ROOT and p.is_file() and not p.is_symlink()
            data=p.read_bytes();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['export_sha256'];checked+=1
    print(json.dumps({'passed':True,'complete_trajectories':len(records),'complete_pairs':{b:s['budgets'][b]['complete_pairs'] for b in ('10','30','50')},'curve_points':sum(len(x) for x in curves.values()),'candidate_oracle_attempts':candidate,'derived_acceptance_trajectories':1,'run_to_run_variance':'unavailable_seed1_only','manifest_files_verified':checked,'scientific_calls_to_recompute_export':0}))
if __name__=='__main__':main()
