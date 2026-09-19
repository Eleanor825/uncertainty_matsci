"""Portable metric validation. Reads only exported scientific observations."""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, io, json, math

ROOT = Path(__file__).resolve().parent
TASKS = ('Al-Li-V', 'Al-V-Zn', 'Au-K-Tb', 'Co-Dy-W', 'Co-Mg-Na')
ARMS = ('uq_esopt', 'uq_only', 'qwen_base', 'es_only')
STATES = {'accepted', 'failed', 'claimed_pending', 'unclaimed',
          'original_result_present_CPU_audit_pending', 'CPU_audit_failed',
          'collection_result_pending_separate_audit'}


def sha(b): return hashlib.sha256(b).hexdigest()
def digest(v): return sha(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
def jsonbytes(v): return (json.dumps(v, indent=2, allow_nan=False) + '\n').encode()
def near(a, b): return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12)


def stats(values):
    m = math.fsum(values) / len(values) if values else None
    v = math.fsum((x-m)**2 for x in values)/(len(values)-1) if len(values) > 1 else None
    return {'n': len(values), 'mean': m, 'sample_variance_ddof1': v,
            'SD': math.sqrt(v) if v is not None else None}


def csvbytes(rows, fields):
    out = io.StringIO(newline='')
    w = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    w.writeheader(); w.writerows(rows)
    return out.getvalue().encode()


def validate_common(s):
    assert s['fingerprint'] == digest({k: v for k, v in s.items() if k != 'fingerprint'})
    assert s['new_scientific_calls_for_export'] == 0
    assert len({r['job']['job_id'] for r in s['rows']}) == len(s['rows'])
    for row in s['rows']:
        assert row['status'] in STATES
        if row['status'] == 'accepted':
            assert row['accepted']['audit']['passed'] is True and row['failure_sha256'] is None
            assert row['accepted']['result_source']['sha256'] == row['completed_source_sha256']
            assert row['accepted']['source_job_id'] == row['job']['job_id']
        else:
            assert row['accepted'] is None


def build_nine(s):
    validate_common(s); rows = s['rows']; assert len(rows) == 22
    tests = [r for r in rows if r['job']['stage'] == 'final_eval']
    collects = [r for r in rows if r['job']['stage'] == 'collection']
    assert len(tests) == 15 and len(collects) == 7
    assert {(r['job']['task_id'], r['job']['seed']) for r in tests} == {(t, z) for t in TASKS for z in (2,3,4)}
    assert all(r['job']['budget'] == 10 and r['job']['split'] == 'test' for r in tests)
    assert all(r['job']['budget'] == 50 and r['job']['split'] in ('train','dev') for r in collects)
    matrix = []; curves = []; costs = []; values = {}; statrows = []
    for row in rows:
        j = row['job']; p = row['accepted']
        metrics = {'SUN': None, 'AUDC': None, 'mSUN': None}
        if p:
            assert j['stage'] == 'final_eval'
            ep = p['episode']; model = p['model']; b = j['budget']
            assert ep['task_id'] == j['task_id'] and ep['seed'] == ep['environment_seed'] == j['seed']
            assert ep['complete'] and ep['status'] == 'succeeded' and ep['method'] == 'baseline'
            assert model['id'] == 'Qwen/Qwen3.5-9B' and model['selected_generation'] == 0 and model['NN_or_ES_applied'] is False
            assert model['revision'] == 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
            cv = ep['discovery_curve']
            assert [x for x,y in cv] == list(range(b+1)) and cv[0] == [0,0]
            assert all(type(y) is int and 0 <= y <= x for x,y in cv)
            audc = math.fsum((x1-x0)*(y1+y0) for (x0,y0),(x1,y1) in zip(cv,cv[1:]))/b**2
            sun = cv[-1][1]
            assert near(ep['metrics']['AUDC'],audc) and near(ep['metrics']['mSUN'],sun/b)
            assert ep['official_metrics']['num_newly_discovered_stable'] == sun
            assert ep['official_metrics']['queries_used'] == ep['costs']['candidate_oracle_attempts'] == b
            metrics = {'SUN':sun,'AUDC':audc,'mSUN':sun/b}
            values[j['task_id'],j['seed']] = metrics
            curves.extend({'job_id':j['job_id'],'candidate_attempt':x,'SUN':y} for x,y in cv)
            costs.extend({'job_id':j['job_id'],'quantity':key,'value':value} for key,value in ep['costs'].items())
        matrix.append({**j,'status':row['status'],**metrics,
                       'result_sha256':row['completed_source_sha256'],'failure_sha256':row['failure_sha256']})
    for task in TASKS:
        seeds = [z for z in (2,3,4) if (task,z) in values]
        for metric in ('SUN','AUDC','mSUN'):
            statrows.append({'task_id':task,'metric':metric,'evaluation_seeds':';'.join(map(str,seeds)),
                'complete_three_seeds':len(seeds)==3,**stats([values[task,z][metric] for z in seeds])})
    macro = []
    for z in (2,3,4):
        if all((t,z) in values for t in TASKS):
            macro.append({'seed':z,'total_SUN':sum(values[t,z]['SUN'] for t in TASKS),
                          'macro_AUDC':math.fsum(values[t,z]['AUDC'] for t in TASKS)/5})
    n = len(values)
    summary = {'schema':'Qwen35_9B_stage1_accepted_results_v1','observed_CST':s['observed_CST'],
        'baseline_test_planned':15,'baseline_test_accepted':n,'collection_planned':7,
        'collection_status_counts':dict(Counter(r['status'] for r in collects)),
        'all_status_counts':dict(Counter(r['status'] for r in rows)),
        'stage1_complete':False,'all_baseline_test_complete':n==15,
        'accepted_test_candidate_calls':10*n,
        'accepted_test_initialization_calls':sum(r['accepted']['episode']['costs']['initialization_oracle_attempts'] for r in tests if r['accepted']),
        'complete_five_system_seed_blocks':macro,
        'macro_statistics':{m:stats([r[m] for r in macro]) for m in ('total_SUN','macro_AUDC')},
        'variance_axis':'evaluation seeds within the same chemistry; ddof1; n<2 unavailable; no training repeats',
        'method':'Unmodified theta0 baseline; no NN controller or ES applied',
        'scope_is_separate_from_4B_1080_and_seed1_180':True,'new_scientific_calls_for_export':0}
    report = ['# Qwen3.5-9B stage1: accepted baseline results','',
        f"Observed **{s['observed_CST']}**. **{n}/15** B10 test trajectories are accepted. The seven B50 train/dev collections are separate and are excluded from test statistics.",'',
        'This model is Qwen/Qwen3.5-9B at the pinned revision. These are unchanged theta0 baseline evaluations: no neural uncertainty controller or ES update is applied. They are separate from the 4B 1080-run experiment and the earlier seed1 180-run study.','',
        '| Chemistry | Seed | SUN | AUDC | Candidate calls | Initialization calls | Episode wall seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in tests:
        if row['accepted']:
            j=row['job']; ep=row['accepted']['episode']; v=values[j['task_id'],j['seed']]; c=ep['costs']
            report.append(f"| {j['task_id']} | {j['seed']} | {v['SUN']} | {v['AUDC']:.8g} | 10 | {c['initialization_oracle_attempts']} | {c['wall_seconds']:.3f} |")
    report += ['', 'Pending/failed rows have blank metrics in `all_jobs.csv`. Sample variance and SD are computed only across completed evaluation seeds of the same chemistry; n<2 is blank, and n<3 is explicitly partial. Five-system macro statistics are emitted only for fully completed five-system seed blocks. An early result is not a reliable model-scale comparison or evidence of an uncertainty/ES benefit.','',
        'AUDC is the normalized trapezoidal discovery-curve area, `sum((x1-x0)*(SUN1+SUN0))/B²`. Candidate calls, initialization calls and tool attempts are reported separately. The final SUN is official end-state SUN; dynamic hull reclassification may lower intermediate values.','',
        'Acceptance re-executed the original source-bound per-job CPU audit, including raw file inventory, actual RPC/physics labels, generation prefixes and unchanged policy identity. It hashes checkpoint files but does not instantiate a model, call the oracle or replay an experiment. Exported episodes retain original result/profile/episode SHA references; they are scientific projections rather than copies of private runtime metadata.','',
        'Run `python3 -B recompute.py` to verify this portable projection and all derived statistics. This does not replace the original raw-evidence audit or reproduce a model run. Snapshots overlap; do not add their counts.','']
    return {'all_jobs.csv':csvbytes(matrix,list(matrix[0])),
        'completed_curves.csv':csvbytes(curves,['job_id','candidate_attempt','SUN']),
        'completed_costs.csv':csvbytes(costs,['job_id','quantity','value']),
        'seed_statistics.csv':csvbytes(statrows,list(statrows[0])),
        'summary.json':jsonbytes(summary),'report.md':'\n'.join(report).encode()},summary


def hypervolume(rows):
    pts=[]
    for row in rows:
        y=row['objectives']; sty=float(y['sty']); ef=float(y['e_factor'])
        assert math.isfinite(sty) and math.isfinite(ef) and sty>=0 and 0<=ef<=1000
        pts.append((sty/13000.,(1000.-ef)/1000.))
    area=height=0.
    for x,y in sorted(pts,reverse=True):
        if y>height:area+=x*(y-height);height=y
    return area


def build_snar(s):
    validate_common(s); rows=s['rows']; assert len(rows)==12
    assert {(r['job']['arm'],r['job']['seed']) for r in rows} == {(a,z) for a in ARMS for z in (5201,5202,5203)}
    prior=s['prior_observations']; assert len(prior)==550
    initial=hypervolume(prior); assert near(initial,.8810117795428745)
    matrix=[]; curves=[]; queries=[]; values={}; statrows=[]; pairs=[]
    for row in rows:
        j=row['job'];p=row['accepted'];metrics={k:None for k in ('final_hv','final_hv_gain','mean_querywise_hv','mean_querywise_hv_gain','brier')}
        assert j['queries']==50 and j['job_id']==f"additional_v1_test_{j['arm']}_{j['seed']}"
        if p:
            q=p['queries']; ss=p['summary']; assert len(q)==50 and ss['queries']==50 and ss['prior_rows']==550
            assert p['audit']['weight_hash_consistency']==s['selected_weight_hash']
            cv=[]; history=list(prior); prev=initial
            for i,item in enumerate(q):
                assert item['query_id']==f"{j['job_id']}/q{i:03d}"
                history.append(item); hv=hypervolume(history); cv.append(hv)
                assert near(item['hv_increment'],hv-prev) and item['no_hvi']==int(hv-prev<=1e-12)
                if item['risk'] is not None:assert 0<=item['risk']<=1
                curves.append({'job_id':j['job_id'],'query':i+1,'HV':hv,'HV_gain':hv-initial})
                queries.append({'job_id':j['job_id'],**{k:item[k] for k in ('query_id','kind','risk','proposal_count','invalid_proposals','hv_increment','no_hvi','elapsed_seconds')},
                    'sty':item['objectives']['sty'],'e_factor':item['objectives']['e_factor']})
                prev=hv
            risky=[r for r in q if r['risk'] is not None]
            brier=sum((r['risk']-r['no_hvi'])**2 for r in risky)/len(risky) if risky else None
            metrics={'final_hv':cv[-1],'final_hv_gain':cv[-1]-initial,'mean_querywise_hv':sum(cv)/50,
                     'mean_querywise_hv_gain':sum(v-initial for v in cv)/50,'brier':brier}
            assert near(ss['prior_hv'],initial) and all(near(a,b) for a,b in zip(ss['hv_curve'],cv))
            for k,v in metrics.items():assert (v is None and ss[k] is None) or (v is not None and near(v,ss[k]))
            assert ss['risk_observations']==len(risky)
            assert ss['proposal_count']==sum(x['proposal_count'] for x in q)
            assert ss['invalid_proposals']==sum(x['invalid_proposals'] for x in q)
            values[j['arm'],j['seed']]=metrics
        matrix.append({**j,'status':row['status'],**metrics,'receipt_sha256':row['completed_source_sha256'],'failure_sha256':row['failure_sha256']})
    for a in ARMS:
        for metric in ('final_hv_gain','mean_querywise_hv_gain','brier'):
            seeds=[z for z in (5201,5202,5203) if (a,z) in values and values[a,z][metric] is not None]
            statrows.append({'arm':a,'metric':metric,'evaluation_seeds':';'.join(map(str,seeds)),
                'complete_three_seeds':len(seeds)==3,**stats([values[a,z][metric] for z in seeds])})
    for a in ARMS:
        if a=='qwen_base':continue
        for z in (5201,5202,5203):
            if (a,z) in values and ('qwen_base',z) in values:
                pairs.append({'arm':a,'seed':z,**{m+'_minus_baseline':values[a,z][m]-values['qwen_base',z][m] for m in ('final_hv_gain','mean_querywise_hv_gain')}})
    summary={'schema':'SnAr_additional_three_seed_accepted_results_v1','observed_CST':s['observed_CST'],
        'planned':12,'accepted':len(values),'accepted_physical_queries':50*len(values),'status_counts':dict(Counter(r['status'] for r in rows)),
        'complete':len(values)==12,'seeds':[5201,5202,5203],'prior_rows':550,'prior_HV':initial,
        'selected_generation':0,'selected_weight_hash':s['selected_weight_hash'],
        'separate_from_original35':True,'additional_fitting_or_selection':False,
        'variance_axis':'evaluation seeds of the same arm; ddof1; n<2 unavailable',
        'new_scientific_calls_for_export':0}
    report=['# SnAr: additional fixed-policy evaluation seeds','',
        f"Observed **{s['observed_CST']}**. **{len(values)}/12** additional B50 trajectories have passed the original oracle, trajectory and graph audit. Only these complete trajectories enter statistics.",'',
        'The new scope comprises four arms × seeds5201/5202/5203 ×50 calls (600 planned). These seeds were checked against prior train/dev/evaluation/mutation seeds before execution. This precision extension was requested after the original35 results were known; it is separate from the original five-seed study, not a retrospective claim of a prespecified eight-seed study. No original trajectory, fitting or checkpoint selection is replayed.','',
        'All arms start from the same frozen550-query adaptation archive (HV0.8810117795428745). Model, NN, controller, evaluator and budget remain fixed. Both ES branches previously selected G0: full uses the same weights as UQ-only, and ES-only uses the same weights as baseline. Extra seeds cannot establish an ES improvement or isolate the NN contribution.','',
        '| Arm | Seed | Final HV gain | Mean querywise HV gain | Selected-action Brier |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        j=row['job']
        if row['accepted']:
            v=values[j['arm'],j['seed']];report.append(f"| {j['arm']} | {j['seed']} | {v['final_hv_gain']:.12g} | {v['mean_querywise_hv_gain']:.12g} | {v['brier']} |")
    report+=['','Metrics use two-objective HV, not MADE AUDC. Coordinate transforms are `sty/13000` and `(1000-e_factor)/1000`. Gain subtracts the shared prior HV. Brier uses observed selected proposals only and `no_hvi=1` for non-improvement; candidates never executed have no invented outcomes.','',
        'Per-arm means/sample variance/SD use accepted evaluation seeds only; incomplete sets are labelled and n<2 variance is blank. Paired differences use exact matching new seeds only. Pending, failed and unknown rows stay metric-free. Early uneven arm coverage is not an efficacy comparison.','',
        'Original journal closure, physical query identity, graph gates and actual executed G0 identity were revalidated using the frozen acceptance functions. The portable input contains executed objectives/risks and the shared prior, with original evidence SHA references. `python3 -B recompute.py` recomputes every HV point, label, gain, Brier and statistic without a model or oracle. Snapshots overlap; do not add their counts.','']
    return {'all_jobs.csv':csvbytes(matrix,list(matrix[0])),
        'completed_curves.csv':csvbytes(curves,['job_id','query','HV','HV_gain']),
        'executed_queries.csv':csvbytes(queries,['job_id','query_id','kind','risk','proposal_count','invalid_proposals','hv_increment','no_hvi','elapsed_seconds','sty','e_factor']),
        'seed_statistics.csv':csvbytes(statrows,list(statrows[0])),
        'paired_results.csv':csvbytes(pairs,['arm','seed','final_hv_gain_minus_baseline','mean_querywise_hv_gain_minus_baseline']),
        'summary.json':jsonbytes(summary),'report.md':'\n'.join(report).encode()},summary


def build(s):
    return build_nine(s) if s['scope']=='qwen35_9b_stage1' else build_snar(s)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');args=ap.parse_args()
    s=json.loads((ROOT/'snapshot.json').read_text());outputs,summary=build(s)
    for name,b in outputs.items():
        if args.write:(ROOT/name).write_bytes(b)
        else:assert (ROOT/name).read_bytes()==b,name
    if not args.write:
        for item in json.loads((ROOT/'manifest.json').read_text())['files']:
            p=ROOT/item['path'];b=p.read_bytes();assert len(b)==item['bytes'] and sha(b)==item['sha256']
    print(json.dumps({'passed':True,'scope':s['scope'],'accepted':summary.get('accepted',summary.get('baseline_test_accepted')),
                      'new_scientific_calls':0}))


if __name__=='__main__':main()
