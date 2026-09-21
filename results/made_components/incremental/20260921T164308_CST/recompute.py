"""Pure metric recomputation for one incremental completed-result snapshot."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,io,json,math,argparse

ROOT=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def stats(v):
    m=math.fsum(v)/len(v) if v else None
    variance=math.fsum((x-m)**2 for x in v)/(len(v)-1) if len(v)>1 else None
    return {'n':len(v),'mean':m,'sample_variance_ddof1':variance,'SD':math.sqrt(variance) if variance is not None else None}
def csvbytes(rows):
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return out.getvalue().encode()
def build(s):
    assert s['fingerprint']==digest({k:v for k,v in s.items() if k!='fingerprint'})
    rr=s['rows'];arms=s['arms'];tasks=s['task_ids'];seeds=s['evaluation_seeds'];budgets=s['budgets']
    assert len(rr)==1080 and len(arms)==4 and len(tasks)==30 and seeds==[2,3,4] and budgets==[10,30,50]
    lookup={(r['arm'],r['task_id'],r['budget'],r['seed']):r for r in rr};assert len(lookup)==1080
    matrix=[];curves=[];accepted={};statistics=[];pairrows=[]
    for r in rr:
        ok=r['status']=='accepted'
        assert r['status'] in ('accepted','failed','claimed_pending','reserved_unknown','unclaimed')
        if ok:
            assert r['receipt'] and r['result'] and not r['failure']
            b=r['budget'];cv=r['curve'];assert [p[0] for p in cv]==list(range(b+1)) and cv[0]==[0,0]
            assert all(type(y) is int and 0<=y<=x for x,y in cv)
            audc=math.fsum((x1-x0)*(y1+y0) for (x0,y0),(x1,y1) in zip(cv,cv[1:]))/b**2
            assert r['SUN']==cv[-1][1] and abs(r['AUDC']-audc)<=1e-12 and r['costs']['candidate_oracle_attempts']==b
            accepted[r['arm'],r['task_id'],b,r['seed']]=r
            curves += [{'job_id':r['job_id'],'query':x,'SUN':y} for x,y in cv]
        else:assert all(r[k] is None for k in ('SUN','AUDC','mSUN','curve','costs'))
        matrix.append({k:r[k] for k in ('job_id','task_id','budget','seed','arm','status','SUN','AUDC','mSUN','selected_generation','model_state_hash')}|{
            'result_sha256':(r['result'] or {}).get('sha256'),'receipt_sha256':(r['receipt'] or {}).get('sha256'),'failure_sha256':(r['failure'] or {}).get('sha256')})
    for arm in arms:
        for task in tasks:
            for b in budgets:
                selected=[accepted[arm,task,b,z] for z in seeds if (arm,task,b,z) in accepted]
                for metric in ('SUN','AUDC','mSUN'):
                    statistics.append({'arm':arm,'task_id':task,'budget':b,'metric':metric,'evaluation_seeds':';'.join(str(r['seed']) for r in selected),
                        'complete_three_seeds':len(selected)==3,**stats([r[metric] for r in selected])})
    pairs={}
    for arm in arms:
        if arm=='baseline_reference':continue
        for b in budgets:
            selected=[]
            for task in tasks:
                for seed in seeds:
                    base=accepted.get(('baseline_reference',task,b,seed));other=accepted.get((arm,task,b,seed))
                    if not(base and other):continue
                    p={'arm':arm,'task_id':task,'budget':b,'seed':seed}
                    for m in ('SUN','AUDC'):p.update({m+'_baseline':base[m],m+'_arm':other[m],m+'_delta':other[m]-base[m]})
                    pairrows.append(p);selected.append(p)
            pairs[arm+'@B'+str(b)]={'matched_pairs':len(selected),'means_are_over_current_matching_set_not_full_study':len(selected)!=90}
            for m in ('SUN','AUDC'):
                vals=[r[m+'_delta'] for r in selected]
                pairs[arm+'@B'+str(b)][m]={'baseline_mean':stats([r[m+'_baseline'] for r in selected])['mean'],'arm_mean':stats([r[m+'_arm'] for r in selected])['mean'],
                    'paired_mean_delta':stats(vals)['mean'],'wins':sum(x>1e-12 for x in vals),'ties':sum(abs(x)<=1e-12 for x in vals),'losses':sum(x< -1e-12 for x in vals)}
    count={a:dict(Counter(r['status'] for r in rr if r['arm']==a)) for a in arms}
    summary={'schema':'incremental_completed_MADE_summary_v1','observed_CST':s['observed_CST'],'observed_UTC':s['observed_UTC'],'source_snapshot_fingerprint':s['fingerprint'],
        'planned':1080,'accepted':len(accepted),'counts_by_arm':count,'full_study_complete':len(accepted)==1080,'matched_comparisons':pairs,
        'variance_axis':'evaluation seeds within fixed system/budget/arm; ddof1; n<2 unavailable; n<3 partial; not training repeats',
        'pending_failed_unknown_excluded_from_means':True,'new_scientific_calls_for_export':0,'raw_trajectory_reaudit_performed':False}
    report=['# Incremental MADE completed-result update','',f"Observed **{s['observed_CST']}**. **{len(accepted)}/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.",'',
        '| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |','|---|---:|---:|---:|---:|---:|']
    for a in arms:report.append('| '+a+' | '+' | '.join(str(count[a].get(k,0)) for k in ('accepted','failed','claimed_pending','reserved_unknown','unclaimed'))+' |')
    report += ['','Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.','',
               '| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |','|---|---:|---:|---:|']
    for name,p in pairs.items():report.append(f"| {name} minus baseline | {p['matched_pairs']} | {p['SUN']['paired_mean_delta']} | {p['AUDC']['paired_mean_delta']} |")
    report+=['','These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.','',
             'Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.','',
             'Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.','']
    outputs={'all_system_results.csv':csvbytes(matrix),'completed_curves.csv':csvbytes(curves),'seed_statistics.csv':csvbytes(statistics),'paired_results.csv':csvbytes(pairrows),
             'summary.json':(json.dumps(summary,indent=2,allow_nan=False)+'\n').encode(),'report.md':'\n'.join(report).encode()}
    return outputs,summary
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');args=ap.parse_args();s=json.loads((ROOT/'snapshot.json').read_text());outputs,summary=build(s)
    for name,b in outputs.items():
        if args.write:(ROOT/name).write_bytes(b)
        else:assert (ROOT/name).read_bytes()==b,name
    if not args.write:
        for ref in json.loads((ROOT/'manifest.json').read_text())['files']:
            b=(ROOT/ref['path']).read_bytes();assert sha(b)==ref['sha256'] and len(b)==ref['bytes']
    print(json.dumps({'passed':True,'planned':1080,'accepted':summary['accepted'],'new_science_calls':0}))
if __name__=='__main__':main()
