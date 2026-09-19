"""Queue-capacity scenarios, not experimental results or confidence intervals."""
from pathlib import Path
from collections import Counter
import json,csv,heapq,hashlib,io,math,argparse

D=Path(__file__).resolve().parent
def q(v,p):
    a=sorted(v);x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def csvbytes(rows):
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return out.getvalue().encode()
def compute():
    for ref in json.loads((D/'sources.json').read_text())['files']:assert hashlib.sha256((D/ref['file']).read_bytes()).hexdigest()==ref['sha256']
    s=json.loads((D/'inputs/MADE_360.json').read_text());tim=json.loads((D/'inputs/measured_job_times.json').read_text())
    old=list(csv.DictReader((D/'inputs/old180_timings.csv').open()));oi={(r['method'],r['task_id'],int(r['budget'])):r for r in old}
    arms=s['arms'];tasks=s['task_ids'];core=s['core5_task_ids'];order={t:i for i,t in enumerate(tasks)}
    base=[r for r in s['rows'] if r['status']=='accepted'];eligible=[r for r in s['rows'] if r['status'] in ('unclaimed','claimed_pending')]
    assert len(base)==360 and len(eligible)==697 and sum(r['status']=='failed' for r in s['rows'])==23
    services=[];cases=[];snar=[]
    for scenario,quantile in [('median',.5),('conservative_p75',.75)]:
        cost={}
        for a in arms:
            ten=[r for r in tim['MADE_frozen360'] if r['arm']==a and r['budget']==10]
            family='baseline' if a in ('baseline_reference','es_only_independent') else 'esopt_graph_risk'
            for b in (10,30,50):
                actual=[r for r in tim['MADE_frozen360'] if r['arm']==a and r['budget']==b]
                ratio=None
                if actual:value=q([r['claim_to_result_seconds']/60 for r in actual],quantile)
                else:
                    ratio=q([float(oi[family,t,b]['wall_seconds'])/float(oi[family,t,10]['wall_seconds']) for t in tasks],quantile)
                    value=q([r['rollout_wall_seconds']/60 for r in ten],quantile)*ratio+q([r['initialization_wall_seconds']/60 for r in ten],quantile)+q([r['other_claim_to_result_seconds']/60 for r in ten],quantile)
                cost[a,b]=value;services.append({'scenario':scenario,'arm':a,'budget':b,'minutes':value,'new_observed_n':len(actual),'historical180_ratio_if_extrapolated':ratio})
        completion={};baseline_allocation={'baseline_reference':1,'es_only_independent':1,'uq_only_support_aware':7,'full_support_aware':8}
        # Preserve current scientific jobs: pending claims cost a full remaining
        # trajectory here because their actual step progress was not in the snapshot.
        for a,n in baseline_allocation.items():
            heap=[(30.0,i) for i in range(n)];heapq.heapify(heap)
            queue=sorted([r for r in eligible if r['arm']==a],key=lambda r:(r['status']!='claimed_pending',r['budget'],r['seed'],order[r['task_id']]))
            for r in queue:
                ready,i=heapq.heappop(heap);end=ready+cost[a,r['budget']];heapq.heappush(heap,(end,i));completion[r['job_id']]=end
        # One additional card: the actual registered four blocks, no scope change.
        priority=[]
        for a,group in [('baseline_reference',core[:2]),('es_only_independent',core[:2]),('baseline_reference',core[2:]),('es_only_independent',core[2:])]:
            block=[next(r for r in eligible if r['arm']==a and r['task_id']==t and r['budget']==50 and r['seed']==seed) for t in group for seed in (2,3,4)]
            priority.append((a,block))
        elapsed=0.;priority_ends=[]
        for a,block in priority:
            elapsed+=30.0 # Explicit planning allowance for each of four cold starts.
            for r in block:
                elapsed+=cost[a,50];priority_ends.append((elapsed,r));completion[r['job_id']]=min(completion[r['job_id']],elapsed)
        done=[r for r in eligible if completion[r['job_id']]<=1440];done_ids={r['job_id'] for r in base+done}
        # Neither original B/E card reaches B50 within24h; therefore no double work
        # is credited to priority jobs here. The durable scheduler still owns claims.
        priority_done=[r for end,r in priority_ends if end<=1440]
        b30=sum(r['budget']==30 for r in done);b50=sum(r['budget']==50 for r in done)
        pairs={b:sum(all(next(r for r in s['rows'] if r['arm']==a and r['task_id']==t and r['budget']==b and r['seed']==seed)['job_id'] in done_ids for a in arms) for t in core for seed in (2,3,4)) for b in (30,50)}
        cases.append({'scenario':scenario,'horizon_hours':24,'original_cards_B_U_E_F':[1,7,1,8],'additional_priority_cards':1,
            'priority_blocks':['baseline first2systems ×3seeds','ES first2systems ×3seeds','baseline last3systems ×3seeds','ES last3systems ×3seeds'],
            'new_B30_from360_origin':b30,'new_B50_from360_origin':b50,'new_total_from360_origin':len(done),
            'later_completed2_deducted_from_B30':2,'conservative_new_B30_after_reported362':b30-2,'conservative_new_total_after_reported362':len(done)-2,
            'projected_total_from360_frozen_origin':360+len(done),'priority_new_B50_in24h':len(priority_done),'hours_for_priority30':elapsed/60,
            'core_B30_complete4arm_pairs_out_of15':pairs[30],'core_B50_complete4arm_pairs_out_of15':pairs[50],
            'new_failure_or_node_outage_modelled':False,'old_failed_jobs_replayed':0,'nine_B_model_new_results_credited':0})
        rates={a:q([r['elapsed_seconds']/60 for r in tim['SnAr_old_complete'] if r['kind']=='evaluate' and r['arm']==a],quantile) for a in ('qwen_base','es_only','uq_only','uq_esopt')}
        for count in (6,12,20):
            sequence=(['uq_esopt','qwen_base']*3) if count==6 else list(('uq_esopt','uq_only','qwen_base','es_only'))*(count//4)
            for cards in (1,2,4):
                heap=[(0.,i) for i in range(cards)];heapq.heapify(heap)
                for a in sequence:
                    ready,i=heapq.heappop(heap);heapq.heappush(heap,(ready+rates[a],i))
                snar.append({'scenario':scenario,'new_trajectories':count,'cards':cards,'new_oracle_calls':count*50,'job_duration_basis':'actual original accepted job elapsed incl load/domain audit','minutes':max(t for t,i in heap),'no_new_training':True,'not_registered_except12':count!=12})
    output={'services.csv':csvbytes(services),'actual_allocation_scenarios.json':(json.dumps(cases,indent=2)+'\n').encode(),'SnAr_queue_times.csv':csvbytes(snar)}
    return output,{'MADE':cases,'SnAr':snar}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');a=ap.parse_args();out,summary=compute()
    for name,b in out.items():
        if a.write:(D/name).write_bytes(b)
        else:assert (D/name).read_bytes()==b,name
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
