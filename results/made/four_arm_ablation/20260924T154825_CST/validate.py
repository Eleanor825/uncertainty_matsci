"""Standalone verification of the published fixed four-arm matrix and aggregates."""
from pathlib import Path
from collections import Counter
import hashlib,json,math,re
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def same(a,b):
    if b is None:assert a is None
    else:assert math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12),(a,b)
def check_stats(record,values,paired=False):
    n=len(values);assert record['n']==n
    mean=sum(values)/n if n else None
    var=sum((v-mean)**2 for v in values)/(n-1) if n>1 else None
    same(record['mean'],mean);same(record['sample_variance'],var);same(record['sample_SD'],math.sqrt(var) if var is not None else None)
    if paired:
        assert [record[k] for k in ('wins','ties','losses')]==[sum(v>1e-12 for v in values),sum(abs(v)<=1e-12 for v in values),sum(v<-1e-12 for v in values)]
        same(record['sum'],sum(values))
def main():
    manifest=read(HERE/'manifest.json')
    for name,pin in manifest['files'].items():
        p=HERE/name;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
    s=read(HERE/'snapshot.json');pairs=s['pairs'];arms=('Native1','Common2','Public2','ActionPool2')
    assert len(pairs)==60 and {(p['task_id'],p['budget'],p['seed']) for p in pairs}=={(t,b,z) for t in ('Au-K-Tb','Mg-Sn-Sr') for b in (10,20,30) for z in range(501,511)}
    for p in pairs:
        assert set(p['arms'])==set(arms)
        assert p['complete_four_arms']==all(r['state']=='accepted' for r in p['arms'].values())
        for arm,r in p['arms'].items():
            assert (r['task_id'],r['budget'],r['seed'],r['arm'])==(p['task_id'],p['budget'],p['seed'],arm)
            if r['state']=='accepted':
                assert r['completion_ref'] and re.fullmatch('[a-f0-9]{64}',r['completion_ref']['sha256'])
                assert 0<=r['metrics']['SUN']<=p['budget'] and 0<=r['metrics']['AUDC']<=1
                if arm in ('Common2','Public2'):assert r['online_costs']['capture_forward_intents']==r['online_costs']['capture_forward_returns']==0
            else:assert r['metrics'] is None and r['completion_ref'] is None
    ab=[p['arms'][a] for p in pairs for a in ('Common2','Public2')]
    assert Counter(r['state'] for r in ab)==s['ablation_state_counts']=={'accepted':16,'claimed':4,'clean_new':99,'blocked_prior_failure':1}
    assert sum(r['state']=='accepted' for r in ab if r['arm']=='Common2')==9
    assert sum(r['state']=='accepted' for r in ab if r['arm']=='Public2')==7
    complete=[p for p in pairs if p['complete_four_arms']]
    assert len(complete)==s['complete_four_arm_groups']==7 and all(p['budget']==10 for p in complete)
    assert Counter(p['task_id'] for p in complete)=={'Au-K-Tb':4,'Mg-Sn-Sr':3}
    for g in s['matched_four_arm_statistics']:
        sub=[p for p in complete if p['budget']==g['budget'] and (g['task_id'] is None or p['task_id']==g['task_id'])]
        assert g['n_complete_four_arms']==len(sub)
        for a in arms:
            for m in ('SUN','AUDC'):check_stats(g['arms'][a][m],[p['arms'][a]['metrics'][m] for p in sub])
        for name,record in g['comparisons'].items():
            a,b=name.split('_minus_')
            for m in ('SUN','AUDC'):check_stats(record[m],[p['arms'][a]['metrics'][m]-p['arms'][b]['metrics'][m] for p in sub],True)
    for g in s['status_by_budget_arm']:
        rows=[p['arms'][g['arm']] for p in pairs if p['budget']==g['budget']]
        assert g['expected']==20 and Counter(r['state'] for r in rows)==g['states']
    diag=s['posthoc_concentration_diagnostic'];sub=[p for p in complete if (p['task_id'],p['seed'])!=('Mg-Sn-Sr',503)]
    assert len(sub)==6
    for m in ('SUN','AUDC'):check_stats(diag['ActionPool2_minus_Public2_remaining_six'][m],[p['arms']['ActionPool2']['metrics'][m]-p['arms']['Public2']['metrics'][m] for p in sub],True)
    assert s['statistics_method']['missing_outcome_imputation'] is False
    forbidden=re.compile(b'/'+rb'(?:mnt|Users|root)/|GPU-[0-9a-fA-F]{8}-|Bearer\s+\S+|BEGIN[^\n]*PRIVATE KEY|"(?:hostname|password|cookie|access_token|raw_prompt|token_ids)"\s*:')
    for name in manifest['files']:
        if name.endswith(('.json','.md')):assert not forbidden.search((HERE/name).read_bytes()),name
    print(json.dumps({'status':'passed','expected_cells':240,'accepted_ablation':16,'complete_four_arm_groups':7,'variance_ddof':1,'sample_variance_SD_and_paired_WTL_recomputed':True,'missing_metrics_imputed':False,'new_scientific_calls':0},sort_keys=True))
if __name__=='__main__':main()
