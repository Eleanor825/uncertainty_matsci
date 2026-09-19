"""Exact cached SnAr candidate analysis. Stdlib only; no model/oracle/network calls.

Default verifies deterministic outputs; --write renders derived analysis only.
No unexecuted candidate is assigned a physical outcome.
"""
from pathlib import Path
from collections import defaultdict,Counter
from decimal import Decimal
from datetime import datetime,timezone,timedelta
import argparse,csv,gzip,hashlib,io,json,math,statistics as st
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'observations.json.gz'
EXPORT_GZIP_SHA256='d970e23e37da299fb1a5688faace1582e420ba185332beab9cf0ce5b0a626649'
EXPORT_JSON_SHA256='9b80f32160cf593831a3f18af8b970dd1694074022c5e9eab4bf5008f750a2e5'
PIN='9262eb9b552fabd6252f6d62cd6159a935730321adf63bb4c7543a04dd06cd1c'
PARAMS=('tau','equiv_pldn','conc_dfnb','temperature')
BOUNDS={'tau':(.5,2),'equiv_pldn':(1,5),'conc_dfnb':(.1,.5),'temperature':(30,120)}
ARMS=('qwen_base','uq_esopt','gp_ei_scalarized')
EPS=1e-12
R='project-artifact:'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean(x):return math.fsum(x)/len(x) if x else None
def close(a,b,tol=1e-11):assert abs(a-b)<=tol*max(1.,abs(a),abs(b)),(a,b)
def key(parameters):
    assert set(parameters)==set(PARAMS)
    result=[]
    for p in PARAMS:
        x=parameters[p];assert isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) and BOUNDS[p][0]<=x<=BOUNDS[p][1]
        result.append(Decimal(str(x)).normalize())
    return tuple(result)
def canonical(k):return '|'.join(format(x,'f') if x else '0' for x in k)
def quantile(values,p):
    if not values:return None
    values=sorted(values);x=(len(values)-1)*p;low=math.floor(x);high=math.ceil(x);return values[low]+(values[high]-values[low])*(x-low)
def desc(values):
    return {'finite_n':len(values),'mean':mean(values),'min':min(values) if values else None,'p05':quantile(values,.05),'p25':quantile(values,.25),'p50':quantile(values,.5),'p75':quantile(values,.75),'p95':quantile(values,.95),'max':max(values) if values else None}
def auc(y,p):
    pos=[v for a,v in zip(y,p) if a];neg=[v for a,v in zip(y,p) if not a]
    return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg)) if pos and neg else None
def ap(y,p):
    positives=sum(y)
    if not positives:return None
    tp=count=0;area=0.
    for threshold in sorted(set(p),reverse=True):
        ys=[a for a,v in zip(y,p) if v==threshold];tp+=sum(ys);count+=len(ys);area+=sum(ys)/positives*tp/count
    return area
def risk_metrics(rows):
    present=[r for r in rows if r['risk'] is not None];y=[r['no_hvi'] for r in present];p=[r['risk'] for r in present]
    for v in p:assert math.isfinite(v) and 0<=v<=1
    return {'query_n':len(rows),'scored_n':len(p),'unscored_n':len(rows)-len(p),'no_HVI_count_all':sum(r['no_hvi'] for r in rows),
      'HVI_count_all':sum(not r['no_hvi'] for r in rows),'failure_rate_scored':mean(y),'mean_risk':mean(p),
      'Brier':mean([(v-a)**2 for a,v in zip(y,p)]),'log_loss':-mean([a*math.log(max(1e-15,v))+(1-a)*math.log(max(1e-15,1-v)) for a,v in zip(y,p)]) if p else None,
      'failure_AUROC':auc(y,p),'failure_AUPRC':ap(y,p),'failure_AP_prevalence_baseline':mean(y),
      'success_AUPRC':ap([1-a for a in y],[1-v for v in p]),'success_AP_prevalence_baseline':mean([1-a for a in y]),
      'fixed_point5_TP':sum(a and v>=.5 for a,v in zip(y,p)),'fixed_point5_FN':sum(a and v<.5 for a,v in zip(y,p)),
      'fixed_point5_FP':sum(not a and v>=.5 for a,v in zip(y,p)),'fixed_point5_TN':sum(not a and v<.5 for a,v in zip(y,p)),
      'HV_increment_sum':math.fsum(r['hv_increment'] for r in rows),'HVI_positive_mean':mean([r['hv_increment'] for r in rows if not r['no_hvi']])}
def csv_bytes(rows):
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return f.getvalue().encode()
def json_bytes(v):return (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def ref(r):return {'source_relative_path':r['path'][len(R):] if r['path'].startswith(R) else r['path'],'source_sha256':r['sha256']}
def analyze(v):
    assert v['gaps']==[] and len(v['queries'])==750 and len(v['candidates'])==248 and len(v['episodes'])==15
    assert all(v[k]==0 for k in ('graph_calls','model_calls','physical_queries_replayed','remote_writes'))
    assert not v['risk_fit']['test_used'] and len(v['risk_fit']['rows']['train'])==75 and len(v['risk_fit']['rows']['dev'])==50
    episodes={e['name']:e for e in v['episodes']};assert len(episodes)==15
    queries={(q['episode'],q['query_index']):q for q in v['queries']};assert len(queries)==750
    candidates=defaultdict(list)
    for c in v['candidates']:candidates[c['episode'],c['query_index']].append(c)
    full=[];retryrows=[];crows=[];queryrows=[];repeats=[];repeatgroups=[];successes=[];group_summary=[]
    for ep in sorted(episodes):
        e=episodes[ep];arm=e['arm'];seed=e['seed'];assert arm in ARMS and seed in range(5101,5106)
        rr=[queries[ep,i] for i in range(50)];summ=e['summary'];curve=[summ['prior_hv'],*summ['hv_curve']];assert len(summ['hv_curve'])==50 and len(curve)==51
        assert summ['budget']==50 and summ['arm']==arm and summ['seed']==seed
        assert len({r['query_id'] for r in rr})==50
        seen={};occurrences=defaultdict(list);ep_post=[];ep_repeats=[]
        for i,q in enumerate(rr):
            assert q['query_index']==i and q['query_number']==i+1 and q['episode']==ep and q['arm']==arm and q['seed']==seed
            assert q['no_hvi']==int(q['hv_increment']<=EPS)
            close(curve[i+1]-curve[i],q['hv_increment']);k=key(q['parameters'])
            previous=seen.get(k);isrepeat=previous is not None
            eq=None
            if isrepeat:
                eq=max(abs(q['objectives'][p]-previous['objectives'][p]) for p in ('sty','e_factor'))
            if i>=5:
                ep_post.append(q)
                if isrepeat:ep_repeats.append(q)
            occurrences[k].append(q)
            if k not in seen:seen[k]=q
            out={'episode':ep,'arm':arm,'seed':seed,'query_number':i+1,'kind':q['kind'],**q['parameters'],
                'canonical_numeric_tuple':canonical(k),'repeat_of_earlier_executed_query':isrepeat,'first_query_number':previous['query_number'] if previous else i+1,
                'objective_max_abs_difference_from_first_same_tuple':eq,'sty':q['objectives']['sty'],'e_factor':q['objectives']['e_factor'],
                'no_HVI':q['no_hvi'],'HV_increment':q['hv_increment'],'risk':q['risk'],'proposal_count':q['proposal_count'],
                'invalid_proposals':q['invalid_proposals'],'elapsed_seconds':q['elapsed_seconds'],**ref(q['source'])}
            queryrows.append(out)
            if not q['no_hvi']:successes.append(out)
            if arm!='uq_esopt' or i<5:
                if i<5:assert q['proposal_count']==0 and q['risk'] is None
                continue
            full.append(q);cc=sorted(candidates[ep,i],key=lambda c:c['proposal_index']);assert [c['proposal_index'] for c in cc]==list(range(q['proposal_count']))
            assert q['proposal_count'] in (1,2) and q['invalid_proposals']==sum(not c['success'] for c in cc)
            first=cc[0];reason='none'
            if len(cc)==2:
                reason='invalid_first' if not first['success'] else 'graph_or_risk_unavailable_first' if first['risk'] is None else 'high_risk_first'
                if reason=='high_risk_first':assert first['risk']>=.5
                if reason=='graph_or_risk_unavailable_first':assert first['graph_unavailable_type'] is not None
            else:assert first['success'] and first['risk'] is not None and first['risk']<.5
            valid=[c for c in cc if c['success']]
            chosen=min(valid,key=lambda c:c['risk'] if c['risk'] is not None else math.inf) if valid else None
            assert [c for c in cc if c['selected']]==([chosen] if chosen else [])
            if chosen:
                assert q['chosen_features_path']==chosen['source']['path'] and q['chosen_generation']==chosen['generation_path']
                assert key(q['parameters'])==key(chosen['parameters']) and q['risk']==chosen['risk']
            else:assert q['kind']=='explicit_invalid_fallback' and q['risk'] is None
            retryrows.append({'episode':ep,'seed':seed,'query_number':i+1,'proposal_count':len(cc),'retry_reason':reason,
                'first_success':first['success'],'first_risk':first['risk'],'first_graph_unavailable':first['graph_unavailable_type'],
                'second_success':cc[1]['success'] if len(cc)==2 else None,'second_risk':cc[1]['risk'] if len(cc)==2 else None,
                'selected_proposal_index':chosen['proposal_index'] if chosen else None,'explicit_invalid_fallback':chosen is None,
                'both_valid_candidates_same_parameters':len(valid)==2 and key(valid[0]['parameters'])==key(valid[1]['parameters']),
                'selected_no_HVI':q['no_hvi'],'selected_HV_increment':q['hv_increment'],'query_source_sha256':q['source']['sha256'],
                'first_candidate_sha256':first['source']['sha256'],'second_candidate_sha256':cc[1]['source']['sha256'] if len(cc)==2 else None})
            for c in cc:
                if c['success']:key(c['parameters'])
                else:assert c['parameters'] is None and c['risk'] is None
                crows.append({'episode':ep,'seed':seed,'query_number':i+1,'proposal_index':c['proposal_index'],'success':c['success'],
                    'failure_code':c['failure_code'],'graph_unavailable_type':c['graph_unavailable_type'],'risk':c['risk'],
                    'features_available':c['features'] is not None,'selected_and_executed':c['selected'],
                    **{p:c['parameters'][p] if c['parameters'] else None for p in PARAMS},
                    'executed_no_HVI':q['no_hvi'] if c['selected'] else None,'executed_HV_increment':q['hv_increment'] if c['selected'] else None,
                    **ref(c['source'])})
        close(math.fsum(q['hv_increment'] for q in rr),summ['final_hv_gain'])
        assert sum(q['proposal_count'] for q in rr)==summ['proposal_count'] and sum(q['invalid_proposals'] for q in rr)==summ['invalid_proposals']
        rm=risk_metrics(ep_post)
        if arm=='uq_esopt':close(rm['Brier'],summ['brier'])
        repeats.append({'episode':ep,'arm':arm,'seed':seed,'all_queries':50,'post_LHS_queries':45,'post_LHS_repeat_count':len(ep_repeats),
            'post_LHS_repeat_rate':len(ep_repeats)/45,'post_LHS_first_occurrence_count':45-len(ep_repeats),
            'repeat_HVI_count':sum(not r['no_hvi'] for r in ep_repeats),'repeat_HV_increment':math.fsum(r['hv_increment'] for r in ep_repeats),
            'post_LHS_HVI_count':sum(not r['no_hvi'] for r in ep_post),'post_LHS_HV_increment':math.fsum(r['hv_increment'] for r in ep_post),
            'unique_numeric_tuples_in_all50':len(seen),'total_proposals':summ['proposal_count'],'invalid_proposals':summ['invalid_proposals']})
        for k,items in occurrences.items():
            if len(items)>1:repeatgroups.append({'episode':ep,'arm':arm,'seed':seed,'canonical_numeric_tuple':canonical(k),
                'query_numbers':';'.join(str(r['query_number']) for r in items),'executions':len(items),'repeated_executions':len(items)-1,
                'first_HV_increment':items[0]['hv_increment'],'later_HV_increment_sum':math.fsum(r['hv_increment'] for r in items[1:]),
                'all_objectives_numerically_equal':all(r['objectives']==items[0]['objectives'] for r in items[1:])})
    assert len(full)==225 and len(retryrows)==225 and len(crows)==248
    for seed in range(5101,5106):
        for i in range(5):
            same=[q for q in v['queries'] if q['seed']==seed and q['query_index']==i];assert len(same)==3
            assert all(key(q['parameters'])==key(same[0]['parameters']) and q['objectives']==same[0]['objectives'] for q in same)
    retries=[r for r in retryrows if r['proposal_count']==2];triggers=Counter(r['retry_reason'] for r in retries)
    selected=[c for c in v['candidates'] if c['selected']];scored=[r for r in full if r['risk'] is not None]
    risk_all=risk_metrics(full);train_rate=mean([r['label'] for r in v['risk_fit']['rows']['train']]);dev_rate=mean([r['label'] for r in v['risk_fit']['rows']['dev']])
    risk_all.update(train_only_constant_probability=train_rate,train_only_constant_Brier=mean([(train_rate-r['no_hvi'])**2 for r in scored]),constant_half_Brier=.25)
    predicates={'all225_post_LHS':lambda q:True,'risk_le_point1':lambda q:q['risk'] is not None and q['risk']<=.1,
       'risk_lt_point5':lambda q:q['risk'] is not None and q['risk']<.5,'risk_ge_point5':lambda q:q['risk'] is not None and q['risk']>=.5,
       'no_risk_post_LHS':lambda q:q['risk'] is None,'single_proposal':lambda q:q['proposal_count']==1,'two_proposals':lambda q:q['proposal_count']==2}
    strata=[{'stratum':name,**risk_metrics([q for q in full if pred(q)])} for name,pred in predicates.items()]
    byseed=[{'seed':seed,**risk_metrics([r for r in full if r['seed']==seed])} for seed in range(5101,5106)]
    bins=[]
    for lo,hi in [(0,.1),(.1,.25),(.25,.5),(.5,.75),(.75,1.)]:
        rr=[q for q in scored if (q['risk']>=lo if lo==0 else q['risk']>lo) and q['risk']<=hi]
        bins.append({'bin':('[' if lo==0 else '(')+str(lo)+','+str(hi)+']',**risk_metrics(rr)})
    # Features are raw stored numbers, not an inference call or imputed fitted matrix.
    names=v['risk_fit']['feature_names'];assert len(names)==len(set(names))==285
    populations={'train':v['risk_fit']['rows']['train'],'dev':v['risk_fit']['rows']['dev'],'full_all_candidates':v['candidates'],'full_selected_candidates':selected}
    def finite(rows,name):return [r['features']['values'][name] for r in rows if r.get('features') is not None and isinstance(r['features']['values'].get(name),(float,int)) and not isinstance(r['features']['values'].get(name),bool) and math.isfinite(r['features']['values'][name])]
    fs=[];shifts=[];recordstats=[];groups=[];popsummary={}
    training={name:finite(populations['train'],name) for name in names}
    for pop,rows in populations.items():
        missing=0;outcells=0;outfeatures=0
        for name in names:
            values=finite(rows,name);d=desc(values);t=desc(training[name]);n=len(rows);miss=n-len(values)
            assert len(training[name])==75
            out=sum(x<t['min'] or x>t['max'] for x in values);missing+=miss;outcells+=out;outfeatures+=out>0
            fs.append({'population':pop,'feature':name,'family':name.split('.')[0],'records':n,'missing_n':miss,'missing_rate':miss/n,**d})
            if pop!='train':
                iq=t['p75']-t['p25'];shifts.append({'population':pop,'feature':name,'family':name.split('.')[0],
                   'finite_n':len(values),'missing_rate':miss/n,'train_min':t['min'],'train_p25':t['p25'],'train_p50':t['p50'],'train_p75':t['p75'],'train_max':t['max'],
                   'target_min':d['min'],'target_p25':d['p25'],'target_p50':d['p50'],'target_p75':d['p75'],'target_max':d['max'],
                   'median_shift_in_training_IQR':(d['p50']-t['p50'])/iq if iq and values else None,
                   'outside_training_range_n':out,'outside_training_range_rate_among_finite':out/len(values) if values else None,
                   'constant_training_feature':t['min']==t['max']})
        popsummary[pop]={'records':len(rows),'no_feature_vectors':sum(r.get('features') is None for r in rows),'feature_cells':len(rows)*285,'missing_cells':missing,'missing_cell_rate':missing/(len(rows)*285),'features_with_any_outside_training_range':outfeatures,'outside_training_range_cells':outcells}
        for fam in ('graph','hidden','sampling'):
            chosen=[r for r in fs if r['population']==pop and r['family']==fam];cols=len(chosen);familyn=sum(r['finite_n'] for r in chosen)
            familyout=sum(sum(x<min(training[r['feature']]) or x>max(training[r['feature']]) for x in finite(rows,r['feature'])) for r in chosen)
            groups.append({'population':pop,'feature_family':fam,'columns':cols,'records':len(rows),'missing_cells':sum(r['missing_n'] for r in chosen),'missing_cell_rate':sum(r['missing_n'] for r in chosen)/(len(rows)*cols),'outside_training_range_cells':familyout,'outside_training_range_fraction_among_finite':familyout/familyn if familyn else None})
        for i,row in enumerate(rows):
            features=row.get('features');vals=features['values'] if features else {};present=[k for k in names if isinstance(vals.get(k),(int,float)) and not isinstance(vals.get(k),bool) and math.isfinite(vals[k])]
            recordstats.append({'population':pop,'row_index':i,'episode':row['episode'],'query_number':row.get('query_number'),'proposal_index':row.get('proposal_index'),
              'selected':row.get('selected'),'risk':row.get('risk'),'feature_vector_available':features is not None,'stored_keys':len(vals),'missing_model_columns':285-len(present),
              'outside_train_range_columns':sum(vals[k]<min(training[k]) or vals[k]>max(training[k]) for k in present),
              'nonfinite_key_count':len(features['nonfinite_keys']) if features else None,'nonnumeric_key_count':len(features['nonnumeric_keys']) if features else None})
    armsummary={}
    for arm in ARMS:
        rr=[r for r in repeats if r['arm']==arm];armsummary[arm]={'episodes':5,'all_queries':250,'post_LHS_queries':225,
            'repeated_post_LHS_queries':sum(r['post_LHS_repeat_count'] for r in rr),'repeat_rate':sum(r['post_LHS_repeat_count'] for r in rr)/225,
            'repeat_HVI_count':sum(r['repeat_HVI_count'] for r in rr),'repeat_HV_increment':math.fsum(r['repeat_HV_increment'] for r in rr),
            'HVI_queries':sum(r['post_LHS_HVI_count'] for r in rr),'total_HV_gain':math.fsum(r['post_LHS_HV_increment'] for r in rr),
            'mean_episode_HV_gain':mean([r['post_LHS_HV_increment'] for r in rr]),'total_generated_proposals':sum(r['total_proposals'] for r in rr),'invalid_generated_proposals':sum(r['invalid_proposals'] for r in rr)}
    retrsummary={'second_proposal_queries':len(retries),'causes':dict(triggers),'selected_second':sum(r['selected_proposal_index']==1 for r in retries),
      'selected_first_despite_retry':sum(r['selected_proposal_index']==0 for r in retries),'no_valid_candidate_fallback':sum(r['explicit_invalid_fallback'] for r in retries),
      'two_proposal_HVI_queries':sum(not r['selected_no_HVI'] for r in retries),
      'invalid_candidates':sum(not c['success'] for c in v['candidates']),'valid_graph_unavailable_candidates':sum(c['success'] and c['risk'] is None for c in v['candidates']),
      'risk_available_candidates':sum(c['risk'] is not None for c in v['candidates']),
      'selected_candidates':len(selected),'unexecuted_candidates_without_labels':len(v['candidates'])-len(selected)}
    retry_by_seed=[]
    for seed in range(5101,5106):
        rr=[r for r in retryrows if r['seed']==seed];retry_by_seed.append({'seed':seed,'queries':len(rr),'second_proposals':sum(r['proposal_count']==2 for r in rr),
            'high_risk_first':sum(r['retry_reason']=='high_risk_first' for r in rr),'invalid_first':sum(r['retry_reason']=='invalid_first' for r in rr),'graph_unavailable_first':sum(r['retry_reason']=='graph_or_risk_unavailable_first' for r in rr),
            'selected_second':sum(r['selected_proposal_index']==1 for r in rr),'explicit_invalid_fallback':sum(r['explicit_invalid_fallback'] for r in rr),'HVI_queries':sum(not r['selected_no_HVI'] for r in rr)})
    featuretop=sorted([r for r in shifts if r['population']=='full_selected_candidates' and r['median_shift_in_training_IQR'] is not None],key=lambda r:abs(r['median_shift_in_training_IQR']),reverse=True)[:12]
    summary={'schema':'SnAr_exact_executed_candidate_diagnostic_v1','observed_CST':datetime.fromtimestamp(v['finished_epoch'],timezone(timedelta(hours=8))).isoformat(),
       'source_sha256':PIN,'source_read_gaps':v['gaps'],'scope':{'queries':750,'episodes':15,'full_candidates':248,'train_dev_feature_rows':125,'model_columns':285},
       'retry':retrsummary,'full_executed_post_LHS_risk':risk_all,'risk_by_seed':byseed,'risk_strata':strata,'parameter_repeats':armsummary,
       'feature_populations':popsummary,'largest_selected_median_shifts':featuretop,'original_risk_train_failure_rate':train_rate,'original_risk_dev_failure_rate':dev_rate,
       'original_risk_dev_metrics':v['risk_fit']['development_metrics'],'successful_full_queries':[{'seed':q['seed'],'query_number':q['query_number'],'risk':q['risk'],'HV_increment':q['hv_increment'],'proposal_count':q['proposal_count'],'parameters':q['parameters']} for q in full if not q['no_hvi']],
       'unexecuted_candidate_outcomes_invented':False,'selection_bias_acknowledged':True,'independent_query_CI_or_significance_claimed':False,
       'new_threshold_selection_or_test_calibration':False,'new_model_or_oracle_calls':0,'causal_effect_estimated':False}
    outputs={'queries.csv':csv_bytes(queryrows),'candidates.csv':csv_bytes(crows),'retry_decisions.csv':csv_bytes(retryrows),
       'retry_by_seed.csv':csv_bytes(retry_by_seed),'risk_by_seed.csv':csv_bytes(byseed),'risk_strata.csv':csv_bytes(strata),'risk_bins.csv':csv_bytes(bins),
       'parameter_repeats_by_episode.csv':csv_bytes(repeats),'repeat_groups.csv':csv_bytes(repeatgroups),'HVI_success_cases.csv':csv_bytes(successes),
       'feature_statistics.csv':csv_bytes(fs),'feature_shift_vs_train.csv':csv_bytes(shifts),'feature_family_statistics.csv':csv_bytes(groups),
       'feature_record_coverage.csv':csv_bytes(recordstats),'summary.json':json_bytes(summary)}
    return outputs,summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    assert sha(SOURCE)==EXPORT_GZIP_SHA256
    raw=gzip.decompress(SOURCE.read_bytes())
    assert hashlib.sha256(raw).hexdigest()==EXPORT_JSON_SHA256
    value=json.loads(raw);outputs,summary=analyze(value)
    for name,raw in outputs.items():
        if args.write:(HERE/name).write_bytes(raw)
        else:assert (HERE/name).read_bytes()==raw,'Mismatch: '+name
    if not args.write:
        manifest=json.loads((HERE/'manifest.json').read_text())
        for item in manifest['files']:
            path=HERE/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
    print(json.dumps({'passed':True,'full_risk':summary['full_executed_post_LHS_risk'],'retry':summary['retry'],'parameter_repeats':summary['parameter_repeats'],'feature_populations':summary['feature_populations']},sort_keys=True))
if __name__=='__main__':main()
