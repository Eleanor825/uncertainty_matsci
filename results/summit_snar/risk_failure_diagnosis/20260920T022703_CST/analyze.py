"""Deterministic descriptive analysis of existing data only; no learned fit/import."""
import csv,gzip,hashlib,json,math,statistics
from pathlib import Path
BASE=Path(__file__).resolve().parent
OUT=Path(__file__).resolve().parent
INPUTS={'es': 'inputs/es.json.gz', 'families': 'inputs/families.json.gz', 'full_dev': 'inputs/full_dev.json.gz', 'projection': 'inputs/projection.json.gz', 'reproduction': 'inputs/reproduction.json.gz', 'union': 'inputs/union.json.gz'}
SOURCES=['benchmark_extensions/summit_snar_20260918/pipeline.py','benchmark_extensions/summit_snar_20260918/study.py','benchmark_extensions/summit_snar_20260918/method/features.py','benchmark_extensions/summit_snar_20260918/method/snar_policy.py','benchmark_extensions/summit_snar_20260918/protocol.json','benchmark_extensions/summit_snar_main_repair_20260919_v2/repair_pipeline.py','benchmark_extensions/summit_snar_main_repair_20260919_v2/acceptance_repair.py','benchmark_extensions/summit_snar_seed_extension_20260920_v1/runner.py','technical_not_main/snar_sampling_hidden_scientific_bundle_20260919_v1/sources/original_uncertainty.py','benchmark_extensions/made_support_aware_method_20260919/frozen_src/matdiscovery/policy.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(rel):
 p=BASE/rel;raw=p.read_bytes();return json.loads(gzip.decompress(raw) if p.suffix=='.gz' else raw)
def mean(xs):return statistics.mean(xs) if xs else None
def summary(xs):return dict(n=len(xs),mean=mean(xs),min=min(xs) if xs else None,max=max(xs) if xs else None)
def output(name,value):
 p=OUT/name;raw=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
 if p.exists():assert p.read_text()==raw,'Existing derived value differs'
 else:p.write_text(raw)
def table(name,rows):
 p=OUT/name
 import io
 f=io.StringIO(newline='');w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows);raw=f.getvalue()
 if p.exists():assert p.read_text()==raw.replace('\r\n','\n')
 else:p.write_text(raw)

def main():
 refs={k:{'path':r,'sha256':sha(BASE/r)} for k,r in INPUTS.items()}
 sources=json.loads((OUT/'code_references.json').read_text())
 x={k:read(r) for k,r in INPUTS.items()};p=x['projection'];r=x['families']
 rows=p['risk_fit']['rows'];names=p['risk_fit']['feature_names'];assert len(names)==285
 assert len(rows['train'])==75 and len(rows['dev'])==50
 cohorts={s:[r['features']['values'] for r in rs] for s,rs in rows.items()}
 valid=[r for r in p['candidates'] if r.get('features') and r.get('risk') is not None]
 selected=[r for r in valid if r['selected']];first=[r for r in valid if r['proposal_index']==0]
 assert len(valid)==239 and len(selected)==224
 cohorts.update(all_scored_candidates=[r['features']['values'] for r in valid],selected=[r['features']['values'] for r in selected],first_scored=[r['features']['values'] for r in first])
 feature_rows=[]
 for n in names:
  train=[r[n] for r in cohorts['train']];assert all(math.isfinite(z) for z in train)
  mu=mean(train);sd=statistics.pstdev(train);used=sd if sd>=1e-8 else 1.0
  for cohort,rs in cohorts.items():
   a=[r[n] for r in rs];assert all(math.isfinite(z) for z in a)
   z=[(v-mu)/used for v in a]
   feature_rows.append(dict(feature=n,family=n.split('.')[0],cohort=cohort,n=len(a),train_mean=mu,train_std=sd,normalizer_std=used,std_floor_applied=sd<1e-8,mean=mean(a),standardized_mean=mean(z),max_abs_z=max(abs(v) for v in z),outside_train_range=sum(v<min(train) or v>max(train) for v in a),abs_z_above3=sum(abs(v)>3 for v in z),abs_z_above5=sum(abs(v)>5 for v in z)))
 table('feature_support.csv',feature_rows)
 norm=[]
 for cohort in cohorts:
  for fam in ('sampling','hidden','graph'):
   a=[r for r in feature_rows if r['cohort']==cohort and r['family']==fam]
   norm.append(dict(cohort=cohort,family=fam,features=len(a),cells=sum(r['n'] for r in a),outside_train_range_cells=sum(r['outside_train_range'] for r in a),abs_z_above3_cells=sum(r['abs_z_above3'] for r in a),abs_z_above5_cells=sum(r['abs_z_above5'] for r in a),max_abs_z=max(r['max_abs_z'] for r in a),std_floor_features=sum(r['std_floor_applied'] for r in a)))
 table('normalization_support.csv',norm)
 original=next(m for m in r['model_reports'] if m['family']=='all' and m['seed']==1729)
 pred=[v for v in r['predictions'] if v['family']=='all' and v['initialization_seed']==1729]
 prob={'dev':summary(original['development_predictions']['calibrated']), 'first_scored':summary([z['risk'] for z in first]),'all_scored_candidates':summary([z['risk'] for z in valid]),'selected':summary([z['risk'] for z in selected]),'unselected_scored':summary([z['risk'] for z in valid if not z['selected']])}
 for label,vs in [('dev',original['development_predictions']['calibrated']),('selected',[z['risk'] for z in selected])]:prob[label]['at_least_point5']=sum(v>=.5 for v in vs)
 pred_by_query={z['query_id']:z for z in pred}
 q=[z for z in p['queries'] if z['arm']=='uq_esopt' and z['query_id'] in pred_by_query]
 assert len(q)==224 and sum(z['no_hvi'] for z in q)==218
 assert all(pred_by_query[z['query_id']]['no_hvi_label']==z['no_hvi'] for z in q)
 assert all(pred_by_query[z['query_id']]['original_saved_probability']==z['risk'] for z in q)
 cohort_labels={s:{'n':len(rs),'no_hvi':sum(z['label'] for z in rs)} for s,rs in rows.items()}
 cohort_labels['selected']={'n':224,'no_hvi':218}
 dev=[]
 for g in range(3):
  ss=[e['summary'] for e in x['full_dev'] if f'G{g}_' in e['name']];assert len(ss)==2
  a={k:mean([s[k] for s in ss]) for k in ['mean_querywise_hv','brier','invalid_proposal_rate','final_hv_gain']}
  a.update(generation=g,fitness=a['mean_querywise_hv']-.1*a['brier']-.1*a['invalid_proposal_rate'],prior_rows=0)
  dev.append(a)
 table('full_development_reward.csv',dev)
 es=[]
 for branch,b in x['es']['branches'].items():
  N=b['manifest_numel'];assert N==4539265536
  for u in b['updates']:
   exploration=math.sqrt(N)*u['sigma'];theory=math.sqrt(N/2)*u['alpha']
   es.append(dict(branch=branch,transition=u['transition'],population=2,normalized_rewards=json.dumps(u['normalized_rewards']),sigma=u['sigma'],alpha=u['alpha'],expected_isotropic_perturbation_l2=exploration,expected_update_l2_if_zscores_plusminus1=theory,actual_delta_l2=u['delta_l2_combined'],actual_to_expected_perturbation_ratio=u['delta_l2_combined']/exploration,selected_generation=b['selected_generation']))
 table('ES_update_scale.csv',es)
 models=[]
 for m in r['model_reports']:
  ps=[z for z in r['predictions'] if z['family']==m['family'] and z['initialization_seed']==m['seed']]
  models.append(dict(family=m['family'],seed=m['seed'],selected_epoch=m['selected_epoch'],optimizer_steps=m['optimizer_steps'],temperature=m['temperature'],dev_mean_p=mean(m['development_predictions']['calibrated']),test_mean_p=mean([z['calibrated'] for z in ps]),dev_brier=m['development']['calibrated']['brier'],test_brier=mean([(z['calibrated']-z['no_hvi_label'])**2 for z in ps]),raw_to_cal_threshold_changes=sum((z['raw']>=.5)!=(z['calibrated']>=.5) for z in ps)))
 table('model_calibration.csv',models)
 out=dict(schema='SnAr_existing_evidence_failure_diagnosis_v1',source_refs=refs,code_refs=sources,
  cohorts=cohort_labels,original_probabilities=prob,normalization_support=norm,
  original_temperature=original['temperature'],original_selected_epoch=original['selected_epoch'],original_optimizer_steps=original['optimizer_steps'],
  selected_raw_development_bce=original['history'][original['selected_epoch']-1]['development_bce'],
  original_development_metrics=original['development'],full_development=dev,
  full_G1_minus_G0={k:dev[1][k]-dev[0][k] for k in ['mean_querywise_hv','brier','fitness']},
  reproduced_original_gate=x['reproduction'],new_fits=0,new_model_calls=0,new_graph_calls=0,new_oracle_calls=0,
  limits=['Statistics on prior accepted projections, no new tensor loading or physical re-audit.','Existing test actions describe failure only; no test-driven hyperparameter/threshold/feature selection.','No causal attribution between archive, prefix context, candidate selection, and probabilities.'])
 output('analysis.json',out)
 output('analysis_manifest.json',{'source':{'path':Path(__file__).name,'sha256':sha(Path(__file__))},'inputs':refs,'files':[{'name':n,'sha256':sha(OUT/n)} for n in ['analysis.json','feature_support.csv','normalization_support.csv','full_development_reward.csv','ES_update_scale.csv','model_calibration.csv']]})
 print(json.dumps({'probabilities':prob,'full_G1_minus_G0':out['full_G1_minus_G0'],'normalization':[z for z in norm if z['cohort'] in ('dev','selected')]}))
if __name__=='__main__':main()
