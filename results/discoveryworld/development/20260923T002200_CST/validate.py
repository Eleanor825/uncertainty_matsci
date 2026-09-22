from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,statistics
SEEDS=(309,310,311);ARMS=("Baseline","SFT","CreditSFT")
MANIFEST_SHA='cb033ae5427889ba2d17da6228caa54d46cf7d3d6cb0d0cfde070932b7414d63'
def need(ok,message):
 if not ok:raise ValueError(message)

def metric(row,name):
 if name=='scoreNormalized':return row['score']
 if name=='F':return row['score']+int(row['task_success'])-.1*row['failure_count']/row['steps']
 return row[name]

def describe(values):return {'mean':statistics.mean(values),'sample_SD':statistics.stdev(values),'sample_variance':statistics.variance(values),'n_evaluation_seeds':3}

def recompute_statistics(rows):
 need(len(rows)==9,'All9 required for three-seed statistics');metrics=('scoreNormalized','F','failure_count','failure_fraction')
 table={(x['job']['policy_seed'],x['job']['arm']):x for x in rows}
 out={arm:{k:describe([metric(table[seed,arm],k)for seed in SEEDS])for k in metrics}for arm in ARMS}
 out['paired_minus_Baseline']={arm:{k:describe([metric(table[seed,arm],k)-metric(table[seed,'Baseline'],k)for seed in SEEDS])for k in metrics}for arm in ARMS[1:]}
 out['variance_scope']='Three paired policy sampling seeds in one fixed world; not cross-world or training variance';return out

def validate_projection(d,require_final=True):
 need(d['schema']=='dw_fixed9_policy_development_action_audit_v1','Different source observer schema')
 need(d['world']==2 and d['prescribed_policy_seeds']==list(SEEDS),'Different planned cohort')
 need(d['raw_full_prompts_returned']is False and d['host_or_process_identity_returned']is False and d['new_model_or_NN_or_environment_calls']==0 and d['filesystem_mutations']==0,'Observer scope changed')
 need(d['frozen_source_manifest']['sha256']==MANIFEST_SHA,'Scientific source manifest differs')
 n=d['closed_seed_groups'];need(n in (1,2,3)and d['closed_episodes']==3*n,'Only complete three-arm seed groups may be exported')
 if require_final:need(n==3 and d['status']=='all9_complete_and_resource_closed'and d['resource_closed']and not d['failure_sources'],'Final export requires all9 and successful resource closure')
 expected=[(seed,arm)for seed in SEEDS[:n]for arm in ARMS];rows=d['episodes']
 need([(r['job']['policy_seed'],r['job']['arm'])for r in rows]==expected,'Rows missing, duplicated or reordered')
 for row in rows:
  job=row['job'];seed=job['policy_seed'];arm=job['arm']
  need(job['job_id']==f'dw-three-policy-dev-v1-{arm}-w2-p{seed}-b30'and job['world_seed']==2 and job['max_agent_attempts']==30 and job['project_split']=='dev','Wrong actual episode identity')
  need(row['environment_RPC_closed']and row['selected_action_to_official_outcome_join_verified'],'Unverified executed outcome')
  actions=row['rows'];need(0<len(actions)==row['steps']<=30 and [x['attempt_index']for x in actions]==list(range(1,len(actions)+1)),'Invalid actual action denominator')
  failures=sum(x['official_success']is False or not x['submitted']for x in actions)
  need(row['failure_count']==failures and row['failure_fraction']==failures/len(actions),'Failure denominator differs')
  need(type(row['task_success'])is bool and isinstance(row['score'],(int,float))and math.isfinite(row['score'])and 0<=row['score']<=1,'Invalid task endpoint')
  counts=Counter(x['action_type']for x in actions);need({k:v['attempts']for k,v in row['action_type_counts'].items()}==counts,'Action type counts differ')
  packets=[json.dumps(x['action'],sort_keys=True,separators=(',',':'),ensure_ascii=False)for x in actions];run=longest=0;previous=None
  for packet in packets:run=run+1 if packet==previous else 1;longest=max(longest,run);previous=packet
  need(len(set(packets))==row['unique_exact_action_packets']and longest==row['longest_identical_packet_run'],'Repeat counts differ')
  positive=[x['attempt_index']for x in actions if x['score_delta']is not None and x['score_delta']>0]
  need(positive==row['observed_positive_score_steps'],'Progress-step audit differs')
  directions=counts['ROTATE_DIRECTION']+counts['MOVE_DIRECTION'];need(directions==row['rotate_or_move_attempts']and row['rotate_or_move_fraction']==directions/len(actions)and row['all_attempts_rotate_or_move']==(directions==len(actions)),'Movement fraction differs')
  for action in ('USE','PICKUP'):need(row[action+'_successes']==sum(x['action_type']==action and x['official_success']is True for x in actions),'Successful scientific action count differs')
 need(d['actual_actions_audited']==sum(r['steps']for r in rows)<=270,'Total actual actions differ')
 table={(r['job']['policy_seed'],r['job']['arm']):r for r in rows}
 need([g['policy_seed']for g in d['paired_differences']]==list(SEEDS[:n]),'Paired cohort differs')
 for group in d['paired_differences']:
  seed=group['policy_seed']
  for arm in ARMS[1:]:
   expected_delta={k:metric(table[seed,arm],k)-metric(table[seed,'Baseline'],k)for k in ('scoreNormalized','F','failure_count','failure_fraction')}
   need(group['minus_Baseline'][arm]==expected_delta,'Paired differences differ')
 if n==3:need(d['statistics']==recompute_statistics(rows),'Mean/SD/sample variance differs')
 else:need(d['statistics']is None,'Partial groups must not pretend to be three-seed statistics')
 return {'validated_closed_episodes':len(rows),'validated_actions':d['actual_actions_audited'],'all9_final':bool(n==3 and d['resource_closed'])}
if __name__=='__main__':
 p=Path(__file__).resolve().parent;d=json.loads((p/'audit_projection.json').read_text());final=json.loads((p/'provenance.json').read_text())['final9_publication']
 result=validate_projection(d,require_final=final)
 rows=list(csv.DictReader((p/'actions.csv').open()));need(len(rows)==d['actual_actions_audited'],'CSV action denominator differs')
 for actual,source in zip(rows,[dict(policy_seed=e['job']['policy_seed'],arm=e['job']['arm'],**r)for e in d['episodes']for r in e['rows']]):
  need(int(actual['policy_seed'])==source['policy_seed']and actual['arm']==source['arm']and int(actual['attempt_index'])==source['attempt_index']and json.loads(actual['action'])==source['action'],'CSV/source join differs')
 for item in json.loads((p/'manifest.json').read_text())['files']:
  b=(p/item['path']).read_bytes();need(len(b)==item['bytes']and hashlib.sha256(b).hexdigest()==item['sha256'],'Bundle hash mismatch')
 print(json.dumps(dict(result,publication_scientific_calls=0)))
