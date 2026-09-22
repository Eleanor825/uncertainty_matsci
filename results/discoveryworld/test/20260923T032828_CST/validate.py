"""Independent static arithmetic, evidence-chain and privacy validation; no model imports."""
from pathlib import Path
import csv,hashlib,json,math,re
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=read('comparison.json');e=read('ES_trace.json');s=read('source_evidence.json');p=read('provenance.json');m=read('manifest.json')
 rows=list(csv.DictReader((P/'metrics.csv').open()));assert len(rows)==len(c['rows'])==len(s['episodes'])==2
 for a,b in zip(rows,c['rows']):
  assert a['arm']==b['arm'];assert int(a['world_seed'])==3 and int(a['policy_seed'])==401 and int(a['budget'])==int(a['actual_actions'])==30
  fail=int(a['action_failures']);score=float(a['task_score']);assert 0<=fail<=30 and 0<=score<=1
  assert math.isclose(float(a['official_fitness']),score-.1*fail/30,abs_tol=1e-12)
  assert a['task_success']=='False'and a['accepted_RPC_closed']=='True'
 n,f=c['rows'];delta=c['Full_minus_Native'];assert delta['task_score']==f['task_score']-n['task_score']==.125
 assert delta['action_failures']==f['action_failures']-n['action_failures']==-7 and math.isclose(delta['failure_fraction'],-7/30)
 assert c['paired_episodes']==1 and c['both_tasks_unsuccessful']and not c['significance_or_multiseed_claimed']and not c['method_component_effect_isolated']and not c['matched_LLM_compute_claimed']
 checks=c['comparison_checks'];assert checks['same_model_prompt_decoding_contract']and checks['same_environment_evaluator_runtime_contract_except_output']and checks['same_world_seed_policy_seed_B30']
 assert len(e['updates'])==2 and all(x['state_changed']and x['record']['state_hash_before']!=x['record']['state_hash_after']for x in e['updates'].values())
 assert e['updates']['1']['record']['state_hash_after']==e['updates']['2']['record']['state_hash_before']
 g=e['selected']['generation'];assert g==c['selected_generation']and e['selected']==e['all_checkpoint_metadata'][str(g)]
 reload=e['test_selected_state_reload']['value'];assert reload['full_parameter_reload_verified']and reload['selected']==e['selected']and reload['actual_model_state_hash']==e['selected']['model_state_hash']
 assert checks['same_actual_initial_parameter_hash']and checks['same_actual_initial_configuration_fingerprint']
 for x in e['updates'].values():
  values=x['record']['parameter_delta_l2'];summary=x['parameter_delta_summary'];assert summary['tensor_count']==len(values)==723 and summary['nonzero_tensor_count']==sum(z>0 for z in values.values())==723 and math.isclose(summary['whole_parameter_delta_l2'],math.sqrt(sum(z*z for z in values.values())))
 assert e['selected_qualification']['unique_token_prefix_count']==8
 assert e['selected_checkpoint_bytes_rehashed']and e['selected_qualification']['qualified']and e['parameter_manifest']['scope']=='full'and e['parameter_manifest']['total_parameters']>0
 assert e['ES_counts']['accepted_episodes']==e['ES_counts']['prescribed_episodes']==7 and e['ES_counts']['generations_computed']==2 and e['ES_counts']['new_ES_episodes']==6 and e['ES_counts']['reused_G0_episodes']==1
 assert p['new_model_environment_graph_NN_ES_calls']==0 and p['existing_MADE_and_DW_results_unchanged']and not p['raw_operation_receipts_published']
 for x in s['episodes']:
  assert x['raw_official_metrics_recomputed'];j=x['job'];assert(j['world_seed'],j['policy_seed'],j['model_key'],j['max_agent_attempts'])==(3,401,'qwen35_4b',30)
  for item in x['source_refs'].values():
   if item:assert not item['path'].startswith('/')and re.fullmatch('[a-f0-9]{64}',item['sha256'])
 upcoming=read('prospective_status.json');assert upcoming['scientific_calls_in_registration_receipt']==0 and not upcoming['science_start_or_completion_inferred_from_registration']
 found={str(x.relative_to(P))for x in P.rglob('*')if x.is_file()and x.name!='manifest.json'};assert set(m['files'])==found
 for name,item in m['files'].items():
  path=P/name;assert path.stat().st_size==item['bytes']and sha(path)==item['sha256']
  if path.suffix in ('.json','.csv','.md'):
   value=path.read_text();assert not re.search(r'/Users/|/mnt/|/private/|/tmp/|jupyter\.[A-Za-z0-9]|GPU-[a-f0-9]{8}|de-41039|tj-3041039|password\s*[:=]|Cookie\s*[:=]',value,re.I),name
   assert 'raw_prompt'not in value and 'prompt_messages'not in value,name
 print(json.dumps({'passed':True,'files':len(found)+1,'paired_episodes':1,'raw_metric_arithmetic':True,'selected_checkpoint_and_actual_reload_match':True,'actual_ES_updates':2,'privacy_scan':'PASS','new_scientific_calls':0},sort_keys=True))
if __name__=='__main__':main()
