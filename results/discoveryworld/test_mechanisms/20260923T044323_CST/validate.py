from pathlib import Path
import csv,hashlib,json,math,re
P=Path(__file__).resolve().parent
read=lambda n:json.loads((P/n).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 m=read('mechanism.json');p=read('provenance.json');manifest=read('manifest.json');rows=list(csv.DictReader((P/'steps.csv').open()))
 assert [x['attempt']for x in m['steps']]==[22,23,24,25]and [int(x['attempt'])for x in rows]==[22,23,24,25]
 assert m['all_four_selected_steps_retained']and m['positive_score_events']==[{'after':.125,'attempt':24,'before':0.,'delta':.125}]
 assert (m['world_seed'],m['policy_seed'],m['budget'],m['selected_generation'])==(3,401,30,1)
 assert m['Native_final_score']==0 and m['Full_final_score']==.125 and m['both_task_success_false']
 assert m['new_scientific_calls']==p['new_model_environment_graph_fitting_calls']==0
 assert not m['unexecuted_outcomes_inferred']and not m['calibration_failure_inferred_from_single_case']and not m['full_private_state_equivalence_claimed']
 for step,row in zip(m['steps'],rows):
  assert step['official_result']['success']is True and row['official_action_success']=='True'
  assert float(row['first_failure_risk'])==step['first_decision']['risk'];assert float(row['official_score_delta'])==step['score_delta']
  actual=step['actual_semantic_action'];first=step['candidates'][0]['semantic_action'];assert (row['actual_packet_changed_from_first']=='True')==(actual!=first)
  for c in step['candidates']:
   raw=c['parsed_action'];canonical=dict(raw)
   if canonical['action']=='TELEPORT_TO_OBJECT':canonical['arg1']=int(canonical['arg1'])
   assert digest(raw)==c['raw_parsed_packet_sha256']and canonical==c['semantic_action']and digest(canonical)==c['semantic_action_sha256']
   source=c['source'];assert not source['path'].startswith('/')and re.fullmatch('[a-f0-9]{64}',source['sha256'])
  assert step['unexecuted_distinct_candidate_outcome']is None and not step['local_necessity_or_counterfactual_benefit_proven']
 a,b,c,d=m['steps'];assert a['actual_semantic_action']==b['actual_semantic_action']and a['actual_raw_action']!=b['actual_raw_action']
 assert a['first_decision']['neural_revision']and b['first_decision']['neural_revision']and a['executed_matching_proposals']==b['executed_matching_proposals']==[1]
 assert c['candidates'][0]['parsed_action']==c['candidates'][1]['parsed_action']==c['actual_raw_action']and c['executed_matching_proposals']==[0,1]and not c['chosen_index_known_from_unique_packet_match']
 assert c['first_decision']['risk']>.5 and c['official_result']['success']and c['score_delta']==.125
 assert len(d['candidates'])==1 and d['first_decision']['risk']<.5 and not d['first_decision']['neural_revision']and d['score_delta']==0
 assert p['total_remote_bytes_read']==1639870 and not p['large_checkpoint_or_policy_attempt_log_read']and not p['existing_numerical_results_changed']
 assert p['original_pair_sources']['Full']['completion']['sha256']=='a78a4ceae6797daaac268fbe0aa8157a842d23caaf8b21ba432b1376f5f441f3'
 assert p['original_pair_sources']['Native']['completion']['sha256']=='cda074b539a0b4f6657893352357b1824ac5ea82c75dbd7720c578eaf8505c55'
 actual={str(x.relative_to(P))for x in P.rglob('*')if x.is_file()and x.name!='manifest.json'};assert set(manifest['files'])==actual
 for name,v in manifest['files'].items():
  f=P/name;assert f.stat().st_size==v['bytes']and hashlib.sha256(f.read_bytes()).hexdigest()==v['sha256']
  if f.suffix in ('.json','.csv','.md'):
   text=f.read_text();assert not re.search(r'/Users/|/mnt/|/private/|/tmp/|GPU-[a-f0-9]{8}|de-41039|tj-3041039|password\s*[:=]|Cookie\s*[:=]',text,re.I)
 print(json.dumps({'passed':True,'selected_steps':4,'all_generated_candidates_retained':7,'semantic_normalization_verified':True,'numerical_endpoints_unchanged':True,'new_scientific_calls':0,'privacy':'PASS','incremental_files_hash_checked':len(actual)+1},sort_keys=True))
if __name__=='__main__':main()
