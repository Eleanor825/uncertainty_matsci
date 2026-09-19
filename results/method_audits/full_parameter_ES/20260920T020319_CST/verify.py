from pathlib import Path
import csv,hashlib,json,math
R=Path(__file__).resolve().parent;d=json.loads((R/'snapshot.json').read_text());b=d['branches']
assert len(b)==4 and d['all_inputs_unchanged'] and d['fits']==d['model_loads']==d['oracle_calls']==d['tensor_reads']==0
rows=list(csv.DictReader((R/'actual_updates.csv').open()));assert len(rows)==8
for name,branch in b.items():
 assert branch['verified_projection'] and branch['parameter_scope']=='full' and branch['manifest_tensor_count']==723 and branch['manifest_numel']==4539265536
 assert branch['manifest_sha256']=='477a30e2c510f587cc43a91892bb2cea33190ea064f865021eefd76f2d745501'
 assert len(branch['updates'])==2
 for update in branch['updates']:
  row=next(r for r in rows if r['run']==name and r['transition']==update['transition'])
  for key in ('parameter_tensor_count','parameter_numel','delta_entry_count','nonzero_delta_count','zero_delta_count'):assert int(row[key])==update[key]
  assert update['nonzero_delta_count']+update['zero_delta_count']==723
  assert math.isclose(float(row['delta_l2_combined']),update['delta_l2_combined'],abs_tol=1e-12)
  assert update['state_changed']==(update['state_hash_before']!=update['state_hash_after'])
  assert row['state_hash_before']==update['state_hash_before'] and row['state_hash_after']==update['state_hash_after']
  assert int(row['selected_generation'])==branch['selected_generation']
 assert branch['selected_matches_initial']==(branch['base_state_hash']==branch['selected_state_hash'])
assert b['SnAr_ES_only']['updates'][0]['nonzero_delta_count']==0
assert b['SnAr_ES_only']['selected_generation']==b['SnAr_full']['selected_generation']==0
assert b['MADE_ES_only']['selected_generation']==1 and b['MADE_full']['selected_generation']==2
for item in json.loads((R/'manifest.json').read_text())['files']:
 p=R/item['path'];assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
print(json.dumps({'passed':True,'branches':4,'recorded_generations':8,'parameter_tensors':723,'scalar_parameters':4539265536,'new_scientific_calls':0,'metadata_recheck_not_parameter_reload':True}))
