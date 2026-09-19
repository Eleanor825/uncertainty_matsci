from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
manifest=json.loads((P/'source/source_manifest.json').read_text())
for row in manifest['files']:
    b=(P/'source'/row['path']).read_bytes()
    assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
for row in [manifest['CPU_validation']]:
    assert hashlib.sha256((P/'source'/row['path']).read_bytes()).hexdigest()==row['sha256']
scope=json.loads((P/'registered_scope.json').read_text())
assert hashlib.sha256((P/'source/source_manifest.json').read_bytes()).hexdigest()==scope['source_manifest_sha256']
assert len(scope['jobs'])==len({j['seed'] for j in scope['jobs']})==5
assert [j['seed'] for j in scope['jobs'] if j['role']=='train']==[6101,6102,6103]
assert [j['seed'] for j in scope['jobs'] if j['role']=='dev']==[6201,6202]
assert all(j['budget']==30 and j['prior_rows']==550 and j['arm']=='qwen_base' for j in scope['jobs'])
assert sum(j['budget'] for j in scope['jobs'])==scope['new_official_ODE_queries']==150
assert scope['old_rows_used_for_NN_fitting']==scope['fresh_test_queries']==scope['actual_GPU_calls_at_registration']==0
assert scope['CPU_fits_maximum']==4 and scope['CPU_optimizer_steps_maximum']==400
for row in json.loads((P/'archive_manifest.json').read_text())['files']:
    b=(P/row['path']).read_bytes();assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
print(json.dumps({'passed':True,'registered_new_queries':150,'scientific_results_in_archive':False,'new_model_or_oracle_calls':0}))
