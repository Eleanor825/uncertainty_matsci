from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
EXPECTED={'accepted_AlAuHf_seed1.json': ('collection-made-qwen35-9b-v1-b2e27268bcd3ed3a45d766e5', 'Al-Au-Hf', 1, 'ca185adedd274df13d4b28b17b46021ec2036ed385ef157067e0a44d5bce740f', '73922184ce48c9c9389c55762efcd0c9ea174c236df14e810eef066056da6d41'), 'accepted_AlAuHf_seed2.json': ('collection-made-qwen35-9b-v1-4eeb32242569e1f59c2fea0b', 'Al-Au-Hf', 2, '0d3a3afe11927768a124159d2dced2a9bcc3b4b3b25df299d65509700d0f2667', 'ba737496257fcee493fac2c1c0d6d11d548ed78200dbd7506bee9174e8ac0676'), 'accepted_AuLiPd_seed1.json': ('collection-made-qwen35-9b-v1-6eea5f96cc5c5c2343bca6c9', 'Au-Li-Pd', 1, '7254f9aedd6fdd836470cc0d17327bedca6a482814af80ad5ce50d5ab60e05e7', '9b9b4a4688e529e31cf2446247df2cf69fc066706fe32bf51defd1a361b50f61')}
def sha(b):return hashlib.sha256(b).hexdigest()
progress=json.loads((P/'progress.json').read_text());assert progress['actual_new_audit_exit_code']==0 and progress['accepted_collections']==3 and progress['newly_accepted_this_audit']==1 and progress['registered_collections']==7
assert progress['test_count_increment']==progress['new_model_or_oracle_calls']==0
calls=initial=proposals=positions=bytecount=known=unknown=0
for name,(jid,task,seed,accepted_hash,result_hash) in EXPECTED.items():
 raw=(P/name).read_bytes();assert sha(raw)==accepted_hash
 for private in (b'/mnt/',b'/root/',b'/Users/',b'"hostname"',b'"argv"',b'"pid"',b'GPU-'):assert private not in raw
 v=json.loads(raw);j=v['job'];q=v['collection_proof'];e=v['episode'];row=next(x for x in progress['jobs'] if x['job_id']==jid)
 assert v['complete'] and v['original_audit_job_reexecuted_once'] and v['not_test_evaluation']
 assert j['job_id']==jid and j['seed']==seed and j['stage']=='collection' and j['split']=='train' and j['task_id']==task and j['budget']==50
 assert v['result_source']['sha256']==result_hash and v['accepted_test_count_increment']==v['new_model_graph_or_oracle_calls']==0
 assert q['activation_layers']==len(q['activation_shards'])==32 and len({s['layer_path'] for s in q['activation_shards']})==32
 assert all(s['rows']==q['activation_rows_per_layer'] for s in q['activation_shards'])
 assert q['activation_tensor_bytes']==32*q['activation_rows_per_layer']*4096*4*2
 assert q['full_prefix_tokens_verified'] and q['actual_LLM_seeds_verified'] and q['all_proposals_preserved']
 assert q['new_oracle_calls_for_audit']==0 and not q['online_graphs_or_NN'] and not q['offline_graphs_complete']
 assert e['costs']['candidate_oracle_attempts']==50 and e['discovery_curve'][-1][0]==50
 cv=e['discovery_curve'];area=sum((x1-x0)*(y1+y0) for (x0,y0),(x1,y1) in zip(cv,cv[1:]))/2500
 assert abs(area-e['metrics']['AUDC'])<1e-12 and row['AUDC']==e['metrics']['AUDC'] and row['SUN']==cv[-1][1]
 assert row['accepted_sha256']==accepted_hash and row['result_sha256']==result_hash
 assert sum(q['future_failure_counts'].values())==q['proposal_rows']
 calls+=50;initial+=e['costs']['initialization_oracle_attempts'];proposals+=q['proposal_rows'];positions+=q['activation_rows_per_layer'];bytecount+=q['activation_tensor_bytes']
 known+=q['future_failure_counts']['0']+q['future_failure_counts']['1'];unknown+=q['future_failure_counts']['unknown']
assert (calls,initial,proposals,positions,bytecount,known,unknown)==(150,80,500,8000,8388608000,449,51)
for ref in json.loads((P/'manifest.json').read_text())['files']:
 raw=(P/ref['path']).read_bytes();assert len(raw)==ref['bytes'] and sha(raw)==ref['sha256']
print(json.dumps({'passed':True,'accepted_training_collections':3,'newly_accepted_this_audit':1,'existing_candidate_calls':150,'test_count_increment':0,'new_model_graph_or_oracle_calls':0}))
