from pathlib import Path
from collections import Counter
import json,hashlib,time
refs={'original_controller': {'path': '/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607/benchmark_extensions/made_controller_repair_20260919/development_validation/jobs/dev-schema-Al-Pd-Sm-B50-seed2-original_controller/8405744691f148a0907d3e292bfc7638/environment/oracle_attempts.jsonl', 'sha256': 'd3fbd8e5544e70725febfa3e5a5a4ba91ddc695f3d5be8007c1782f3e1ef1f3e', 'bytes': 177170}, 'schema_repaired_controller': {'path': '/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607/benchmark_extensions/made_controller_repair_20260919/development_validation/jobs/dev-schema-Al-Pd-Sm-B50-seed2-schema_repaired_controller/53429ab1c2854b62ad8b6a59019af9bf/environment/oracle_attempts.jsonl', 'sha256': 'a70449ddad68577da45731eee8fd6f77ffeab7d795f6d6450d566c9c80c040a0', 'bytes': 272779}}
out={'read_at':time.time(),'arms':{},'source_files':{}}
for arm,ref in refs.items():
 p=Path(ref['path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==ref['sha256'];rows=[json.loads(x) for x in b.splitlines()]
 starts={r['attempt_id']:r for r in rows if r['kind']=='oracle_attempt_started'};returns={r['attempt_id']:r for r in rows if r['kind']=='oracle_attempt_returned'};errors=[r for r in rows if r['kind'] not in ['oracle_constructor_started','oracle_constructor_finished','oracle_batch_started','oracle_batch_finished','oracle_attempt_started','oracle_attempt_returned']]
 assert set(starts)==set(returns) and len(starts)==sum(r['kind']=='oracle_attempt_started' for r in rows)
 out['arms'][arm]={'source':ref,'kind_counts':dict(Counter(r['kind'] for r in rows)),'attempts_by_role_phase':dict(Counter(r.get('role','')+'|'+r.get('phase','') for r in starts.values())),'attempts_by_counter':dict(Counter(r.get('counter') for r in starts.values())),'returned_attempts':len(returns),'unclosed_attempts':len(set(starts)-set(returns)),'other_kinds':dict(Counter(r['kind'] for r in errors)),'candidate_hashes':[{k:r.get(k) for k in ['sequence','attempt_id','candidate_hash','role','phase','counter']} for r in starts.values() if r.get('phase')!='initialization']}
D=Path('/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607/benchmark_extensions/made_controller_repair_20260919/development_validation')
for arm in refs:
 for name,sections in [('failure_controller.py',[(1,340)]),('benchmark_adapters.py',[(350,430),(540,640)]),('rollouts.py',[(270,465)]),('failure_labels.py',[(120,170)])]:
  p=D/arm/'src/matdiscovery'/name;b=p.read_bytes();lines=b.decode().splitlines();out['source_files'][arm+'/'+name]={'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'excerpts':[{'first_line':lo,'last_line':min(hi,len(lines)),'text':'\n'.join(lines[lo-1:hi])} for lo,hi in sections]}
print(json.dumps(out,allow_nan=False),flush=True)
