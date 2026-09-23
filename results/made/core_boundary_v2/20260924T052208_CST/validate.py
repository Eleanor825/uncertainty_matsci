from pathlib import Path
from collections import Counter
import hashlib,json,math
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_bytes())
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-12)
m=read('manifest.json');assert set(m['files'])=={p.name for p in P.iterdir()if p.is_file()and p.name!='manifest.json'}
for n,pin in m['files'].items():
 b=(P/n).read_bytes();assert len(b)==pin['bytes']and hashlib.sha256(b).hexdigest()==pin['sha256']
s=read('snapshot.json');pairs=s['pairs'];rows=[r for p in pairs for r in p['arms'].values()];assert len(pairs)==60 and len(rows)==len({r['job_id']for r in rows})==120
assert Counter(r['state']for r in rows)==s['all_status_counts']=={'accepted':22,'active_or_claimed':2,'pending':96}
assert Counter(r['state']for r in rows if r['origin']=='new_v2')==s['new_status_counts']=={'accepted':18,'active_or_claimed':2,'pending':96}
assert sum(r['origin']=='fixed_reuse'and r['accepted']for r in rows)==4
complete=[p for p in pairs if p['accepted_pair']];assert len(complete)==s['complete_pairs']==11
for p in pairs:
 assert p['accepted_pair']==all(r['accepted']for r in p['arms'].values())
 if not p['accepted_pair']:assert p['ActionPool2_minus_Native1']is None;continue
 for k in('SUN','AUDC','mSUN'):assert close(p['ActionPool2_minus_Native1'][k],p['arms']['ActionPool2']['metrics'][k]-p['arms']['Native1']['metrics'][k])
for g in s['by_budget_and_system']:
 ps=[p for p in complete if p['budget']==g['budget']and(g['task_id']is None or p['task_id']==g['task_id'])];assert len(ps)==g['complete_pairs']
 for k,v in g['metrics'].items():
  if not ps:assert v['Native1_mean']is v['ActionPool2_mean']is v['mean_paired_difference']is None;continue
  for arm in('Native1','ActionPool2'):assert close(v[arm+'_mean'],sum(p['arms'][arm]['metrics'][k]for p in ps)/len(ps))
  ds=[p['ActionPool2_minus_Native1'][k]for p in ps];assert close(v['mean_paired_difference'],sum(ds)/len(ds))
  assert(v['wins'],v['ties'],v['losses'])==(sum(x>1e-12 for x in ds),sum(abs(x)<=1e-12 for x in ds),sum(x<-1e-12 for x in ds))
assert s['native_device_observations']==[] and s['active_status_is_metadata_not_fresh_process_proof']
assert all(not v['directory_exists']for v in s['shared_resource_metadata'].values())and s['shared_resource_metadata_is_not_GPU_liveness']
assert not s['statistical_significance_claimed']and not s['uncertainty_net_increment_established']and not s['incomplete_pair_metrics_imputed']and not s['later_results_merged']
for c in s['seed501_only_Common2_checks']:
 assert c['seed']==501 and not c['Common2_uses_internal_risk']
 for k in('SUN','AUDC'):assert close(c['difference'][k],c['ActionPool2']['metrics'][k]-c['Common2']['metrics'][k])
assert m['new_scientific_calls']==s['new_scientific_calls_for_publication']==0
print(json.dumps({'status':'passed','accepted':22,'complete_pairs':11,'new_accepted':18,'reused':4,'active_or_claimed':2,'pending':96,'new_scientific_calls':0},sort_keys=True))
