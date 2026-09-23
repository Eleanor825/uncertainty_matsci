"""Render an authenticated readonly export locally; never publish or change scientific inputs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,json,statistics
ARMS=('Native1','NoGraphRisk','ExplicitRepeatRisk','ExplicitRepeatCommon2','ExplicitRepeatInternal2')
def render(d):
 assert d['schema']=='dw_seeded_world4_ten_arm_readonly_export_v1'and len(d['arms'])==10 and all(x==0 for x in d['scientific_calls'].values())
 rows=[]
 for seed in (336,337):
  selected=[r for r in d['arms']if r['policy_seed']==seed];assert [r['condition']for r in selected]==list(ARMS)
  for r in selected:
   m=r.get('metrics');assert (m is not None)==(r['status']=='complete')
   rows.append(f"| {seed} | {r['condition']} | {r['status']} | "+(f"{m['agent_attempts']} | {m['scoreNormalized']:.6f} | {m['failure_count']} | {m['completedSuccessfully']}"if m else f"{r.get('progress',{}).get('recorded_policy_attempts','unavailable')} recorded | — | — | —")+' |')
 table='\n'.join(rows);comparisons=[]
 for p in d['same_seed_prefix_audits']:
  comparisons.append(f"| {p['policy_seed']} | {p['left']} / {p['right']} | {p['same_action_prefix_length']} | {p['public_mismatch_after_same_prefix']} | {p['global_rng_mismatch_after_same_prefix']} |")
 aggregate=[]
 if d['all_ten_complete']:
  for a,b in (('Native1','NoGraphRisk'),('NoGraphRisk','ExplicitRepeatRisk'),('Native1','ExplicitRepeatRisk'),('ExplicitRepeatCommon2','ExplicitRepeatInternal2')):
   pairs=[r for r in d['paired_contrasts']if r['left']==a and r['right']==b];assert len(pairs)==2 and all(r['complete_pair']for r in pairs)
   for key in ('scoreNormalized','F','failure_count','failure_fraction'):
    v=[r['right_minus_left'][key]for r in pairs];aggregate.append({'right_minus_left':b+' - '+a,'metric':key,'n':2,'values_by_seed':dict(zip((336,337),v)),'mean':statistics.mean(v),'sample_variance':statistics.variance(v),'sample_SD':statistics.stdev(v)})
 observed=datetime.fromtimestamp(d['observed_unix'],timezone(timedelta(hours=8))).isoformat()
 text=f'''# Seeded world4 confirmation — local snapshot

Observed {observed}. Publication-ready: **{d['publication_ready']}**. This file does not publish anything. Until all ten arms and both resource closures are verified, treat it as progress; unavailable final metrics are not zero outcomes.

| Policy seed | Condition | Status | Actions | Final score | Failed actions | Task success |
|---|---|---|---:|---:|---:|---|
{table}

## Same-seed, same-executed-prefix audit

Every comparison is within its policy seed. After the first executed-action difference, later states are not required to match and are not tested as the same prefix. Different sampling seeds are not required to produce the same actions. Empty mismatch lists mean no mismatch among the available aligned snapshots, not complete private-state equivalence. When an adapter rejects input without a tick, the prior public UI is explicitly reused rather than treated as a new observation.

| Policy seed | Pair | Matching executed prefix length | Public UI mismatch after attempt | Recorded global RNG mismatch after attempt |
|---|---|---:|---|---|
{chr(10).join(comparisons)}

The actual sidecar records global Python RNG fingerprints and object-construction count/order fingerprints. It does **not** record current per-object, world or UUID RNG state, so this report cannot repeat the separate fixed-13 diagnostic's five-gate equality claim. Raw prompts, token IDs, private answers/task maps, model weights and activation arrays are not exported.

Closed metrics are checked against original public action-return events and official summary scalars; candidate0 prompt/token/seed identities are recorded only as hashes or seed metadata. Unsupported or inconsistent reads retain an explicit unavailable status. Positive transitions reflect action plus world tick, without labels for unexecuted candidates.

Both policy seeds use one world. Any complete two-seed mean, sample variance and SD are descriptive sampling-seed summaries; they do not establish cross-world generalization. Historical unseeded p335 and the original Full follow-up are separate. No controller or threshold is selected or changed by this export.

All exporter scientific-call counts are zero. The accompanying snapshot, source receipt hash and local manifest preserve the evidence for a later complete-results publication.
'''
 return text,aggregate

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();source=Path(a.input);raw=source.read_bytes();wrapper=json.loads(raw);d=wrapper.get('data',wrapper);text,aggregate=render(d);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 files={'report.md':text.encode(),'snapshot.json':(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+'\n').encode(),'complete_two_seed_statistics.json':(json.dumps(aggregate,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()}
 for n,b in files.items():(out/n).write_bytes(b)
 manifest={'source_receipt_sha256':hashlib.sha256(raw).hexdigest(),'source_receipt_bytes':len(raw),'exporter_source_sha256':wrapper.get('transport',{}).get('source_sha256'),'renderer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'publication_ready':d['publication_ready'],'published':False,'new_scientific_calls':0,'files':{n:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}for n,b in files.items()}}
 (out/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n');print(json.dumps({'output':str(out),'publication_ready':d['publication_ready'],'published':False},sort_keys=True))
if __name__=='__main__':main()
