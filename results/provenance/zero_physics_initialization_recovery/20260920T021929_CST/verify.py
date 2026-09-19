from pathlib import Path
import collections,hashlib,json
R=Path(__file__).resolve().parent
d=json.loads((R/'scientific_provenance.json').read_text());rows=d['audit']['rows']
assert len(rows)==23 and len({r['job_id'] for r in rows})==23
assert collections.Counter(r['arm'] for r in rows)=={'baseline_reference':4,'es_only_independent':5,'uq_only_support_aware':8,'full_support_aware':6}
assert collections.Counter(r['budget'] for r in rows)=={10:13,30:10}
assert sum(r['budget'] for r in rows)==430
for r in rows:
 assert r['eligible'] and all(r['checks'].values()) and r['classification']=='confirmed_pre_execution_missing_asset'
 assert r['initialization_requests']==1 and r['physical_oracle_attempts_proved']==r['tool_requests']==r['step_requests']==0
 assert not r['failed_checks'] and r['error']['code']=='missing_asset'
 assert r['evidence']['claim']['sha256'] and r['evidence']['failure']['sha256'] and r['closed_phase_exit_sha256']
apply=d['recovery']['driver_apply_receipt.json']['value']
assert apply['complete'] and apply['archived_jobs']==23 and apply['direct_scientific_calls']==apply['new_claims_observed']==apply['new_receipts_observed']==0
assert apply['original_jobs_not_declared_scientifically_complete'] is True
assert d['recovery']['batch_requeued.json']['sha256']=='0878ce48531ff9b482e5c45e2e24e28f0c63efab344459f57e7309d96b3c5dfa'
for item in json.loads((R/'manifest.json').read_text())['files']:
 b=(R/item['path']).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
print(json.dumps({'passed':True,'audited_original_init_failures':23,'old_physical_calls':0,'original_candidate_budget':430,'new_scientific_calls':0,'not_a_live_process_or_raw_RPC_reaudit':True}))
