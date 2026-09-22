"""Portable check of this derived snapshot only; no historical scan/science."""
from pathlib import Path
import csv,hashlib,json,math
HERE=Path(__file__).resolve().parent
m=json.loads((HERE/'manifest.json').read_bytes())
for name,pin in m['files'].items():
 p=HERE/name;assert p.stat().st_size==pin['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256']
s=json.loads((HERE/'snapshot.json').read_bytes());t=s['TRAIN36'];f=t['utility_fit'];assert sum(x['closed_windows']for x in t['groups'])==36
assert sum(x['positive_task_gain_windows']for x in t['groups'])==4 and f['support']['positive_task_gain_sources']==['102']
assert f['status']=='not_run_support'and not f['models_sealed']and all(v==0 for v in f['counts'].values())
assert t['CPU_accounting_totals']=={'bootstrap_ticks':36,'environment_initializations':36,'prefix_replay_actions':834,'window_action_attempts':288}
c=s['p332']['conditions'];assert[(x['status'],x['observed_attempts'],x['observed_failures'],x['final_task_score'])for x in c]==[('closed',30,29,0),('closed',30,9,.125),('pending',12,11,None)]
assert c[2]['final_task_success']is None and c[2]['final_failure_fraction']is None and c[2]['final_F']is None
assert all(not x['final_task_success']for x in c[:2]);assert not s['p332']['study_complete']and not s['p332']['resource_complete']
rows=list(csv.DictReader((HERE/'p332_status.csv').open()));assert rows[2]['final_task_score']==rows[2]['final_F']==rows[2]['final_task_success']==''
assert s['prospective_status']['NoGraph_preflight']['qualification_prefix_results']==8 and not s['prospective_status']['p335']['environment_phase_started']
assert s['new_scientific_calls_for_publication']==0
print(json.dumps({'passed':True,'TRAIN_windows':36,'utility_fits':0,'utility_updates':0,'p332_closed_arms':2,'p332_pending_arms':1,'new_scientific_calls':0},sort_keys=True))

a=json.loads((HERE/'gate_audit.json').read_bytes());assert len(a['rows'])==36 and sum(r['R8']==0 for r in a['rows'])==32
positive=[r for r in a['rows']if r['positive_R8']];assert len(positive)==4 and sum(r['below_0_5']for r in positive)==2 and all(r['source_episode']=='102'for r in positive)
assert a['counts']['NN_inference_rows']==36 and a['counts']['NN_inference_batches']==1 and a['counts']['NN_updates']==a['counts']['LLM_forward_calls']==a['counts']['environment_calls']==0
assert len(list(csv.DictReader((HERE/'gate_audit_all36.csv').open())))==36
n=s['prospective_status']['NoGraph_preflight'];assert len(n['case_comparisons'])==3 and n['case_parities_all_passed']and n['final_completion_available']and n['resource_closed']
print(json.dumps({'gate_audit_all_rows':36,'zero_R8_rows':32,'positive_R8_rows':4,'positive_below_05':2,'NoGraph_TRAIN_case_checks_passed':3,'p335_online_outcomes':0},sort_keys=True))
