"""Validate public aggregate arithmetic, provenance hashes and sanitization only."""
from pathlib import Path
import csv,hashlib,json,math,re
ROOT=Path(__file__).resolve().parent
def need(value,message):
    if not value:raise ValueError(message)
def read(name):return json.loads((ROOT/name).read_bytes())
def close(a,b):return abs(a-b)<=1e-12
def safe(value):
    forbidden={'raw_text','public_prompt_messages','input_ids_with_completion','token_ids','observation','observations','weights','hostname','gpu_uuid','process','pid','action','arg1','arg2','selected_action','submitted_action','errors','password','cookie','access_token'}
    if isinstance(value,dict):
        need(not(set(value)&forbidden),'Raw or private field in public artifact')
        for key,item in value.items():
            if key.endswith('sha256'):need(isinstance(item,str)and re.fullmatch('[0-9a-f]{64}',item),'Invalid provenance digest')
            if key in ('path','repo'):need(isinstance(item,str)and not item.startswith('/')and '..'not in Path(item).parts,'Nonportable source identifier')
            safe(item)
    elif isinstance(value,list):
        for item in value:safe(item)
    elif isinstance(value,str):
        need(not any('/'+part+'/'in value for part in ('mnt','Users','root')),'Private filesystem identifier')
        key_pattern='BEGIN'+r'[^\n]*PRIVATE'+' KEY'
        need(not re.search(r'GPU-[0-9a-fA-F]{8}-|Bearer\s+\S+|'+key_pattern,value),'Private identifier in public artifact')

manifest=read('manifest.json');need(manifest['source_audit_verified_remotely']is True,'Missing original raw-artifact audit')
names={p.name for p in ROOT.iterdir()if p.is_file()};need(names==set(manifest['files'])|{'manifest.json'},'Unexpected staged file')
for name,pin in manifest['files'].items():
    need(Path(name).name==name,'Invalid manifest name');data=(ROOT/name).read_bytes()
    need(hashlib.sha256(data).hexdigest()==pin['sha256']and len(data)==pin['bytes'],'Changed public artifact')
    if name.endswith('.json'):safe(json.loads(data))
    elif name!='validate.py':safe(data.decode())
summary=read('summary.json');metrics=summary['metrics'];need(summary['status']=='interim_single_pair_world4_p336'and summary['full_four_group_study_complete']is False,'False complete-study claim')
need(summary['study_groups']['p337-Internal']==summary['study_groups']['p337-NoGraph']=='running_not_yet_complete_at_snapshot','Missing ongoing-group status')
need(not summary['statistical_significance_claimed']and not summary['new_holdout_or_world_claimed']and not summary['unexecuted_candidate_labels_imputed'],'Unsupported inference')
need([m['arm']for m in metrics]==['Internal','NoGraph'],'Wrong paired arms')
for m in metrics:
    need(m['environment_requests']==m['actual_action_returns']==10 and m['generated_candidates']==m['full_graph_attempts']==m['full_graph_returns']==20,'Changed B10/fixed-two accounting')
    need(m['successful_full_graphs']+m['unavailable_full_graphs']==20 and m['fresh_qualification_prefixes']==8 and m['G0_unchanged']is True,'Missing capture/qualification/G0 accounting')
    need(close(m['failure_fraction'],m['action_failures']/10)and close(m['F'],m['task_score']-.1*m['failure_fraction']),'Changed original F accounting')
    need(m['task_score']==0 and m['task_completed_successfully']is False,'Task-score zero must remain explicit')
a,b=metrics;delta=summary['Internal_minus_NoGraph']
for key,field in (('F','F'),('task_score','task_score'),('action_failures','action_failures'),('failure_fraction','failure_fraction')):need(close(delta[key],a[field]-b[field]),'Paired contrast arithmetic differs')
with(ROOT/'paired_metrics.csv').open()as f:
    csv_rows=list(csv.DictReader(f));need(len(csv_rows)==2,'Wrong CSV row count')
    for row,m in zip(csv_rows,metrics):
        need(row['arm']==m['arm'],'CSV arm differs')
        for field in ('F','task_score','action_failures','environment_requests','successful_full_graphs','unavailable_full_graphs','supported_risk_comparisons','risk_rank_changes_vs_common_hash'):need(close(float(row[field]),m[field]),'CSV metric differs')
first=read('first_selection_difference.json')
for value in first.values():
    need(value['attempt_index']==1 and value['same_public_prompt_and_both_candidates']is True,'First difference evidence differs')
    for pair in value['candidate_pairs']:
        need(all(pair['equality_checks'].values())and pair['same_public_prompt_and_candidate'],'Unmatched first candidate pair')
        for field in ('public_prompt_sha256','candidate_text_sha256','action_packet_sha256','proposal_seed','token_prefix_sha256'):need(pair['Internal'][field]==pair['NoGraph'][field],'Different same-position candidate evidence')
    for arm,choice in value['choices'].items():
        risks=choice['recorded_candidate_risks'];need(len(risks)==2 and all(math.isfinite(x)and 0<=x<=1 for x in risks),'Invalid recorded risk')
        need(choice['selected_index']==min(range(2),key=lambda i:(risks[i],i)),'Recorded supported selector differs')
        need(choice['actual_selected_action_success']is True and choice['action_type']=='TELEPORT_TO_LOCATION'and choice['unexecuted_candidate_label']is None,'Actual first action or missing counterfactual scope differs')
    need(value['choices']['Internal']['selected_index']==1 and value['choices']['NoGraph']['selected_index']==0,'Expected opposite first choice missing')
with(ROOT/'selection_audit.csv').open()as f:
    rows=list(csv.DictReader(f));need([int(r['attempt_index'])for r in rows]==list(range(1,11)),'Noncontiguous selection audit')
for link in re.findall(r'\]\(([^)]+)\)',(ROOT/'README.md').read_text()):need((ROOT/link).is_file(),'Broken report link')
print(json.dumps({'status':'passed','verified_metric_rows':2,'verified_native_requests':20,'verified_full_graph_returns':40,'fresh_qualification_prefixes':16,'selection_audit_steps':10,'first_matched_candidate_pair_checked':True,'task_scores_both_zero':True,'full_study_complete':False,'new_model_or_environment_calls':0,'raw_prompts_observations_arguments_or_weights_published':False},sort_keys=True))
