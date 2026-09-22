"""Portable terminal-state arithmetic and provenance checks; no scientific calls."""
from pathlib import Path
import hashlib,json,math
H=Path(__file__).resolve().parent
def read(n):return json.loads((H/n).read_text())
def main():
    m=read('manifest.json')
    for f in m['files']:
        b=(H/f['path']).read_bytes();assert len(b)==f['bytes']and hashlib.sha256(b).hexdigest()==f['sha256']
    p=read('terminal_source_projection.json');s=read('summary.json')
    assert p['completion.json']['status']=='complete'and p['completion.json']['schema']=='dw_frozen_matched_proposal_development_pair_complete_v5'
    assert p['completion.json']['policy_parameter_updates']==0 and p['qualification_passed']==8 and p['qualification_failed']==0
    for name,fail in [('Common2',1),('Internal2',2)]:
        c=p[name];f=c['fitness'];j=next(x for x in p['jobs']if '-'+name+'-'in x['id']);r=next(x for x in s['conditions']if x['condition']==name)
        assert c['accepted']is True and j['id']=='dw-matched-proposal-v5-dev-'+name+'-w2-p305-b10-g0'
        assert j['selected_attempts']==j['action_returns']==j['ticks_returned']==f['agent_attempts']==10
        assert j['failed_action_returns']==f['failure_count']==fail and j['graph_returns']==20
        assert f['scoreNormalized']==0 and f['completedSuccessfully']is False and f['lambda_cal']==0
        assert math.isclose(f['failure_fraction'],fail/10)and math.isclose(f['F'],-.1*fail/10)and f['J']==f['F']
        assert r['failed_actions']==fail and r['successful_actions']==10-fail and r['candidate_graph_returns']==20
        assert len(f['source_events']['sha256'])==len(f['source_summary']['sha256'])==64
    assert s['world_seed']==2 and s['policy_seed']==305 and s['budget']==10 and s['paired_cases']==1
    d=s['Internal2_minus_Common2'];assert d['failed_actions']==1 and math.isclose(d['failure_fraction'],.1)and math.isclose(d['F'],-.01)and d['task_score']==d['candidate_graph_returns']==0
    assert s['Full_vs_Native']is False and s['pooled_with_p304']is False and s['previous_Common2_reused']is True and s['newly_closed_since_previous_export']==1
    for path in H.iterdir():
        if path.suffix not in ('.json','.md','.csv'):continue
        t=path.read_text();assert not any(x in t for x in ('/mnt/','/Users/','"pid"','"host"','"hostname"','"password"'))
    print(json.dumps({'status':'PASS','completed_matched_conditions':2,'paired_cases':1,'failed_actions':[1,2],'task_scores':[0,0],
        'candidate_graph_returns':[20,20],'new_scientific_calls':0,'validated_files':len(m['files'])}))
if __name__=='__main__':main()
