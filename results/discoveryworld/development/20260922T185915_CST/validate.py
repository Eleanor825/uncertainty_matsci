"""Portable pair, action-change, closure and graph-count verification."""
from pathlib import Path
import hashlib,json,math
H=Path(__file__).resolve().parent
def read(n):return json.loads((H/n).read_text())
def main():
    manifest=read('manifest.json')
    for f in manifest['files']:
        b=(H/f['path']).read_bytes();assert len(b)==f['bytes']and hashlib.sha256(b).hexdigest()==f['sha256']
    p=read('closed_pair.json');s=read('summary.json');assert p['status']=='complete'and p['policy_parameter_updates']==p['test_episodes']==0
    assert p['actual_policy_loads']==p['actual_full_eight_prefix_qualifications']==1 and p['Posterior05_supported_activity_gate_passed']is True
    assert p['calibration_or_threshold_fitting']is False and p['Full_ES_registered_or_executed']is False
    assert len(p['results'])==2 and len({r['actual_G0_tensor_hash_after']for r in p['results']})==1
    by={r['job']['operating_point']:r for r in p['results']};assert set(by)=={'Q75','Posterior05'}
    assert [r['executed_action']for r in by['Q75']['action_change_and_repetition']['rows']]==[r['executed_action']for r in by['Posterior05']['action_change_and_repetition']['rows']]
    for name,threshold,rev,rank,graphs in [('Q75',.9867005323054212,3,2,13),('Posterior05',.5,9,7,19)]:
        r=by[name];j=r['job'];f=r['fitness'];e=r['result'];rows=r['action_change_and_repetition']['rows'];st=next(x for x in s['conditions']if x['condition']==name)
        assert r['accepted']is True and r['supported_activity_gate_passed']is True and e['accepted']is True and e['completed']is True and e['failure']is None
        closure=e['environment_process_closure'];assert closure['closed_response']is True and closure['returncode']==0 and closure['child_still_running']is False and closure['unknown_connection']is False
        assert j['world_seed']==2 and j['policy_seed']==304 and j['max_agent_attempts']==10 and j['generation']==0
        assert e['attempts']==f['agent_attempts']==f['failure_count']==10 and f['failure_fraction']==1 and f['scoreNormalized']==0 and f['completedSuccessfully']is False
        assert math.isclose(f['F'],-.1)and f['J']==f['F']and f['lambda_cal']==0
        assert r['controller_provenance']['effective_primary_threshold']==threshold and r['controller_provenance']['temperature_or_NN_changed']is False
        assert e['controller_counts']=={'supported_neural_revisions':rev,'supported_rank_changes':rank}
        assert len(rows)==10 and [x['attempt_index']for x in rows]==list(range(1,11))
        assert sum(x['candidate_count']for x in rows)==graphs==st['generated_proposals']==st['candidate_graph_returns']
        assert sum(x['candidate_count']==2 for x in rows)==rev
        run=longest=0;last=None;repeats=0
        for x in rows:
            assert x['official_or_parser_failure']is True
            assert x['executed_action_changed_from_first_proposal']==(x['executed_action']!=x['first_action'])
            assert not x['generated_revision_action_different']and not x['revision_generated_but_unparsed']
            assert x['revision_action']is None or x['revision_action']==x['first_action']
            same=last is not None and x['executed_action']==last;assert same==x['repeats_previous_failed_action']
            repeats+=same;run=run+1 if same else 1;longest=max(longest,run);last=x['executed_action']
        assert longest==9 and repeats==8 and st['actual_action_changes']==st['different_second_actions']==0
        assert r['qualification']==p['shared_qualification']
        assert f['source_events']==r['action_change_and_repetition']['environment_events']and f['source_summary']==e['environment_summary']
    expected={k:s['conditions'][1][k]-s['conditions'][0][k]for k in s['Posterior05_minus_Q75']};assert expected==s['Posterior05_minus_Q75']
    assert all(expected[k]==0 for k in ('failures','score','F','actual_action_changes','different_second_actions'))
    assert expected['generated_proposals']==expected['candidate_graph_returns']==6
    assert s['not_Full_versus_Native']is True and s['newly_closed_since_previous_export']==1
    live=read('related_progress.json');assert live['Full_ES_v5']['qualification_complete']is False and live['Full_ES_v5']['ES_parameter_updates_confirmed']==0
    assert live['Internal2_p305']['status']=='partial_not_final'and live['Internal2_p305']['final_fitness']is None
    for path in H.iterdir():
        if path.suffix not in ('.md','.json','.csv'):continue
        t=path.read_text();assert not any(x in t for x in ('/mnt/','/Users/','"child_pid"','"pid"','"hostname"','"host"','"password"'))
    print(json.dumps({'status':'PASS','paired_conditions':2,'action_rows':20,'failure_score_fitness_differences':0,'candidate_graph_returns':[13,19],
        'effective_action_changes':[0,0],'new_scientific_calls':0,'validated_files':len(manifest['files'])}))
if __name__=='__main__':main()
