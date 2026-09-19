from pathlib import Path
import json,math
p=Path(__file__).parent;v=json.loads((p/'comparison.json').read_text());rows=v['rows'];assert [r['generation'] for r in rows]==[0,1,2]
assert v['audit']['G0_acceptance']=='passed' and v['audit']['returncode']==0 and v['audit']['original_G0_evidence_unchanged']
assert v['original_selection_rewritten'] is False and v['main_1080_member'] is False
for r in rows:
 e=r['physical_evidence'];assert e['metric_name']=='AUDC' and e['metric_value']==r['AUDC'];assert e['environment_seeds']==[v['environment_seed']];assert e['costs']['candidate_oracle_attempts']==v['budget']==10
 assert e['official_commit']==rows[0]['physical_evidence']['official_commit'] and e['reward_is_llm_self_score'] is False
assert rows[0]['SUN']==6 and sum(r['new_candidate_calls_in_this_diagnostic'] for r in rows)==10
assert math.isclose(rows[1]['AUDC']-rows[0]['AUDC'],-.06,abs_tol=1e-12)
assert math.isclose(rows[2]['AUDC']-rows[0]['AUDC'],-.04,abs_tol=1e-12)
print(json.dumps({'passed':True,'AUDC':[r['AUDC'] for r in rows],'new_calls':10,'new_calls_by_recomputation':0}))
