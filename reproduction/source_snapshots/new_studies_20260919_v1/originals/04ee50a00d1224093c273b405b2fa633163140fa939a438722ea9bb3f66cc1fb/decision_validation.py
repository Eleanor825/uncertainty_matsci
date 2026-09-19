"""CPU support-contract evidence checks; no inference, fitting or oracle calls."""
from matdiscovery.support_control import support_for_row
from matdiscovery.failure_labels import HEADS
from matdiscovery.support_scope import require

def validate_support_decisions(rows):
    require(bool(rows),'Missing actual support-aware decisions')
    counts={'proposals':len(rows),'supported':0,'unknown':0}
    for row in rows:
        support=support_for_row(row)
        require(row['uncertainty_support']==support,'Changed graph-support evidence')
        diagnostic=row['diagnostic_risk_prediction']
        require(diagnostic['decision_eligible']==support['supported']
                and diagnostic['scalar_is_NN_prediction']==row['generation']['success'],'Prediction scope changed')
        plan=row['failure_controller']
        if support['supported']:
            counts['supported']+=1
            require(row['predicted_failure_probability']==diagnostic['scalar']
                    and row['failure_type_probabilities']==diagnostic['types'],'Supported probabilities changed')
        else:
            counts['unknown']+=1
            require(row['predicted_failure_probability'] is None
                    and row['failure_type_probabilities']==dict.fromkeys(HEADS)
                    and row.get('confidence_used_for_control') is False
                    and row.get('failure_types_used_for_control') is False,'Unsupported NN inference used as risk')
            require(plan.get('uncertainty_support')==support and plan['trigger'] not in ('risk_unavailable','overall_risk')
                    and plan.get('trigger_source')!='predicted_failure_type','Unknown graph triggered learned-risk feedback')
            require(not plan['request_retry'] or (plan.get('trigger_source')=='local_schema_validation'
                    and bool(plan.get('local_schema_issues'))),'Unknown support requested a retry')
        selection=row.get('failure_candidate_selection')
        if selection and 'uncertainty_supports' in selection:
            require(selection['common_comparison_heads']==[] and selection['type_signals_used'] is False
                    and selection['ranking_rule']=='valid_first; original_index_when_any_native_support_unknown',
                    'Unknown support was ranked by risk')
            ranking=selection['ranking']
            require(ranking==sorted(ranking,key=lambda r:(not r['valid'],r['candidate_index']))
                    and all(r['rank_key']==[int(not r['valid']),r['candidate_index']] for r in ranking),
                    'Unknown support acquired a ranking preference')
    return counts
