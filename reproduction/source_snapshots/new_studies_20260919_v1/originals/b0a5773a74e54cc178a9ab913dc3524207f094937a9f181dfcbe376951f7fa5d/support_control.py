"""An explicit support contract around the unchanged schema-fixed controller.

No missing-graph probability is imputed. All-supported calls delegate verbatim
to the original controller; observed generation/schema validity is independent.
"""
from copy import deepcopy
import math
from .failure_controller import plan_failure_response, rank_failure_candidates
from .failure_labels import HEADS

CONTRACT = 'native_graph_support_abstention_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def support_for_row(row):
    status = row.get('graph_status')
    require(status in ('succeeded', 'unavailable'), 'Support-aware execution requires an explicit native graph status')
    value = {'contract': CONTRACT, 'supported': status == 'succeeded',
             'graph_status': status, 'reason': None if status == 'succeeded' else row.get('graph_error', 'native graph unavailable')}
    if status == 'succeeded':
        from .native_attribution import CONTRACT as NATIVE_CONTRACT
        meta = row['graph_metadata']
        require(row['graph_contract'] == NATIVE_CONTRACT and meta['backend_validation']['passed'] is True,
                'A successful graph must retain its original native/FD admission')
        require(meta['policy']['state_id'] == row['policy_stamp']['state_id']
                and meta['policy']['generation'] == row['policy_stamp']['generation']
                and meta['full_prefix_hash'] == row['prefix_hash'], 'Graph support is not the actual policy/prefix')
        fidelity = meta['fidelity']
        require(len(fidelity) == 32 and all(v['fvu_undefined'] == 0
                and type(v['output_fvu']) in (int, float) and math.isfinite(v['output_fvu'])
                and v['output_fvu'] <= .5 for v in fidelity.values()), 'Graph support lacks complete32 unchanged fidelity gates')
    return value


def apply_support(row, risk):
    """Retain actual raw inference separately; return the decision-time value."""
    support = support_for_row(row)
    row['uncertainty_support'] = support
    row['diagnostic_risk_prediction'] = {
        'scalar': row['predicted_failure_probability'],
        'types': deepcopy(row['failure_type_probabilities']),
        'decision_eligible': support['supported'],
        'scalar_is_NN_prediction': row['generation']['success'] is True}
    if not support['supported']:
        row['predicted_failure_probability'] = None
        row['failure_type_probabilities'] = {head: None for head in HEADS}
        row['confidence_used_for_control'] = False
        row['failure_types_used_for_control'] = False
        return None
    return risk


def plan_with_support(benchmark, proposed_action, risk_types, overall_risk,
                      public_observation, threshold=.6, *, support):
    require(support.get('contract') == CONTRACT and type(support.get('supported')) is bool,
            'Missing explicit uncertainty support')
    if support['supported']:
        return plan_failure_response(benchmark, proposed_action, risk_types, overall_risk, public_observation, threshold)
    require(overall_risk is None and all(risk_types.get(h) is None for h in HEADS),
            'Unsupported diagnostic predictions cannot be used for control')
    plan = plan_failure_response(benchmark, proposed_action, risk_types, overall_risk, public_observation, threshold)
    # Keep genuine observed schema failures, tool limits and pending selection.
    if plan['trigger'] == 'risk_unavailable':
        plan.update(trigger=None, controller_action='keep_candidate', request_retry=False,
                    feedback_for_retry='', optional_tool_preference=None,
                    reason='Native graph support is unknown; no learned risk comparison or retry is justified.',
                    fallback='unsupported_native_graph_neutral')
    plan['uncertainty_support'] = deepcopy(support)
    return plan


def rank_with_support(benchmark, candidates, public_observation, threshold=.6, *, supports):
    require(len(candidates) == len(supports) and all(s.get('contract') == CONTRACT
            and type(s.get('supported')) is bool for s in supports), 'Candidate support inventory differs')
    if all(s['supported'] for s in supports):
        return rank_failure_candidates(benchmark, candidates, public_observation, threshold)
    # Obtain the unchanged definition of a valid action, but disable *all*
    # cross-candidate risk comparisons when any candidate is unsupported.
    neutral = [{**c, 'risk_types': {h: None for h in HEADS}, 'overall_risk': None} for c in candidates]
    result = rank_failure_candidates(benchmark, neutral, public_observation, threshold)
    for record in result['ranking']:
        i = record['candidate_index']
        c = candidates[i]
        record['plan'] = plan_with_support(benchmark, c.get('action', c.get('proposed_action')),
            c.get('risk_types'), c.get('overall_risk'), public_observation, threshold, support=supports[i])
        record['rank_key'] = [int(not record['valid']), i]
    result['ranking'].sort(key=lambda r: r['rank_key'])
    result.update(selected_index=next((r['candidate_index'] for r in result['ranking'] if r['valid']), None),
                  ranking_rule='valid_first; original_index_when_any_native_support_unknown',
                  common_comparison_heads=[], type_signals_used=False,
                  fallback='unsupported_native_graph_neutral',
                  uncertainty_supports=deepcopy(supports), missing_probabilities_are_not_imputed=True)
    return result
