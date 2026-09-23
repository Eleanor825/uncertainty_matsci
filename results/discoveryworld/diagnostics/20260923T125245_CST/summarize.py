"""Descriptive n=2 sampling-seed statistics from the immutable exported snapshot."""
import statistics
ARMS = ('Native1', 'NoGraphRisk', 'ExplicitRepeatRisk', 'ExplicitRepeatCommon2', 'ExplicitRepeatInternal2')
METRICS = ('scoreNormalized', 'failure_count', 'failure_fraction', 'F')
PAIRS = (('Native1', 'NoGraphRisk'), ('NoGraphRisk', 'ExplicitRepeatRisk'), ('Native1', 'ExplicitRepeatRisk'), ('ExplicitRepeatCommon2', 'ExplicitRepeatInternal2'), ('Native1', 'ExplicitRepeatCommon2'), ('Native1', 'ExplicitRepeatInternal2'))
def summarize(d):
    assert d['all_ten_complete'] and d['publication_ready']
    a = {(r['policy_seed'], r['condition']): r['metrics'] for r in d['arms']}
    def stats(values):
        return {'n': 2, 'values_by_seed': dict(zip(('336', '337'), values)), 'mean': statistics.mean(values), 'sample_variance': statistics.variance(values), 'sample_SD': statistics.stdev(values)}
    arm = [{'condition': name, 'metric': metric, **stats([a[(s, name)][metric] for s in (336, 337)])} for name in ARMS for metric in METRICS]
    pairs = [{'right_minus_left': right + ' - ' + left, 'metric': metric, 'contrast_scope': 'registered paired contrast' if (left, right) in PAIRS[:4] else 'supplementary descriptive comparison to Native1', **stats([a[(s, right)][metric] - a[(s, left)][metric] for s in (336, 337)])} for left, right in PAIRS for metric in METRICS]
    return {'schema': 'dw_world4_complete_descriptive_statistics_v1', 'world_seed': 4, 'policy_seeds': [336, 337], 'independent_worlds': 1, 'sample_variance_denominator': 'n - 1 = 1', 'scope': 'descriptive sampling-seed statistics; no inferential significance or cross-world generalization', 'old_unseeded_p335_included': False, 'original_Full_included': False, 'arms': arm, 'paired_contrasts': pairs}
