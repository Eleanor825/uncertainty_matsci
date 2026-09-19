"""Pure local arithmetic, no model, network, oracle, or refitting.

python3 -B recompute.py --write  # regenerate derived CSV/TeX/JSON
python3 -B recompute.py          # verify exact derived bytes
"""
from collections import defaultdict
from pathlib import Path
from statistics import mean, variance, stdev
import argparse
import csv
import hashlib
import io
import json
import math

ROOT = Path(__file__).resolve().parent
ARMS = [('random', 'Random'), ('gp_ei_scalarized', 'GP-EI (fixed scalarization)'),
        ('qwen_base', 'Qwen baseline'), ('uq_only', 'UQ-only'),
        ('es_only', 'Independent ES-only (G0)'), ('random_controller', 'Random controller'),
        ('uq_esopt', 'Full: UQ + ES (G0)')]
METRICS = ('final_hv_gain', 'mean_querywise_hv_gain')
SEEDS = list(range(5101, 5106))

def data(name):
    return json.loads((ROOT/'inputs'/name).read_text())

def rows(name):
    with (ROOT/'inputs'/name).open(newline='') as f:
        return list(csv.DictReader(f))

def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-22), (a,b)

def csv_bytes(values):
    s=io.StringIO(newline='')
    w=csv.DictWriter(s,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    return s.getvalue().encode()

def derive():
    for ref in json.loads((ROOT/'input_manifest.json').read_text())['inputs']:
        b=(ROOT/ref['local_input']).read_bytes()
        assert len(b)==ref['bytes'] and hashlib.sha256(b).hexdigest()==ref['sha256']
    per_seed=rows('per_seed.csv'); by={(r['arm'],int(r['seed'])):r for r in per_seed}
    assert len(by)==len(per_seed)==35
    assert set(by)=={(a,s) for a,_ in ARMS for s in SEEDS}
    protocol=data('protocol.json');summary=data('summary.json');selection=data('prior_and_selection.json')
    assert summary['test_oracle_calls']==1750 and summary['distinct_physical_oracle_calls']==2300
    assert summary['adaptation_oracle_calls']==550 and summary['failed_or_unknown_physical_attempts']==0
    assert protocol['evaluation']['prefix_reports_are_independent_runs'] is False
    assert len({r['prior_fingerprint'] for r in per_seed})==1
    curve=defaultdict(list)
    for r in rows('curves.csv'):curve[r['name']].append(r)
    assert sum(map(len,curve.values()))==1750
    for r in per_seed:
        ps=curve[r['name']]; assert [int(x['query_index']) for x in ps]==list(range(1,51))
        hv=[float(x['HV']) for x in ps];prior=float(r['prior_hv'])
        assert prior==selection['prior_HV'] and int(r['prior_rows'])==550
        assert int(r['queries'])==int(r['budget'])==50
        close(r['final_hv_gain'],hv[-1]-prior)
        # Average individual gains avoids cancellation in the published tiny gains.
        close(r['mean_querywise_hv_gain'],mean(x-prior for x in hv))
    original_stats={(r['arm'],r['metric']):r for r in rows('statistics.csv')}
    table=[]
    for arm,label in ARMS:
        out={'arm':arm,'method':label,'n_evaluation_seeds':5}
        for metric in METRICS:
            values=[float(by[arm,s][metric]) for s in SEEDS]
            for name,value in [('mean',mean(values)),('sample_variance_ddof1',variance(values)),('SD',stdev(values))]:
                close(value,original_stats[arm,metric][name])
                out[metric+'_'+name]=value
                out[metric+'_'+name+'_scaled']=value*(1e8 if name=='sample_variance_ddof1' else 1e4)
        table.append(out)
    paired=[]
    for seed in SEEDS:
        d={'seed':seed}
        for arm,_ in ARMS:
            for metric in METRICS:d[arm+'_'+metric]=float(by[arm,seed][metric])
        for metric in METRICS:d['full_minus_base_'+metric]=d['uq_esopt_'+metric]-d['qwen_base_'+metric]
        paired.append(d)
    pair_stats={}
    for metric in METRICS:
        vals=[r['full_minus_base_'+metric] for r in paired]
        pair_stats[metric]={'mean_delta':mean(vals),'sample_variance_ddof1':variance(vals),
                            'SD':stdev(vals),'wins':sum(v>1e-12 for v in vals),
                            'ties':sum(abs(v)<=1e-12 for v in vals),'losses':sum(v< -1e-12 for v in vals)}
    for seed in SEEDS:
        for a,b in [('uq_esopt','uq_only'),('es_only','qwen_base')]:
            assert by[a,seed]['weight_hash']==by[b,seed]['weight_hash']
            assert [x['HV'] for x in curve[by[a,seed]['name']]]==[x['HV'] for x in curve[by[b,seed]['name']]]
    full=selection['full_checkpoint_selection'];es=data('es_only_development.json')
    for s in [full,es['selection']['value']]:
        assert s['selected']['generation']==0 and s['test_used'] is False
        assert s['selected']['weight_hash']==s['development'][0]['weight_hash']
        assert s['selected']==max(s['development'],key=lambda x:(x['fitness'],-x['generation']))
    assert len({x['weight_hash'] for x in full['development']})==3
    assert es['updates'][0]['nonzero_parameter_delta_count']==0
    assert es['updates'][1]['nonzero_parameter_delta_count']==723
    tex=[r'\begin{table*}[t]',r'\centering',r'\small',
         r'\caption{Completed Summit SnAr evaluation with Qwen3.5-4B: seven arms, five paired evaluation seeds, 50 oracle queries per trajectory. Gains are relative to the same 550-query adaptation prior (HV $0.8810117795$). Means and standard deviations are in $10^{-4}$ HV units; sample variances are in $10^{-8}$ HV$^2$ units ($n=5$, ddof$=1$). The ES checkpoint selection returned G0 in both branches.}',
         r'\label{tab:snar-main}',r'\begin{tabular}{lrrrr}',r'\toprule',
         r'& \multicolumn{2}{c}{Final HV gain $\uparrow$} & \multicolumn{2}{c}{Mean querywise HV gain $\uparrow$} \\',
         r'Method & Mean $\pm$ SD & Sample variance & Mean $\pm$ SD & Sample variance \\',r'\midrule']
    for row in table:
        vals=[]
        for metric in METRICS:
            vals += [f"${row[metric+'_mean_scaled']:.4f} \\pm {row[metric+'_SD_scaled']:.4f}$",
                     f"{row[metric+'_sample_variance_ddof1_scaled']:.4f}"]
        tex.append(row['method']+' & '+' & '.join(vals)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    results={'schema':'SnAr_paper_table_from_existing_35_v1','complete_heldout_trajectories':35,
             'seed_axis':'evaluation_seeds_5101_to_5105_conditional_on_one_fixed_adaptation',
             'metrics_units':'unscaled_normalized_hypervolume; table means/SD times 1e4 and variance times 1e8',
             'common_prior_hv':selection['prior_HV'],'adaptation_calls':550,'heldout_calls':1750,
             'total_distinct_physical_calls':2300,'new_physical_model_or_fit_calls':0,
             'paired_full_minus_baseline':pair_stats,'selected_generation':{'full':0,'es_only':0},
             'full_changed_G1_and_G2_weights':True,'es_only_G1_update_nonzero_tensor_count':0,
             'es_only_G2_update_nonzero_tensor_count':723,
             'full_equals_uq_curves':True,'es_only_equals_baseline_curves':True}
    return {'table.csv':csv_bytes(table),'per_seed.csv':csv_bytes(paired),
            'table.tex':'\n'.join(tex).encode(),'summary.json':(json.dumps(results,indent=2)+'\n').encode()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    derived=derive()
    for name,b in derived.items():
        if args.write:(ROOT/name).write_bytes(b)
        else: assert (ROOT/name).read_bytes()==b,name
    if not args.write and (ROOT/'manifest.json').exists():
        for ref in json.loads((ROOT/'manifest.json').read_text())['files']:
            b=(ROOT/ref['path']).read_bytes(); assert len(b)==ref['bytes'] and hashlib.sha256(b).hexdigest()==ref['sha256']
    print(json.dumps({'passed':True,'trajectories':35,'curve_points':1750,'metric_mean_variance_sd_sets':14,
                      'derived_files_byte_exact':list(derived),'new_science_calls':0}))

if __name__=='__main__':main()
