"""Validate only this derived B10 export; no experiment imports, network or writes."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ARMS = ('Native1','NoGraphRisk','ExplicitRepeatRisk','ExplicitRepeatCommon2','ExplicitRepeatInternal2')
SEEDS = (336,337)
PAIRS = (('Native1','NoGraphRisk'),('NoGraphRisk','ExplicitRepeatRisk'),('Native1','ExplicitRepeatRisk'),('ExplicitRepeatCommon2','ExplicitRepeatInternal2'))
PRIMARY = {('Native1','ExplicitRepeatRisk'),('ExplicitRepeatCommon2','ExplicitRepeatInternal2')}
METRICS = ('score_normalized','F','failure_count','failure_fraction','agent_attempts','generated_proposals','NoGraph_captures','NoGraph_forward_intents','NoGraph_forward_returns','consecutive_failed_action_repeats')

def need(ok, message):
    if not ok:
        raise ValueError(message)

def read_csv(name):
    with (HERE/name).open(newline='') as stream:
        return list(csv.DictReader(stream))

def near(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)

def check_stats(actual, values):
    need(int(actual['n'])==len(values)==2,'Two-seed denominator differs')
    expected={'mean':statistics.mean(values),'sample_variance_ddof1':statistics.variance(values),
              'sample_std_ddof1':statistics.stdev(values),'min':min(values),'max':max(values)}
    for key,value in expected.items():
        need(near(actual[key],value),'Seed statistic differs: '+key)

def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    need(manifest['schema']=='dw_B10_closed_public_export_manifest_v1','Wrong manifest schema')
    names=set(manifest['files'])
    need(names=={p.name for p in HERE.iterdir() if p.is_file()}-{'manifest.json'},'Unexpected or missing file')
    for name,pin in manifest['files'].items():
        path=HERE/name
        need(path.parent==HERE and not path.is_symlink(),'Foreign export path')
        raw=path.read_bytes()
        need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Export hash differs: '+name)
    summary=json.loads((HERE/'summary.json').read_text())
    need(summary['accepted_episodes']==10 and summary['conditions']==list(ARMS) and summary['policy_seeds']==list(SEEDS),'Scope differs')
    metadata=summary['budget_extension_metadata']
    need(metadata['B30_outcomes_already_observed'] and metadata['budget_extension_selected_after_B30'] and metadata['world_previously_used'] and metadata['policy_seeds_previously_used'],'Prior exposure disclosure missing')
    need(not metadata['new_holdout_claimed'] and not summary['original_Full_equivalence_claimed'],'Invalid scope claim')
    rows=read_csv('episodes.csv')
    need([(int(r['policy_seed']),r['condition']) for r in rows]==[(s,a) for s in SEEDS for a in ARMS],'All ten seed-condition rows required')
    need(len({r['job_id'] for r in rows})==10,'Duplicate job')
    by={(int(r['policy_seed']),r['condition']):r for r in rows}
    for r in rows:
        need(r['status']=='complete' and int(r['world_seed'])==4 and int(r['budget'])==10,'Scope/closure differs')
        n=int(r['agent_attempts']);failures=int(r['failure_count'])
        need(n==10 and 0<=failures<=n and near(r['failure_fraction'],failures/n),'Failure denominator differs')
        need(int(r['action_returns'])+int(r['parser_rejections'])==n and int(r['tick_returns'])==int(r['action_returns']),'Action/tick closure differs')
        need(near(r['F'],float(r['score_normalized'])+int(r['task_completed_successfully']=='true')-.1*failures/n),'F differs')
        need(all(int(r[x])==0 for x in ('candidate_graphs','candidate_assemble_calls','candidate_backward_calls','policy_parameter_updates')),'Unexpected graph/update cost')
    counts=summary['actual_counts']
    for aggregate,key in [('environment_attempts','agent_attempts'),('generated_proposals','generated_proposals'),('NoGraph_captures','NoGraph_captures'),('NoGraph_forward_intents','NoGraph_forward_intents'),('NoGraph_forward_returns','NoGraph_forward_returns')]:
        need(sum(int(r[key]) for r in rows)==counts[aggregate],'Total cost differs')
    stats=read_csv('condition_seed_statistics.csv')
    need(len(stats)==len(ARMS)*len(METRICS),'Condition variance coverage differs')
    need({(r['condition'],r['metric']) for r in stats}=={(a,m) for a in ARMS for m in METRICS},'Condition statistic identity differs')
    for r in stats:
        check_stats(r,[float(by[(s,r['condition'])][r['metric']]) for s in SEEDS])
    pairs=read_csv('paired_contrasts.csv')
    need([(int(r['policy_seed']),r['left'],r['right']) for r in pairs]==[(s,a,b) for s in SEEDS for a,b in PAIRS],'Paired coverage differs')
    for r in pairs:
        seed=int(r['policy_seed']);left=by[(seed,r['left'])];right=by[(seed,r['right'])]
        need(r['direction']=='right_minus_left' and (r['registered_primary']=='true')==((r['left'],r['right']) in PRIMARY),'Contrast interpretation differs')
        for metric in METRICS:
            need(near(r[metric+'_delta'],float(right[metric])-float(left[metric])),'Contrast value differs')
    paired_stats=read_csv('paired_seed_statistics.csv')
    need(len(paired_stats)==len(PAIRS)*len(METRICS),'Paired variance coverage differs')
    need({(r['left'],r['right'],r['metric']) for r in paired_stats}=={(a,b,m+'_delta') for a,b in PAIRS for m in METRICS},'Paired statistic identity differs')
    for r in paired_stats:
        need((r['registered_primary']=='true')==((r['left'],r['right']) in PRIMARY),'Paired primary label differs')
        check_stats(r,[float(x[r['metric']]) for x in pairs if (x['left'],x['right'])==(r['left'],r['right'])])
    provenance=json.loads((HERE/'provenance.json').read_text())
    need(len(provenance['episodes'])==10 and len(provenance['closed_groups'])==6,'Provenance coverage differs')
    need(provenance['registration']==summary['registration'],'Registration differs')
    text=(HERE/'report.md').read_text()
    need(all(phrase in text for phrase in ('not a new holdout','not the original Full','ddof=1','No condition or negative result was excluded')),'Report scope boundary missing')
    need(summary['new_scientific_calls']=={'LLM':0,'NN':0,'environment':0,'training':0,'graphs':0,'GPU_queries':0},'Export claims new science')
    need(all(float(r['score_normalized'])==0 and r['task_completed_successfully']=='false' for r in rows),'Expected observed zero-score readout differs')
    need(int(by[(337,'ExplicitRepeatInternal2')]['failure_count'])-int(by[(337,'ExplicitRepeatCommon2')]['failure_count'])==4,'Negative fixed-two result lost')
    print(json.dumps({'passed':True,'episodes':len(rows),'conditions':len(ARMS),'policy_seeds':len(SEEDS),
        'paired_rows':len(pairs),'condition_statistics':len(stats),'paired_statistics':len(paired_stats),
        'export_files_hash_checked':len(names),'actual_counts':counts,'all_scores_zero':True,
        'negative_fixed_two_seed337_preserved':True,'new_scientific_calls':0},sort_keys=True))

if __name__=='__main__':main()
