"""Validate and render this preparatory DW export; no scientific/network calls."""
from pathlib import Path
import hashlib,json,math

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def report(d):
    assert d['schema']=='discoveryworld_public_progress_v1'
    assert d['public_export_fingerprint']==digest({k:v for k,v in d.items()if k!='public_export_fingerprint'})
    assert d['scientific_calls_for_export']==0 and d['effectiveness_demonstrated']is False
    assert d['held_out_paired_tests']==[] and not d['NN_fit_complete'] and not d['Full_ES_complete']
    native=d['native_collection'];assert len(native)==8
    assert {(r['job']['world_seed'],r['job']['policy_seed'])for r in native}=={(w,s)for w in(0,1)for s in(101,102,103)}|{(2,201),(2,202)}
    assert all(r['accepted'] and not r['failure_record_present'] and r['attempts']==100 for r in native)
    for r in native:
        assert 0<=r['final_score_normalized']<=1 and type(r['completed_successfully'])is bool
    assert len({r['job']['job_id']for r in d['legacy_ES_only']})==len(d['legacy_ES_only'])
    for r in d['legacy_ES_only']:
        j=r['job'];m=r['metrics']
        assert r['accepted'] and j['arm']=='ES_only' and j['stage']in('es_train','es_dev')
        assert j['max_agent_attempts']==100 and 0<m['agent_attempts']<=100
        assert math.isclose(m['failure_fraction'],m['failure_count']/m['agent_attempts'],abs_tol=1e-12)
        assert math.isclose(m['F'],m['scoreNormalized']+int(m['completedSuccessfully'])-.1*m['failure_fraction'],abs_tol=1e-12)
        assert m['J']==m['F']
    assert d['valid_subset_graph_rows']==d['written_subset_graph_records']==0
    for g in d['graph_groups']:
        assert not g['complete']
        if g['failed']:assert g['error_type']=='UnsupportedAttribution' and 'FVU=0.843559980392456' in g['message']
    lines=['# DiscoveryWorld: preparatory results and incomplete comparison','',
        f"Observed **{d['observed_UTC']}**. Model: **Qwen3.5-4B**; scenario: **Proteomics Normal**.",'',
        '**There is no completed held-out Baseline-versus-Full comparison and no demonstrated method improvement in this export.**','',
        '| Stage | Completed | Interpretation |','|---|---:|---|',
        '| Native data collection | 8/8 episodes, 800 action attempts | Training/calibration/development sources only |',
        f"| Earlier ES-only branch | {len(d['legacy_ES_only'])}/7 episodes | Separate B100 branch, not the Full method |",
        '| Valid subset graph records | 0/160 | Cross-domain bank fidelity failed |',
        '| Risk NN fit | 0 | Not started |','| Subset Full ES branch | 0/7 episodes | Not started |',
        '| Paired held-out evaluation | 0/2 episodes | Baseline and Full, B30; pending |','',
        'The intended subset retains internal graph features, a trained/calibrated failure-risk NN, bounded proposal revisions, and full-parameter Agentic ESOpt. Its Full branch has two ES generations, population two, four B10 candidate trajectories and three B10 development trajectories including G0. The held-out pair uses world seed 3 and policy seed 401. Extra seeds, other worlds and the larger study are deferred.','',
        '## Failed bank transfer','',
        'The MADE-trained transcoder bank did not pass the original DiscoveryWorld qualification gate: layer 0 output FVU was **0.843559980392456**, above the unchanged **0.5** limit. No successful graph was produced. These attempts share the same bank and fixed prefix suite; they are not independent efficacy trials. The failed records and hashes are retained. A DW-domain bank must pass the original fidelity and attribution checks before the Full pipeline can proceed.','',
        '| Graph group | Claimed | Failed | Complete |','|---|---|---|---|']
    for g in d['graph_groups']:lines.append(f"| {g['group']} | {g['claimed']} | {g['failed']} | {g['complete']} |")
    lines+=['','## Native source trajectories','',
        'Scores below belong to preparatory B100 collection episodes. They are not paired held-out baseline results.','',
        '| World seed | Policy seed | Role | Attempts | Final normalized score | Task success |','|---:|---:|---|---:|---:|---|']
    for r in native:
        j=r['job'];role={0:'NN fit / TC training',1:'NN calibration / TC development',2:'NN development'}[j['world_seed']]
        lines.append(f"| {j['world_seed']} | {j['policy_seed']} | {role} | {r['attempts']} | {r['final_score_normalized']} | {r['completed_successfully']} |")
    lines+=['','## Earlier ES-only preparation','',
        'These completed B100 training/development episodes are separate from the new B10 Full search. Do not pool their scores or count them as held-out Full evaluations.','',
        '| Stage | Generation | Member | World / policy seed | Attempts | Normalized score | Failure fraction | Fitness |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in d['legacy_ES_only']:
        j=r['job'];m=r['metrics'];member=j['population_member']if j['population_member']is not None else '—'
        lines.append(f"| {j['stage']} | {j['generation']} | {member} | {j['world_seed']} / {j['policy_seed']} | {m['agent_attempts']} | {m['scoreNormalized']} | {m['failure_fraction']} | {m['F']} |")
    lines+=['','## Provenance','',
        '`snapshot.json` contains a scientific-field projection of existing records, source hashes and project-relative references. No prompts, credentials, hostnames or process records are included. Acceptance here reads original completed records; it does not rerun the simulator or raw acceptance procedure. The snapshot is sequential while other work may continue.','',
        'Run `python3 -B validate.py` to verify identities, counts, metric arithmetic, file hashes and this rendered report.','']
    return '\n'.join(lines)

def main():
    root=Path(__file__).resolve().parent;d=json.loads((root/'snapshot.json').read_text())
    assert (root/'report.md').read_text()==report(d)
    for r in json.loads((root/'manifest.json').read_text())['files']:
        raw=(root/r['path']).read_bytes();assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
    print(json.dumps({'passed':True,'native_collection':8,'held_out_tests':0,'valid_graphs':0,'scientific_calls':0}))

if __name__=='__main__':main()
