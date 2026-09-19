"""Report every preregistered seed and paired changes, including negative/null."""
import argparse
from pathlib import Path
import statistics
import math
from study import canonical,digest,read,write_once

def calibration(rows):
    values=[(r["no_hvi"],r["risk"]) for r in rows if r["risk"] is not None]
    if not values: return {"n":0,"available":False}
    n=len(values); positive=[p for y,p in values if y==1]; negative=[p for y,p in values if y==0]
    ece=0.
    for b in range(10):
        bucket=[(y,p) for y,p in values if min(9,int(p*10))==b]
        if bucket: ece+=len(bucket)/n*abs(statistics.mean(y for y,p in bucket)-statistics.mean(p for y,p in bucket))
    return {"n":n,"available":True,"label":"observed_no_hypervolume_improvement",
        "brier":statistics.mean((p-y)**2 for y,p in values),"ece_10_bins":ece,
        "nll":statistics.mean(-y*math.log(max(1e-12,min(1-1e-12,p)))-(1-y)*math.log(max(1e-12,min(1-1e-12,1-p))) for y,p in values),
        "overconfident_nonimprovement_rate_p_le_0_1":sum(y==1 and p<=.1 for y,p in values)/n,
        "auroc":sum(float(a>b)+.5*(a==b) for a in positive for b in negative)/(len(positive)*len(negative)) if positive and negative else None}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True); a=p.parse_args(); root=Path(a.output)
    protocol=read(Path(__file__).with_name("protocol.json")); data={}; summaries={}
    from acceptance import accept_study
    acceptance=accept_study(root,protocol)
    write_once(root/"acceptance.json",acceptance)
    for arm in protocol["evaluation"]["arms"]:
        data[arm]=[read(root/"episodes"/f"test_{arm}_{s}"/"summary.json") for s in protocol["evaluation"]["seeds"]]
        summaries[arm]={}
        for metric in ("final_hv_gain","mean_querywise_hv_gain","final_hv","invalid_proposal_rate"):
            values=[r[metric] for r in data[arm]]
            summaries[arm][metric]={"n":len(values),"mean":statistics.mean(values),"sample_variance":statistics.variance(values),"sample_std":statistics.stdev(values),"per_seed":values}
    paired={}
    for metric in ("final_hv_gain","mean_querywise_hv_gain"):
        differences=[b[metric]-a[metric] for a,b in zip(data["qwen_base"],data["uq_esopt"])]
        paired[metric]={"per_seed_delta":differences,"mean_delta":statistics.mean(differences),"sample_variance":statistics.variance(differences),"wins":sum(x>1e-12 for x in differences),"ties":sum(abs(x)<=1e-12 for x in differences),"losses":sum(x < -1e-12 for x in differences)}
    ablations={}; prefix_reports={}; calibration_reports={}
    for first,second in (("qwen_base","uq_only"),("qwen_base","es_only"),("es_only","uq_esopt"),("uq_only","uq_esopt"),("random_controller","uq_only")):
        ablations[second+"_minus_"+first]={}
        for metric in ("final_hv_gain","mean_querywise_hv_gain"):
            values=[b[metric]-a[metric] for a,b in zip(data[first],data[second])]
            ablations[second+"_minus_"+first][metric]={"per_seed_delta":values,"mean_delta":statistics.mean(values),"sample_std":statistics.stdev(values)}
    for arm,episodes in data.items():
        prefix_reports[arm]={}
        for budget in (10,30,50):
            values=[r["hv_curve"][budget-1]-r["prior_hv"] for r in episodes]
            prefix_reports[arm][str(budget)]={"per_seed_hv_gain":values,"mean":statistics.mean(values),"sample_std":statistics.stdev(values),"independent_budget_run":False}
        raw=[read(p) for seed in protocol["evaluation"]["seeds"] for p in sorted((root/"episodes"/f"test_{arm}_{seed}").glob("query[0-9][0-9][0-9].json"))]
        calibration_reports[arm]=calibration(raw)
    result={"schema":"summit_snar_complete_report_v1","protocol_fingerprint":digest(protocol),"adaptation_oracle_calls":550,
        "test_oracle_calls":1750,"acceptance":acceptance,"arms":summaries,"paired_full_minus_base":paired,"ablation_paired_deltas":ablations,"episodes":data,
        "selected_checkpoint":{b:read(root/f"{b}_checkpoint_selection.json") for b in ("full","es_only")},"risk_development":read(root/"risk_fit.json")["development_metrics"],
        "test_calibration":calibration_reports,"same_trajectory_prefixes":prefix_reports,
        "limitations":["one known reaction function; five heldout evaluation seeds, one adaptation training seed schedule",
            "shared 550-query adaptation prior; online gains must not be called total-cost cold-start efficiency",
            "GP-EI uses fixed product scalarization, not EHVI or Summit TSEMO",
            "full method may generate a second proposal; policy compute and wall time differ",
            "B10/B30 are prefixes of the same B50 episodes, not independent budget experiments",
            "native cut-Jacobian CRV-inspired graph is not the original CRV replacement model"]}
    write_once(root/"results.json",result)
    lines=["# Summit SnAr registered results","","Official deterministic ODE; original Qwen3.5-4B. All seven arms use the same frozen 550-query prior. Each arm has five independent evaluation seeds and 50 new oracle calls per seed.","",
        "| Arm | Final HV gain mean ± SD | Querywise HV gain mean ± SD |","|---|---:|---:|"]
    for arm,s in summaries.items():
        f=s["final_hv_gain"]; q=s["mean_querywise_hv_gain"]
        lines.append(f"| {arm} | {f['mean']:.6g} ± {f['sample_std']:.6g} | {q['mean']:.6g} ± {q['sample_std']:.6g} |")
    lines += ["","Full-minus-base paired deltas: `"+canonical(paired)+"`.","","Limitations:",""]+["- "+x for x in result["limitations"]]
    (root/"results.md").write_text("\n".join(lines)+"\n")
    write_once(root/"study_complete.json",{"episodes":35,"adaptation_queries":550,"test_queries":1750,"results_fingerprint":digest(result)})

if __name__=="__main__": main()
