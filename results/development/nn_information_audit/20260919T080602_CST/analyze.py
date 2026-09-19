"""Recompute a frozen train/dev diagnostic. No fitting, external I/O or science."""
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import math

D = Path(__file__).resolve().parent
E = json.loads((D / "evidence.json").read_text())
R = E["rows"]
HEADS = ["future_failure", *R[0]["typed_labels"]]
SCIENCE = {"future_failure", "scientific_evaluation_failure", "unstable", "not_new"}


def dump(name, value):
    (D/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def csv_dump(name, rows):
    with (D/name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def target(r, h):
    return r["label_future_failure"] if h == "future_failure" else r["typed_labels"][h]


def prediction(r, h):
    return r["scalar_probability"] if h == "future_failure" else r["typed_probabilities"][h]


def ordinal(r):
    return tuple(map(int, r["local_decision_id"][1:].split("-c")))


def event_key(r, h):
    if h == "generation_invalid":
        return r["episode_id"], r["decision_id"]
    source = "scientific_evaluation_failure" if h == "future_failure" else h
    ids = r["source_rpc_ids"][source]
    assert len(ids) == 1, (r["decision_id"], h, ids)
    return r["episode_id"], ids[0]


def observations(head, split, view, graph):
    base = [r for r in R if r["split"] == split and r["fit_exclusion"] is None and target(r, head) is not None]
    if graph != "all":
        base = [r for r in base if r["graph_available"] == (graph == "available")]
    if view == "proposal":
        return [{"episode":r["episode_id"], "y":target(r,head), "p":prediction(r,head), "id":r["decision_id"], "n_proposals":1} for r in base]
    assert head in SCIENCE
    groups = defaultdict(list)
    for r in base:
        groups[event_key(r, head)].append(r)
    out = []
    for key, rows in sorted(groups.items()):
        assert len({target(r,head) for r in rows}) == 1
        final = max(rows, key=ordinal)
        assert final["tool"] == "select_for_evaluation" and final["scientific_trigger"]["decision_id"] == final["decision_id"]
        ps = [prediction(r,head) for r in rows]
        p = prediction(final,head) if view == "event_last_select" else (mean(ps) if all(v is not None for v in ps) else None)
        out.append({"episode":key[0], "y":target(final,head), "p":p, "id":str(key[1]), "n_proposals":len(rows)})
    return out


def episode_prevalence(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["episode"]].append(r["y"])
    return mean(mean(y) for y in groups.values()) if groups else None


def score(rows, constant=None):
    if constant is None:
        used = [r for r in rows if r["p"] is not None]
        p = [r["p"] for r in used]
    else:
        used, p = rows, [constant] * len(rows)
    if not used:
        return {"n":0,"episodes":0,"positives":0,"mean_probability":None,"brier":None,"logloss":None,
                "auc":None,"auprc":None,"TP":0,"FP":0,"TN":0,"FN":0,"p_min":None,"p_max":None}
    y = [r["y"] for r in used]
    pos, neg = [v for t,v in zip(y,p) if t], [v for t,v in zip(y,p) if not t]
    auc = sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg)) if pos and neg else None
    # Average precision grouped at ties; no ranking of equal probabilities.
    bins = defaultdict(list)
    for yy,pp in zip(y,p): bins[pp].append(yy)
    tp = seen = 0; ap = 0.
    for pp in sorted(bins,reverse=True):
        by = bins[pp]; tp += sum(by); seen += len(by)
        if pos: ap += sum(by)/len(pos) * tp/seen
    groups = defaultdict(list)
    for r,pp in zip(used,p): groups[r["episode"]].append((r["y"],pp))
    eps = 1e-15  # Numerical log stabilizer, not a selected decision threshold.
    brier = mean(mean((yy-pp)**2 for yy,pp in values) for values in groups.values())
    nll = mean(mean(-(yy*math.log(min(1-eps,max(eps,pp)))+(1-yy)*math.log(min(1-eps,max(eps,1-pp)))) for yy,pp in values) for values in groups.values())
    return {"n":len(y), "episodes":len(groups), "positives":sum(y), "mean_probability":mean(p),
        "brier":brier,"logloss":nll,"auc":auc,"auprc":ap if pos else None,
        "TP":sum(yy==1 and pp>=.6 for yy,pp in zip(y,p)), "FP":sum(yy==0 and pp>=.6 for yy,pp in zip(y,p)),
        "TN":sum(yy==0 and pp<.6 for yy,pp in zip(y,p)), "FN":sum(yy==1 and pp<.6 for yy,pp in zip(y,p)),
        "p_min":min(p), "p_max":max(p)}


assert len(R) == 386
assert all(r["split"] in {"train","dev"} for r in R)
assert {r["task_id"] for r in R if r["split"]=="train"} == {"Al-Au-Hf"}
assert {r["task_id"] for r in R if r["split"]=="dev"} == {"Al-Pd-Sm"}
assert not ({r["prefix_hash"] for r in R if r["split"]=="train"} & {r["prefix_hash"] for r in R if r["split"]=="dev"})
assert all(r["typed_labels"]["generation_invalid"] == 1-int(r["generation_success"]) == 1-r["features"]["sampling.valid_json_action"] for r in R)
for r in R:
    if r["disposition"] != "executed":
        assert r["label_future_failure"] is None
        assert all(v is None for h,v in r["typed_labels"].items() if h != "generation_invalid")
    else:
        assert r["label_future_failure"] == int(bool(r["typed_labels"]["unstable"]) or bool(r["typed_labels"]["not_new"]))

coverage = []
for split in ["train","dev"]:
    for graph in ["all","available","missing"]:
        rows = [r for r in R if r["split"]==split and (graph=="all" or r["graph_available"]==(graph=="available"))]
        for head in HEADS:
            observed = [r for r in rows if target(r,head) is not None]
            coverage.append({"split":split,"graph_stratum":graph,"head":head,"rows":len(rows),
                "episodes":len({r["episode_id"] for r in rows}), "label_observed":len(observed),
                "positive":sum(target(r,head) for r in observed), "unknown":len(rows)-len(observed),
                "model_probability_available":sum(prediction(r,head) is not None for r in observed),
                "unique_observed_events":len({event_key(r,head) for r in observed}),
                "nonexecuted_with_known_label":sum(r["disposition"]!="executed" for r in observed)})

metrics, metrics_flat = [], []
for head in HEADS:
    views = ["proposal", "event_mean", "event_last_select"] if head in SCIENCE else ["proposal"]
    for view in views:
        for graph in ["all","available","missing"]:
            train = observations(head,"train",view,graph)
            dev = observations(head,"dev",view,graph)
            base = episode_prevalence(train)
            nn = score(dev)
            constant = score(dev,base) if base is not None else None
            fixed_half = score(dev,.5)
            item = {"head":head,"view":view,"graph_stratum":graph,
                "train_label_observations":len(train),"train_episodes":len({r['episode'] for r in train}),
                "dev_label_observations":len(dev),"dev_episodes":len({r['episode'] for r in dev}),
                "train_only_constant_probability_episode_equal":base,
                "frozen_NN":nn,"train_only_constant":constant,"fixed_half_reference":fixed_half,
                "Brier_difference_NN_minus_train_constant":nn["brier"]-constant["brier"] if constant and nn["brier"] is not None else None}
            metrics.append(item)
            for method, data in [("frozen_NN",nn),("train_only_constant",constant),("fixed_half_reference",fixed_half)]:
                if data is None:continue
                metrics_flat.append({"head":head,"view":view,"graph_stratum":graph,"predictor":method,
                    "train_constant_probability":base, **data})

episodes = []
events = []
for entry in E["episodes"]:
    job = entry["job"]
    rows = [r for r in R if r["decision_id"].startswith(job["job_id"]+":")]
    assert len({r["episode_id"] for r in rows}) == 1
    groups = defaultdict(list)
    for r in rows:
        ids = r["source_rpc_ids"]["scientific_evaluation_failure"]
        if ids: groups[ids[0]].append(r)
    assert set(groups) == {s["rpc_id"] for s in entry["steps"]}
    for step, outcome in enumerate(entry["steps"],1):
        candidates = groups[outcome["rpc_id"]]; last = max(candidates,key=ordinal)
        actual = outcome["official_observation"]
        assert outcome["ok"] and last["typed_labels"]["unstable"] == int(not actual["is_stable"])
        assert last["typed_labels"]["not_new"] == int(not actual["is_newly_discovered"])
        events.append({"split":job["split"],"task":job["task_id"],"seed":job["seed"],"job_id":job["job_id"],
            "step":step,"rpc_id":outcome["rpc_id"],"response_line":outcome["response_line"],
            "propagated_proposal_labels":len(candidates),"last_select_decision_id":last["decision_id"],
            "future_failure":last["label_future_failure"],"future_probability":last["scalar_probability"],
            "unstable":last["typed_labels"]["unstable"],"unstable_probability":last["typed_probabilities"]["unstable"],
            "not_new":last["typed_labels"]["not_new"],"not_new_probability":last["typed_probabilities"]["not_new"],
            "graph_available":last["graph_available"]})
    counts = Counter(len(v) for v in groups.values())
    episodes.append({"job_id":job["job_id"],"task":job["task_id"],"split":job["split"],"seed":job["seed"],
        "proposals":len(rows),"executed":sum(r["disposition"]=="executed" for r in rows),
        "nonexecuted":sum(r["disposition"]!="executed" for r in rows),"ORB_events":len(groups),
        "future_positive_proposals":sum(r["label_future_failure"]==1 for r in rows),
        "future_positive_events":sum(max(v,key=ordinal)["label_future_failure"] for v in groups.values()),
        "maximum_labels_per_ORB_event":max(counts),
        "graph_missing":sum(not r["graph_available"] for r in rows),
        "valid_generation_missing_graph":sum(r["generation_success"] and not r["graph_available"] for r in rows)})

feature_audit = []
for name in E["feature_names"]:
    tr = [r["features"][name] for r in R if r["split"]=="train" and r["features"][name] is not None]
    de = [r["features"][name] for r in R if r["split"]=="dev" and r["features"][name] is not None]
    feature_audit.append({"feature":name,"prefix":name.split(".")[0], "train_finite":len(tr),"dev_finite":len(de),
        "train_unique_values":len(set(tr)), "train_min":min(tr) if tr else None,"train_max":max(tr) if tr else None,
        "dev_outside_train_range":sum(v<min(tr) or v>max(tr) for v in de) if tr else None})
assert all(f["prefix"] in {"graph","hidden","action","sampling"} for f in feature_audit)

# The saved original descriptive metrics are reproduced, not used to choose a model.
repro = []
for head in HEADS:
    m = next(x for x in metrics if x["head"]==head and x["view"]=="proposal" and x["graph_stratum"]=="all")
    expected = E["fit_metadata"]["scalar"]["development_metrics"] if head=="future_failure" else E["fit_metadata"]["typed"]["development_metrics"][head]["descriptive_metrics"]
    if expected is not None:
        for old,new in [("brier","brier"),("auroc","auc"),("nll","logloss")]:
            if expected.get(old) is not None and m["frozen_NN"][new] is not None:
                error = abs(expected[old]-m["frozen_NN"][new])
                assert error < 1e-6, (head,old,error)
                repro.append({"head":head,"metric":old,"saved":expected[old],"recomputed":m["frozen_NN"][new],"absolute_difference":error})

summary = {"schema":"original_train_dev_frozen_NN_information_audit_v1", "complete":True,
    "source_evidence_sha256":hashlib.sha256((D/"evidence.json").read_bytes()).hexdigest(),
    "rows":386,"train_episodes":3,"dev_episodes":1,"train_ORB_events":150,"dev_ORB_events":50,
    "test_rows_read":0,"new_NN_fits":0,"new_oracle_calls":0,"new_LLM_calls":0,"GPU_calls":0,
    "dev_previously_used_for_epoch_selection_and_temperature":True,
    "OOF_CV_performed":False,"OOF_unavailable_reason":"The one frozen model was fitted on all 3 train episodes; valid OOF needs new fits, which this task forbids.",
    "fit_metadata":E["fit_metadata"],"episodes":episodes,
    "feature_prefix_counts":dict(Counter(f["prefix"] for f in feature_audit)),
    "constant_train_features":sum(f["train_unique_values"]<=1 for f in feature_audit),
    "generation_invalid_exactly_reconstructible_from_existing_feature_rows":386,
    "nonexecuted_future_or_postaction_labels_observed":0,
    "valid_generation_missing_graph_train":0,"valid_generation_missing_graph_dev":0,
    "main_metrics":[m for m in metrics if m["graph_stratum"]=="all" and ((m["head"] in {"future_failure","unstable","not_new"} and m["view"]=="event_last_select") or (m["head"] not in SCIENCE and m["view"]=="proposal"))],
    "aggregation":"Brier/logloss average within episode then equally across episodes; training constant averages episode prevalences in the same head/view/support stratum. AUC/AP descriptive pooled ranks (dev has exactly one episode).",
    "threshold":.6,"log_numerical_clip":1e-15,
    "limitations":["One previously calibrated dev trajectory is not untouched validation or a multi-seed estimate.",
       "Frozen train scores are in-sample; no OOF generalization claim.",
       "Scientific labels propagated to multiple prior executed tools are predictive associations, not causal responsibility.",
       "No counterfactual label for rejected proposals.",
       "Generation-invalid target is already encoded in parsing-success feature; perfect detection cannot establish added NN/graph value.",
       "No valid-generation missing-graph rows exist in the original train/dev corpus, so reliable risk there is unsupported.",
       "No feature ablation or new model is fitted; graph-specific incremental information is not identified."]}
dump("summary.json",summary)
dump("all_metrics.json",metrics)
dump("saved_metric_reproduction.json",repro)
csv_dump("metrics.csv",metrics_flat)
csv_dump("label_support.csv",coverage)
csv_dump("episodes.csv",episodes)
csv_dump("scientific_events.csv",events)
csv_dump("feature_support.csv",feature_audit)
per_episode = []
for head in HEADS:
    view = "event_last_select" if head in SCIENCE else "proposal"
    training = observations(head,"train",view,"all")
    base = episode_prevalence(training)
    for split in ["train","dev"]:
        values = observations(head,split,view,"all")
        for episode in sorted({x["episode"] for x in values}):
            subset = [x for x in values if x["episode"]==episode]
            for predictor, result in [("frozen_NN",score(subset)), ("train_only_constant",score(subset,base))]:
                per_episode.append({"episode_id":episode,"split":split,"interpretation":"training_in_sample" if split=="train" else "development_used_for_selection_and_calibration",
                    "head":head,"view":view,"predictor":predictor, **result})
csv_dump("per_episode_metrics.csv",per_episode)
csv_dump("proposal_predictions.csv",[{"decision_id":r["decision_id"],"split":r["split"],"episode_id":r["episode_id"],
    "graph_available":r["graph_available"],"disposition":r["disposition"],"scalar_fit_mask":r["scalar_fit_mask"],
    "typed_eligible":r["typed_eligible"],"future_failure":r["label_future_failure"],"scalar_probability":r["scalar_probability"],
    **{h+"_label":r["typed_labels"][h] for h in R[0]["typed_labels"]},
    **{h+"_probability":r["typed_probabilities"][h] for h in R[0]["typed_labels"]}} for r in R])
print(json.dumps({"complete":True,"rows":len(R),"events":len(events),"saved_metric_checks":len(repro),
    "main":[{"head":m["head"],"view":m["view"],"NN":m["frozen_NN"],"constant":m["train_only_constant_probability_episode_equal"],
             "delta_Brier":m["Brier_difference_NN_minus_train_constant"]} for m in summary["main_metrics"]]},ensure_ascii=False))
