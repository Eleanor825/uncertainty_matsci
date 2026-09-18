"""Create the full training/development collection matrix before observing results."""
from __future__ import annotations

import json
from pathlib import Path

from .accounting import fingerprint, pack_environment_seed


def collection_jobs(project: Path) -> list[dict]:
    project = Path(project)
    protocol = json.loads((project / "configs/main_protocol.json").read_text())
    tasks = json.loads((project / "configs/benchmark_tasks.json").read_text())
    splits = json.loads((project / "configs/made_splits.json").read_text())["splits"]
    model_manifest = json.loads((project / "configs/model_manifest.json").read_text())
    models = {m["key"]: m for m in model_manifest["models"]}
    if protocol["models"] != ["qwen35_4b", "qwen35_9b"] or protocol["training_seeds"] != [1, 2, 3, 4, 5]:
        raise ValueError("This locked full study requires both Qwen sizes and all five seeds")
    made_config = protocol["collection"]["made"]
    if made_config != {"train_systems": 30, "development_systems": 12, "episodes_per_system": 5, "oracle_attempts_per_episode": 50}:
        raise ValueError("MADE collection budget differs from the complete registered protocol")
    if len(splits["train"]) != 30 or len(splits["dev"]) != 12 or len(splits["test"]) != 30:
        raise ValueError("The full chemical-system splits must not be reduced")
    crystal_config = protocol["collection"]["crystalgym"]
    if (set(crystal_config["properties"]) != {"bm", "density", "band_gap"}
            or crystal_config["training_prototypes"] != 5 or crystal_config["policy_seeds"] != 5
            or crystal_config["training_rollouts_per_prototype_and_seed"] != 10
            or crystal_config["development_rollouts_per_prototype_and_seed"] != 2):
        raise ValueError("CrystalGym collection budget differs from the full registered protocol")
    jobs = []
    for model_key in protocol["models"]:
        model = models[model_key]
        base = {"stage": "collection", "model_key": model_key, "model_id": model["model_id"], "model_revision": model["revision"], "method": "baseline"}
        for split in ["train", "dev"]:
            for elements in sorted(splits[split]):
                task_id = "-".join(sorted(elements))
                for seed in protocol["training_seeds"]:
                    job = {**base, "benchmark": "made", "split": split, "task_id": task_id, "group_id": task_id,
                           "task": {"id": task_id, "elements": elements}, "seed": seed, "environment_seeds": [seed],
                           "episode_ids": ["0"], "budget": 50, "budget_unit": "candidate_oracle_attempts_per_episode",
                           "expected_counts": {"episodes": 1, "candidate_oracle_attempts": 50, "dft_episode_attempts": 0}}
                    job["job_id"] = "collection-made-" + fingerprint(job)[:24]
                    jobs.append(job)
        for prop in tasks["crystalgym"]["properties"]:
            # Task file stores the fixed OOD targets used by the final benchmark.
            if isinstance(prop, str):
                raise ValueError("CrystalGym properties need their fixed target records")
            for ordinal, prototype in enumerate(p for p in tasks["crystalgym"]["prototypes"] if p["split"] == "train"):
                for seed in protocol["training_seeds"]:
                    for split, count in [("train", 10), ("dev", 2)]:
                        partition = "uncertainty_fit_and_es_train" if split == "train" else "uncertainty_calibration_dev"
                        # Reserve disjoint rollout ranges for each of the five fixed
                        # training prototypes within the training-family code.
                        seeds = [pack_environment_seed(seed, prop["id"], "training_mixture", ordinal * 10000 + r, partition=partition) for r in range(count)]
                        task_id = prop["id"] + ":" + prototype["id"]
                        fixed_prop = {"id": prop["id"], "target": prop.get("target", prop.get("out_of_distribution_target")), "unit": prop.get("unit")}
                        if fixed_prop["target"] is None:
                            raise ValueError("CrystalGym OOD target is not specified")
                        job = {**base, "benchmark": "crystalgym", "split": split, "task_id": task_id,
                               "group_id": f"{task_id}:policy_seed:{seed}:partition:{split}", "task": {"property": fixed_prop, "prototype": prototype},
                               "seed": seed, "policy_sampling_seed": seeds[0], "environment_seeds": seeds, "episode_ids": [str(i) for i in range(count)],
                               "budget": count, "budget_unit": "dft_episode_attempts", "expected_counts": {"episodes": count, "candidate_oracle_attempts": 0, "dft_episode_attempts": count}}
                        job["job_id"] = "collection-crystalgym-" + fingerprint(job)[:24]
                        jobs.append(job)
    if len(jobs) != 720 or sum(j["expected_counts"]["candidate_oracle_attempts"] for j in jobs) != 21000 or sum(j["expected_counts"]["dft_episode_attempts"] for j in jobs) != 1800:
        raise ValueError("The generated collection matrix is incomplete")
    return sorted(jobs, key=lambda j: (protocol["models"].index(j["model_key"]), j["benchmark"] != "made", j["split"] != "train", len(j["task"].get("elements", [])), j["task_id"], j["seed"]))
