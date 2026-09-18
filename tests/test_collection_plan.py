from pathlib import Path

from matdiscovery.experiment_plan import collection_jobs


def test_collection_has_every_training_and_development_condition():
    jobs = collection_jobs(Path(__file__).resolve().parents[1])
    assert len(jobs) == 720
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    assert sum(j["expected_counts"]["candidate_oracle_attempts"] for j in jobs) == 21000
    assert sum(j["expected_counts"]["dft_episode_attempts"] for j in jobs) == 1800
    assert all(j["split"] in {"train", "dev"} for j in jobs)
    assert all(j["budget"] == 50 for j in jobs if j["benchmark"] == "made")


def test_crystal_calibration_groups_and_random_streams_are_disjoint():
    jobs = [j for j in collection_jobs(Path(__file__).resolve().parents[1]) if j["benchmark"] == "crystalgym"]
    for model in {j["model_key"] for j in jobs}:
        train = [j for j in jobs if j["model_key"] == model and j["split"] == "train"]
        dev = [j for j in jobs if j["model_key"] == model and j["split"] == "dev"]
        assert not ({s for j in train for s in j["environment_seeds"]} & {s for j in dev for s in j["environment_seeds"]})
        assert not ({j["policy_sampling_seed"] for j in train} & {j["policy_sampling_seed"] for j in dev})
        assert all(j["policy_sampling_seed"] == j["environment_seeds"][0] for j in train + dev)
        assert not ({j["group_id"] for j in train} & {j["group_id"] for j in dev})
        assert {j["task"]["prototype"]["id"] for j in train} == {"C2", "C3", "C4", "C5", "C6"}
