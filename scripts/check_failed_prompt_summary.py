#!/usr/bin/env python3
"""Read-only regression of a recorded failed prompt; no scientific calls."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from matdiscovery.rollouts import _messages, _small, RolloutSettings, dump_json, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--rpc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rpc.read_text().splitlines()]
    requests, responses = {}, {}
    observation, recent, history = None, [], []
    for row in rows:
        value = row.get("payload", {})
        if row.get("direction") == "request":
            requests[value["id"]] = value
        elif row.get("direction") == "response":
            responses[value["id"]] = value
            operation = requests[value["id"]]["op"]
            result = value.get("result", {})
            if operation == "tool":
                recent.append(_small(value))
            if value.get("ok"):
                if operation == "observe":
                    observation = result
                elif "observation" in result:
                    observation = result["observation"]
                if operation == "step" and "official_observation" in result:
                    history.append(_small(result["official_observation"]))
    assert observation is not None and len(history) > 0
    assert set(requests) == set(responses), "Unresolved physical request cannot be replayed"
    tokenizer = AutoTokenizer.from_pretrained(args.project / "data/models/qwen35_4b", local_files_only=True)
    counts = []
    for level in range(3):
        messages = _messages("made", observation, recent, history, settings=RolloutSettings(), level=level)
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, truncation=False, enable_thinking=False)
        counts.append(int(encoded["input_ids"].shape[1]))
    assert all(n > 100 for n in counts), "Regression must count tokens, not BatchEncoding keys"
    assert min(counts) <= 3072, counts
    record = {"technical_only": True, "scientific_calls": 0, "source_rpc": str(args.rpc),
        "source_rpc_sha256": file_hash(args.rpc), "repaired_prompt_token_counts": counts,
        "all_recorded_requests_have_responses": True, "last_scientific_counts": observation.get("counts"),
        "summary_source_sha256": file_hash(args.project / "src/matdiscovery/rollouts.py")}
    dump_json(args.output, record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
