#!/usr/bin/env python3
"""Select new MADE train/dev chemical spaces from fresh official public MP data.

No oracle runs, old results, network requests, or third-party imports are used.
The complete 30 official main systems remain test. Exact stable 3/4-element
systems are preferred when sufficient. The official table contains zero stable
5-element intermetallic compounds with <=20 sites: the explicitly recorded
fallback samples 5-element *chemical spaces* supported by stable subphases,
consistent with Materials Project get_entries_in_chemsys semantics.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re


COMMIT = "bffda0c9a1fa1904b35c0ef67b455881c975df86"
CSV_SHA256 = "cd505bc2500bc92b60bc4f0fe80b9ade7e6c70bdceb0ff6ff74f6866efc928d6"
REF_SHA256 = "3e66c10bfd00035c6f08f5737e2b7eda2637b9aeeb349c024bca8b5bbb535fa2"
# Official generator defaults to atomic number <=84. These are the only reference
# elements beyond that bound, identified by the standard periodic table.
REFERENCE_ELEMENTS_ABOVE_84 = {"Ac", "Th", "Pa", "U", "Np", "Pu"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_constants(path: Path) -> dict[str, set[str]]:
    """Read literal official element sets with AST; never execute vendor code."""
    wanted = {"METAL_ELEMENTS", "DEFAULT_EXCLUDED_ELEMENTS"}
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, set) or not all(isinstance(x, str) for x in value):
                        raise ValueError(f"Nonliteral official element set: {target.id}")
                    result[target.id] = value
    if set(result) != wanted:
        raise ValueError("Required official metallic/excluded-element sets not found.")
    return result


def group(formula: str) -> tuple[str, ...]:
    symbols = re.findall(r"[A-Z][a-z]?", formula)
    if not symbols:
        raise ValueError(f"Formula contains no element symbols: {formula!r}")
    return tuple(sorted(set(symbols)))


def subgroup_records(system, index, minimum_order=1):
    records = {}
    for size in range(minimum_order, len(system)+1):
        for subset in itertools.combinations(system, size):
            for row in index.get(subset, []):
                records[row["material_id"]] = row
    return [records[key] for key in sorted(records)]


def has_stable_subphases(system, stable_counts, required=2):
    count = 0
    for size in range(2, len(system)+1):
        for subset in itertools.combinations(system, size):
            count += stable_counts.get(subset, 0)
            if count >= required:
                return True
    return False


def reservoir_sample_sorted_spaces(elements, size, excluded, stable_counts, number, seed):
    """Uniform Algorithm-R sample of eligible spaces in lexicographic order.

    Streaming avoids retaining millions of five-element combinations. Ordering,
    rejection predicate and per-stratum RNG seed are frozen in provenance.
    """
    rng, sample, eligible = random.Random(seed), [], 0
    for system in itertools.combinations(sorted(elements), size):
        if system in excluded or not has_stable_subphases(system, stable_counts):
            continue
        eligible += 1
        if len(sample) < number:
            sample.append(system)
        else:
            replacement = rng.randrange(eligible)
            if replacement < number:
                sample[replacement] = system
    if eligible < number:
        raise ValueError(f"{size}-element chemical spaces insufficient: need {number}, found {eligible}; no reduced split written.")
    # Randomize train/dev assignment independently of reservoir slot age/order.
    sample.sort()
    random.Random(seed + 1_000_000).shuffle(sample)
    return sample, eligible


def build_splits(vendor: Path, seed=20260915, train_per_size=10, dev_per_size=4, require_exact=False):
    data = vendor / "data"
    csv_path, reference_path = data/"2025-02-01-mp-energies.csv", data/"2023-02-07-mp-elemental-reference-entries.json"
    actual_csv, actual_ref = sha256(csv_path), sha256(reference_path)
    if actual_csv != CSV_SHA256 or actual_ref != REF_SHA256:
        raise ValueError("Public data differs from the audited fresh official commit; freeze a new protocol explicitly.")
    generator_path = vendor/"scripts/generate_systems.py"
    constants = official_constants(generator_path)
    references = json.loads(reference_path.read_text())
    allowed = sorted(set(references) & constants["METAL_ELEMENTS"] - constants["DEFAULT_EXCLUDED_ELEMENTS"] - REFERENCE_ELEMENTS_ABOVE_84)
    tests, test_files = [], []
    for name in ("ternary", "quaternary", "quinary"):
        path = data/f"systems_10_mp_20/systems_{name}_n10_maxatoms20_intermetallic_smact.json"
        systems = json.loads(path.read_text())
        if len(systems) != 10:
            raise ValueError(f"Official test list is not complete: {path}")
        tests.extend(tuple(sorted(system)) for system in systems)
        test_files.append({"path": str(path.relative_to(vendor)), "sha256": sha256(path)})
    if len(set(tests)) != 30:
        raise ValueError("Official test must contain exactly 30 distinct chemical systems.")
    excluded = set(tests)
    public_rows, stable_rows = defaultdict(list), defaultdict(list)
    reject, total = Counter(), 0
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            system = group(row["formula"])
            if not set(system).issubset(allowed):
                reject["not_allowed_intermetallic"] += 1
                continue
            try:
                sites, hull = float(row["n_sites"]), float(row["energy_above_hull"])
            except ValueError:
                reject["invalid_numeric"] += 1
                continue
            if not math.isfinite(sites) or not math.isfinite(hull) or not 1 <= sites <= 20:
                reject["invalid_or_over_20_sites"] += 1
                continue
            if row["energy_type"] not in {"GGA", "GGA+U"}:
                reject["different_energy_type"] += 1
                continue
            record = {"material_id": row["material_id"], "formula": row["formula"], "n_sites": int(sites),
                      "public_energy_above_hull": hull, "energy_type": row["energy_type"]}
            public_rows[system].append(record)
            if hull <= 1e-8:
                stable_rows[system].append(record)
    stable_counts = {key: len(rows) for key, rows in stable_rows.items()}
    train, dev, strata = [], [], {}
    requested = train_per_size + dev_per_size
    for size in (3, 4, 5):
        exact_all = sorted(system for system in stable_rows if len(system) == size)
        exact = [system for system in exact_all if system not in excluded]
        size_seed = seed + size
        if len(exact) >= requested:
            selected = random.Random(size_seed).sample(exact, requested)
            mode, eligible = "exact_order_public_stable_systems", len(exact)
        elif require_exact:
            raise ValueError(f"{size}-element exact stable candidates insufficient: {len(exact)} available vs {requested} required. No partial/smaller split written.")
        else:
            selected, eligible = reservoir_sample_sorted_spaces(allowed, size, excluded, stable_counts, requested, size_seed)
            mode = "chemical_spaces_supported_by_at_least_two_stable_multielement_subphase_entries"
        chosen_train, chosen_dev = selected[:train_per_size], selected[train_per_size:]
        train.extend(chosen_train)
        dev.extend(chosen_dev)
        strata[str(size)] = {"seed": size_seed, "selection_mode": mode, "eligible_candidate_count": eligible,
            "exact_order_stable_candidate_count_before_test_exclusion": len(exact_all),
            "exact_order_stable_candidate_count_after_test_exclusion": len(exact),
            "train_count": len(chosen_train), "dev_count": len(chosen_dev),
            "shortage_disclosed": len(exact) < requested}
    sets = {"train": set(train), "dev": set(dev), "test": set(tests)}
    if any(sets[a] & sets[b] for a, b in itertools.combinations(sets, 2)):
        raise ValueError("Exact chemical-system leakage detected between splits.")
    if len(train) != 3*train_per_size or len(dev) != 3*dev_per_size:
        raise ValueError("Incomplete requested train/dev counts; refusing partial output.")
    evidence = {}
    for split, systems in (("train", train), ("dev", dev)):
        for system in systems:
            stable = subgroup_records(system, stable_rows, minimum_order=2)
            evidence["-".join(system)] = {"split": split, "exact_order_stable_entry_count": len(stable_rows.get(system, [])),
                "stable_multielement_subphase_entry_count": len(stable),
                "stable_reference_examples": stable[:10], "examples_cap": 10,
                "evidence_role": "public-data task selection only; never inject labels as policy observations"}
    materials = {split: {row["material_id"] for system in systems for row in subgroup_records(system, public_rows)} for split, systems in sets.items()}
    overlaps = {}
    for a, b in itertools.combinations(sets, 2):
        overlap = materials[a] & materials[b]
        nested = [{a: list(x), b: list(y)} for x in sorted(sets[a]) for y in sorted(sets[b]) if set(x) < set(y) or set(y) < set(x)]
        overlaps[f"{a}__{b}"] = {"equal_chemical_system_count": len(sets[a] & sets[b]),
            "shared_public_material_ids_inclusive_subsystems": len(overlap),
            "shared_id_examples": sorted(overlap)[:20], "strict_nested_system_pairs": nested}
    return {"schema_version": 1, "status": "complete_30_train_12_dev_30_test" if (len(train), len(dev), len(tests)) == (30,12,30) else "complete_explicit_custom_counts",
        "splits": {"train": [list(x) for x in train], "dev": [list(x) for x in dev], "test": [list(x) for x in tests]},
        "provenance": {"official_repository": "https://github.com/diffractivelabs/MADE", "official_commit": COMMIT,
            "public_energy_table": {"path": str(csv_path.relative_to(vendor)), "sha256": actual_csv, "rows": total},
            "reference_elements": {"path": str(reference_path.relative_to(vendor)), "sha256": actual_ref, "count": len(references)},
            "official_generator": {"path": str(generator_path.relative_to(vendor)), "sha256": sha256(generator_path), "constants_read_via_ast_without_execution": True},
            "test_files": test_files, "selection_script_sha256": sha256(Path(__file__)), "seed": seed,
            "sorting": "chemical symbols sorted lexicographically; candidate systems sorted lexicographically before RNG sampling",
            "exact_sampling": "random.Random(seed + system_size).sample(sorted_exact_candidates, n)",
            "space_sampling": "Algorithm-R reservoir over lexicographic itertools.combinations; Random(seed+size); sorted selected set then shuffle with seed+size+1000000 before train/dev assignment",
            "max_atoms": 20, "stable_reference_threshold_ev_per_atom": 1e-8, "energy_types": ["GGA", "GGA+U"],
            "allowed_elements": allowed, "strata": strata, "row_rejections": dict(reject),
            "uses_new_experiment_results": False, "uses_old_project_code_or_results": False,
            "quinary_limitation": "Zero exact five-element stable intermetallic systems with <=20 sites in this official table. Selected quinary spaces have stable multielement subphase evidence; they are not existing stable five-element compounds.",
            "raw_snapshot_requirement": "Fetch fresh complete Materials Project entries with structures for each selected space, then freeze/hash. The scalar CSV does not replace official benchmark structures or oracle."},
        "selected_system_evidence": evidence,
        "overlap_audit": {"exact_group_disjoint": True, "pairs": overlaps,
            "interpretation": "Chemical-space group disjointness does not imply disjoint structures. Elements/subsystems and public material IDs may overlap. IDs are only a proxy: this CSV has no structures for StructureMatcher deduplication.",
            "pretraining_risk": "Official ORB/MACE/Chemeleon pretraining includes MP/Alex/OMat data. These splits do not establish pretraining-data independence.",
            "policy_rule": "Do not train uncertainty/ES on official test trajectories or labels. Public selection evidence is evaluator metadata, not an agent observation."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--vendor-root", type=Path, default=project/"vendor/MADE")
    parser.add_argument("--output", type=Path, default=project/"configs/made_splits.json")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--train-per-size", type=int, default=10)
    parser.add_argument("--dev-per-size", type=int, default=4)
    parser.add_argument("--require-exact", action="store_true", help="Fail rather than use explicitly disclosed stable-subphase spaces.")
    args = parser.parse_args()
    if args.train_per_size <= 0 or args.dev_per_size <= 0:
        parser.error("Split counts must be positive.")
    result = build_splits(args.vendor_root.resolve(), args.seed, args.train_per_size, args.dev_per_size, args.require_exact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({"output": str(args.output), "counts": {k: len(v) for k,v in result["splits"].items()}, "strata": result["provenance"]["strata"], "overlap_audit": result["overlap_audit"]["pairs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
