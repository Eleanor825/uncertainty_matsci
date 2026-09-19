# Original sources for the new MADE and SnAr studies

This is an **original-byte source snapshot for inspection and source-layout reconstruction**, not a runnable copy of the historical server workspace. It adds61 missing scientific/code-validation/configuration payloads (887,801bytes) identified against the actual registered studies. Existing `src/`, prior scientific results and already published source bytes are unchanged.

## What is covered

- MADE baseline and independently trained ES-only: the component registration (`c11cfbe1…`) plus its53 runtime and5 external source bindings.
- MADE support-aware UQ-only and newly trained full ES: the support registration (`0088ab73…`) plus54 runtime and8 external sources; the actual import-corrected full-ES loader is included separately.
- SnAr seven-arm study:93 main source bindings from registration `71fe749d…`, the original causal128 capture, the actual TCv3 registration `034dbb1a…`, and the import-only report acceptance adapter.

These bindings overlap. [source_mapping.json](source_mapping.json) maps all226 main/transitive binding rows by their original SHA to either an existing repository file, one of the61 new payloads, or an explicit external dependency. [payload_index.json](payload_index.json) identifies the exact original bytes. Duplicate content is stored once. Scientific tests that were source-bound are included; reporting analyses are not substituted for experiment implementations.

The active MADE matrix is baseline270 + independentES270 + support-UQ270 + support-full270 =1,080. The older component registration also contains a legacy schema-only UQ definition; it is not part of this four-arm main matrix. The SnAr35 episodes are a separate study, and its selected G0 results do not demonstrate an ES benefit. Nothing in this snapshot adds experiments or changes scores.

## Verify without loading a model

Use the Python standard library and the existing repository checkout:

```sh
python3 -B verify.py --repo-root /path/to/uncertainty_matsci
```

The verifier hashes this snapshot, all61 original payloads, every mapped existing repository file, and the derived configuration projections. It never imports the scientific source, installs a package, calls the network, or executes a model/oracle. `-B` prevents local Python bytecode files from being added to the snapshot.

To copy only the available original source layout into a **new** directory whose parent already exists:

```sh
python3 -B materialize.py --repo-root /path/to/uncertainty_matsci --output /path/to/new-source-layout
```

This first verifies all bytes, refuses any existing output directory, copies source/config bytes, and records a materialization summary. It does not create production registrations, old receipts, data, weights or resource leases. Missing external/runtime dependencies remain explicitly missing. The copied source layout is not an accepted experiment workspace; the command does not run original CLIs.

## Configuration identity and portability

The two historical runtime JSON files under `originals/` retain their original bytes, including historical absolute paths and GPU identifiers. They are archival scientific-runtime references, not instructions to access that machine. All61 payload SHA values remain exactly the original values.

[configuration_projections/index.json](configuration_projections/index.json) instead contains original/export SHA pairs for four **derived** registration projections. Project path prefixes are relocated consistently in keys and values; all numeric values, arrays, nulls, job definitions and selection settings are retained. Each projection is wrapped in a different schema, explicitly marked non-executable, and is not represented as the original registration. Original fingerprints appearing inside it are historical references only. No old success, receipt or lease was invented.

Original source has strict parent/source/checkpoint/receipt and sometimes resource-producer checks. A new-machine runnable implementation still needs a separately reviewed resource interface and explicit artifact-path reconstruction, with new registration fingerprints where paths change. The import-recovery and `resource_bank.py` modules are included to document actual scientific execution/acceptance lineage; their external resource dependencies are not silently bypassed.

## External assets and excluded administration

[external_dependencies.json](external_dependencies.json) and [configuration_coverage.json](configuration_coverage.json) list pinned Qwen4B weights, original MADE/corpus/evaluator and MACE/generator assets, learned TC/risk/selected-policy artifacts, SnAr activation data and550-point prior, and dependency locks. The public tables can be recomputed without these large assets; re-executing the original experiment cannot.

[missing_execution_adapters.json](missing_execution_adapters.json) separates the historical guards, node supervisor/deployment records and unavailable bank manifest from scientific payloads. No authentication code, credential, node-access helper, lifecycle/GPU takeover script or fake resource receipt is included. Do not copy unrelated server administration to make a source hash gate appear satisfied.

Upstream code/model licenses remain with their respective projects. The Summit source lock and public wheel URLs are metadata; this snapshot does not vendor external libraries or publish model weights.
