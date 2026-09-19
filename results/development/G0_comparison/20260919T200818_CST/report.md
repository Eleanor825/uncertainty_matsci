# G0 versus existing evolved checkpoints on one development trajectory

The added G0 controller run achieved **AUDC0.68**, compared with **0.62 at G1** and **0.64 at G2** under the matched Al-Pd-Sm/B10 development setup. Neither evolved checkpoint exceeds G0 in this comparison. This is one previously used development seed; it does not establish independent generalization, select a new checkpoint, or replace the frozen G2 used by ongoing test evaluations.

| Generation | AUDC | Change versus G0 | New candidate ORB calls in this diagnostic |
|---|---:|---:|---:|
| G0 |0.68|0|10|
| G1 |0.62|−0.06|0: existing result referenced|
| G2 |0.64|−0.04|0: existing result referenced|

G0 uses the original Qwen3.5-4B theta0 and the same controller profile as the original full-method development comparison. Its SUN is6, with22/22 supported proposals. Actual candidate ORB calls were10; initialization ORB calls were26 and surrogate MACE calls46, separately recorded. Its measured graph time was1244.674s out of1575.412s wall time (initialization67.292s is recorded separately).

The diagnostic's original CPU audit passed and verified the result, raw physical records, unchanged source/model/controller identity and shared seed schedule. Existing G1/G2 physical outputs are referenced without replay. Driver seed4289295197 maps to environment seed4272278902. Only10 candidate calls are new; this diagnostic is outside both the original180-trajectory study and the ongoing1080-trajectory repeat matrix. No prior failed result, uncertainty model or checkpoint selection was changed.

`comparison.json` is a reduced scientific projection with original artifact hashes. Its file hash is distinct from the original result hash. `recompute.py` checks the reported metric differences, matched evaluator/budget/seeds and cost accounting from this projection; it does not rerun the original acceptance, model, graph or oracle.
