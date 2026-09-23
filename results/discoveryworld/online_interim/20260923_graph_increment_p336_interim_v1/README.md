# Interim online graph-feature comparison: world4, seed336

**Internal produced fewer action failures (2/10 versus 5/10), while both task
scores remained 0 and neither trajectory completed the task successfully.**
This is one completed matched pair on a previously exposed world. The seed337
pair was still running and uncompleted at this snapshot; the four-group study
is not reported as complete.

| Verified quantity | Internal (293 features) | NoGraph (140 features) |
|---|---:|---:|
| Task score | 0 | 0 |
| Task completed successfully | No | No |
| Action failures / actual requests | 2 / 10 | 5 / 10 |
| F | -0.02 | -0.05 |
| Generated candidates / full-graph returns | 20 / 20 | 20 / 20 |
| Successful / unavailable full graphs | 13 / 7 | 12 / 8 |
| Supported two-candidate risk comparisons | 3 | 2 |
| Risk selections differing from common hash | 1 | 0 |
| Fresh qualification prefixes | 8 | 8 |
| Total graph computation time (seconds) | 2130.56 | 2240.29 |
| Recorded group wall time (seconds) | 2951.03 | 3045.43 |

Internal-minus-NoGraph is **+0.03 F**, **-3 action failures**, and **0 task-score
change**. The failure fraction is lower by 0.30. These are descriptive results;
ten sequential actions are not ten independent experimental replicates.

## What changed on the first action

The first action already differed. The audit verified identical public prompts,
proposal seeds, generated token sequences, candidate packets, generation
configuration and actual G0 tensors for both candidate positions. The two frozen
predictors ranked that same candidate pair oppositely:

| Predictor | Candidate0 risk | Candidate1 risk | Selected index |
|---|---:|---:|---:|
| Internal | 0.0373256691 | 0.0069630304 | 1 |
| NoGraph | 0.2168112546 | 0.3506183326 | 0 |

Both selected actions were `TELEPORT_TO_LOCATION`, with different packet hashes.
Each actually executed first action returned success. This directly demonstrates
a different ranking and choice on matched candidates; it does **not** show an
immediate first-step failure prevented by Internal. The later trajectories then
diverged. No outcome was assigned to an unexecuted candidate.

## Scope and verification

Both arms used the same original Qwen3.5-4B G0, frozen full transcoder bank,
full-eight-prefix qualification, two candidates, explicit public feedback,
common fallback rule, evaluator, environment seeding and unchanged full graph
capture. Internal used293 raw features; NoGraph used140 non-graph raw features projected
from the same full capture. Their frozen predictors and temperatures came from
the same fitting/calibration lineage. There was no new training or ES update.
Failed/unavailable graph captures stayed in the record and limit supported
neural comparisons; twenty graph returns does not mean twenty successful graphs.

The original committed scientific registry, source manifests, raw B10 fitness,
all twenty candidate graph records per group, G0 hash, complete qualification,
environment closure and resource process-tree closure were verified on the
actual server. The audit was read-only and made no model or environment calls.
Risk numbers above are recorded online outputs, not newly inferred labels.

This interim result supports a narrow observation of lower action failure in
this pair. It establishes neither improved task success nor a general winner
across worlds. Seed337 and the separately registered GraphOnly/Common2 appendix
remain separate until their results and comparability checks are complete.

Files: [metrics](paired_metrics.csv), [summary/status](summary.json),
[first selection difference](first_selection_difference.json),
[per-step selection audit](selection_audit.csv), [provenance](provenance.json),
[public-package validation](validate.py), [manifest](manifest.json).
Raw prompts, observations, candidate argument values, private answers and weights
are not included. Hashes and project-relative identifiers retain provenance.
