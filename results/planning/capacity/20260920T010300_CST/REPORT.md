# 24-hour capacity under the actually assigned queues

**Planning scenarios, not completed experimental results or statistical confidence intervals.** The duration sample is the frozen360/1080 snapshot at00:47:24 CST on September20. Exact claim-to-result times for those360 runs and original35 SnAr jobs were read back at01:09:13. The actual planning baseline was **362 completed jobs at01:03 CST**, reported by the root from its live observation. The forecast deducts the two completions after the360-row timing snapshot rather than counting them again. The later **369/1080 at01:17:20 CST is an actual completed-result export**, [published separately](../../../made_components/incremental/20260920T011720_CST/report.md). It is not a forecast outcome. Do not add this planning range to369 as though both used the same start time; the model retains its362 baseline and estimates24hours under the stated allocation. This package does not itself republish an independently verified362-row result matrix.

## Actual allocation and expected new work

The modeled allocation is the observed17 original workers (baseline1, ES-only1, UQ7, full8) plus **one** additional card with the actual four priority blocks: baseline first2 core systems×3seeds (6B50), ES the same6, baseline remaining3 systems×3seeds (9B50), ES the same9. The first2 systems are Al–Li–V and Al–V–Zn, following the fixed core order. The other three cards are assigned to4B collection/graphs,9B baseline→collection, and SnAr12→9B. Their unmeasured future contributions are not added to MADE4B throughput.

| Future24h, from reported362 | Conservative scenario | Median scenario |
|---|---:|---:|
| Additional completed4B MADE trajectories | **321** | **392** |
| Of those: B30 | 257 | 267 |
| Of those: B50 | 64 | 125 |
| Approximate total from the anchored observation | 683/1080 | 754/1080 |
| B50 completed on the extra priority card | 17/30 | 30/30 |
| Complete core B50 four-arm task/seed sets | **2/15** | **5/15** |

For user-facing planning, round this to **about320–390 new4B trajectories in24h**, including about260B30 and65–125B50. These are conditional scenarios assuming the assigned workers remain available, jobs succeed and the measured timing regime continues. They are not a guaranteed lower/upper bound. Additional node failures, new physical failures, shared-storage slowdown or prolonged resource handoff can reduce delivery. These counts do not use21×24 raw GPU-hours as a substitute for actual scientific jobs.

The initial360 snapshot has347 acceptedB10 jobs and13 acceptedB30 jobs. B10 core5×3seeds×4arms is already **60/60 complete**. Across the full matrix,23 jobs are failed and697 are still eligible/uncompleted; failed jobs are not automatically replayed or assumed recoverable. Complete main1080 acceptance therefore requires separate failure reconciliation as well as compute.

## Why the current order still limits matched comparisons

A single baseline worker must process about85 remainingB30 jobs before its normal B50 loop; the median service estimate alone makes that roughly54hours. The single ES worker has a similar34-hour barrier. Adding the explicit B/E priority card addresses that part, but UQ still processes its completeB30 matrix beforeB50 in seed order. Within24h its B50 completions concentrate in seed2, leaving core seeds3/4 late. Full and B/E can therefore finish many jobs without delivering all15 matched core task/seed sets.

The extra priority card itself needs about**24.0hours median /39.8hours conservative** for all30B50 jobs when four separate cold starts each receive a30-minute planning allowance. In the conservative schedule it completes6baseline,6ES and5of the next9baseline jobs; it does not magically complete all30.

There is also an independent acceptance blocker: UQ Al–Li–V B30 seed2 was already failed in the frozen snapshot. Even perfect scheduling cannot produce coreB30 **60/60** by simply continuing the unchanged queue. In the modeled24h original ordering,9of15 coreB30 task/seed sets are complete; the theoretical unchanged-queue ceiling after all runnable jobs is14/15 because of that failed cell.

If the priority is a complete B50 core table, the next resource to target is **UQ core B50 seeds3/4 (10 jobs)** on newly admitted or naturally released cards, rather than more unpaired full jobs. This is a scheduling recommendation, not an authorization to stop a running science process or replay any failed claim. The old small warm-sublist runner is hard-limited toB10/≤3jobs; its original `evaluate.run_job(...,_context=context)` interface can support a new registered-job order, but its old plan must not be relabelled as a B30/B50 plan.

## Measured service times and the explicit B50 extrapolation

Times below are minutes per accepted trajectory. B10 and observedB30 include actual claim→result time, including initialization, model setup after claim and result validation. For unavailable cells, empirical per-system budget ratios from the completed old30-system seed1 study are applied to the current arm's B10 rollout cost, then current initialization and other claim-to-result cost are added.

| Arm | B10 median /p75 (new n) | B30 median /conservative (new n) | B50 median /conservative (new n) |
|---|---|---|---|
| Baseline | 9.63 /12.59 (90) | 38.33 /40.89 (5) | **49.37 /86.99 (0; extrapolated)** |
| ES-only | 8.08 /9.42 (90) | 25.70 /30.44 (8) | **38.46 /64.04 (0; extrapolated)** |
| UQ | 31.56 /35.44 (83) | **69.73 /90.59 (0; extrapolated)** | **125.94 /161.64 (0; extrapolated)** |
| Full | 22.30 /27.03 (84) | **49.81 /65.45 (0; extrapolated)** | **89.52 /116.25 (0; extrapolated)** |

The old baseline within-system B50/B10 rollout multiplier has median6.023 andp75=8.246; the old graph-risk multiplier has median4.518 andp75=4.949. Its B30/B10 graph-risk multipliers are2.478/2.745. This avoids assuming a naive3×/5× budget scaling. However, the old graph-risk policy differs from the new support-aware policies, and the old matrix has only one evaluation seed. Transferring these ratios is an explicit modeling assumption, not a measurement of new B50 runtime. Current B30 baseline/ES samples are only5/8 completed cases, so survivor/system-selection effects and slow tails remain uncertain.

The simulation uses the actual job inventory and fixed queue order. Existing claimed jobs are charged a full remaining service time because this source snapshot has no per-query progress; no fabricated fraction-complete is credited. Each original worker receives a30-minute one-off source/model admission allowance; the separate priority card receives30minutes per6/9-job block. These allowances are planning margins, not claimed measurements of every transition. It never stops/reorders an already claimed job. The code and [service table](services.csv) expose every assumption; [scenario JSON](actual_allocation_scenarios.json) gives the exact computed counts.

## SnAr: actual elapsed times, additional seeds and limits

The original35 completed B50 jobs provide these full-job medians: baseline5.43min, ES-only6.59min, UQ28.81min and full30.17min. The corresponding maxima are5.83/6.72/29.32/30.30min. These include the original per-job model load and domain audit, unlike the shorter query-only timing. They exclude some preclaim source admission; allow extra margin on newly restored containers.

| Additional scope | New calls | One card measured-rate estimate | Two cards | Four cards |
|---|---:|---:|---:|---:|
| 6: baseline/full ×3new seeds (hypothetical) | 300 | 1.78h | 1.10h | 0.59h |
| **12: four arms ×3new seeds (registered)** | **600** | **3.55h** | **1.79h** | **1.07h** |
| 20: four arms ×5new seeds (hypothetical) | 1000 | 5.92h | 2.97h | 1.66h |

P75-based simulated times are within about1% here because the five original repeats have similar timing. That small spread is not a guarantee on a new node. Under the **actual one-card assignment**, budget approximately**3.6–4.5hours** for the registered12 before that card continues to9B; the2-/4-card columns are alternatives, not the actual current plan. The original fullES development took2.37hours for200calls and independent ES-only0.43hours, but neither is rerun by this seed extension.

New seeds5201/5202/5203 are disjoint from the original train/dev/test/ES seed catalog. The same original weights, NN, evaluator and550-query prior remain fixed. Both prior ES branches selectedG0. Consequently these12 jobs primarily improve evaluation-seed precision; they cannot by themselves demonstrate an evolved-weight gain or isolate an internal-NN causal benefit. Original35 episodes remain untouched. Additional independent fitting or a risk-masked controller experiment would require a separately frozen development/evaluation plan; no such success is claimed here.

The root's9B target22trajectories/500calls remains a **conditional task target**, not a throughput forecast:15 baselineB10 trajectories plus7 collection trajectories are not22held-out comparisons. No9B rate is borrowed from4B, and zero9B completed results are credited in this capacity calculation.

## Recompute and provenance

`python3 -B recompute.py` verifies input hashes and reconstructs every service and queue-simulation output. The input files are existing scientific result/cost projections; operator hostname and credentials were excluded. No model, graph, training, oracle, threshold or running worker was modified. These forecasts must remain labelled planning when linked beside incremental result reports; they must never be counted as completed trajectories or included in experimental means.
