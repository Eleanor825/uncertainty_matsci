# Label direction, calibration metric and archive conditions

This is a source/metadata check against the locally available frozen files. No model probability is recomputed and no oracle is called. Source hashes and links to the quoted scientific functions are recorded in [source_contract.json](source_contract.json). See [label/metric code](source_evidence/uncertainty.md), [episode/prior code](source_evidence/pipeline.md), [repaired fit/test code](source_evidence/repair_pipeline.md), [test metric code](source_evidence/study.md), and [acceptance checks](source_evidence/acceptance.md).

## Label and probability direction are consistent

1. Original `benchmark_extensions/summit_snar_20260918/pipeline.py:222–223` computes the next executed query's hypervolume increment against **prior + earlier episode records**, then stores `no_hvi = int(increment <= 1e-12)`. Therefore1 means failure to add HV;0 means a positive increment.
2. `benchmark_extensions/summit_snar_representation_repair_20260919_v2/audit.py:179` records the diagnostic label from the executed query's `no_hvi`. Repaired pipeline `repair_pipeline.py:86` carries this label forward, and lines142–153 build `ys` without inversion and call `CalibratedRiskModel.fit`/`predict_proba`.
3. Frozen `src/matdiscovery/uncertainty.py:143` uses binary cross entropy with these labels; lines163–174 fit a positive scalar temperature on development labels and return `expit(logits / temperature)`. The output isP(no-HVI=1), notP(improvement).
4. Original pipeline lines198–206 accept a first valid candidate when risk<.5 and otherwise permit a second candidate; selection uses minimum available risk. This uses the same direction as the training label.
5. Development metrics use `brier_score_loss(y,p)` in `uncertainty.py:208`. Test episode Brier is the mean of `(risk - no_hvi)**2` over actually scored executed rows in `study.py:89–91`. These are the same binary Brier convention. No factor-of-two or label inversion is present in these paths.
6. Existing acceptance source `acceptance_repair.py:346` requires risk-fit labels to equal original executed `query.no_hvi`; lines286/390 independently require the stored query label to match the recomputed HV increment and the registered tolerance. This audit was already completed; it is not rerun here.

Accordingly the local evidence does not support a label-direction or Brier-definition bug. It also does not independently certify every runtime feature/probability; that would require the bounded stored-record extraction in the accompanying plan.

## 150 collected calls are not a150-row initial prior

| Phase | Data/queries used | Starting external archive | Probability use / evaluated condition |
|---|---|---:|---|
|Risk train collection|3 separate episodes×30 calls;75 non-LHS graph rows|0 in each episode|Qwen base, episode-local history;70/75 labels1|
|Risk development and temperature calibration|2 separate episodes×30 calls;50 non-LHS graph rows|0 in each episode|Same50dev rows serve early stopping/temperature calibration/development metrics;45/50 labels1|
|Full and ES-only population / checkpoint development|20-query episodes,200 calls per branch|0 in each episode|Own LHS/history; full includes controller and Brier penalty|
|Prior freeze|150 collection +200full +200ES-only=550 completed rows|Pooled only after adaptation|Immutable shared archive before any test call|
|Held-out comparison|7arms×5seeds×50calls|550 in every episode|LLMs see a deterministic≤4-Pareto-row summary; GP sees raw archive; metric uses all550|

Source evidence:

- `pipeline.py:172`: `episode(..., prior=(), ...)` default.
- `pipeline.py:237–240`: collection invokes each episode with `budget=30` and **does not pass prior**. It resets `records=[]` at179. The150 is total collection cost across five episodes.
- `repair_pipeline.py:138–158`: the125 rows split into75train/50dev and temperature-calibrated development probabilities are obtained from the same development input population. `uncertainty.py:149–165` shows epoch selection and temperature calibration use those development labels.
- `pipeline.py:346` and356: checkpoint-development and population episodes also omit prior.
- `pipeline.py:375–384`: freeze concatenates all completed collection/ES query records and explicitly requires550 rows before writing the shared prior.
- `repair_pipeline.py:188–191`: held-out episodes explicitly receive `shared_prior_archive.json`.
- The frozen test summaries directly record `prior_rows=550`, `prior_hv=.8810117795428745` and the same prior fingerprint in all35 episodes. The six full checkpoint-development summaries directly record `prior_rows=0` and`prior_hv=0`.

These are **different archive and selection conditions**, not merely different random seeds. The label is context-dependent: a condition can improve an episode's cold-start front yet fail to improve a larger pooled front. Test queries additionally condition on the UQ controller's selection, unlike the base-policy collection population. Therefore comparing dev Brier.097142 and selected-test Brier.677841 is evidence of poor transfer/calibration under the held-out protocol; it is **not direct proof that the NN training algorithm malfunctioned or that label direction reversed**.

The score distribution changed in the observed controller behavior: all50dev actions met the high-risk retry condition, whereas202 of225 held-out full/UQ decisions stopped after their first valid risk<.5 proposal. Feature/prompt/archive shift and selection effects require stored per-proposal risk/features to separate. No test-tuned repair has been made or proposed as an accepted result.

## Leakage assessment within the recorded workflow

The source and already accepted evidence support **a frozen common reference archive, not incorporation of held-out test outcomes**. Original freeze rejects any non-collection/non-ES episode at `pipeline.py:381`, requires exactly550 rows at383, and writes an immutable archive at384. Repaired evaluation loads that archive before each test episode. Existing acceptance checks require the archive to equal the exact ordered adaptation rows (`acceptance_repair.py:333`), check each episode's correct prior fingerprint (270–272), check candidate prompts against the appropriate prior summary and earlier-only history (309), require development-only checkpoint selection (211), and require risk-fit test-use flags false (340). These checks belong to the completed accepted study; this task inspects their source and existing summaries rather than executing them again.

Collection-development observations and ES checkpoint-development observations are intentionally included in the common550-row **test-time** prior after their training/selection roles finish. All seven arms use the same frozen prior for metric initialization. This is disclosed warm-start information on one known reaction function; it is not an independent holdout of a novel chemical function. The same50 risk-development actions are reused for early stopping, scalar-temperature calibration and reported dev metrics, so those metrics are not an independent calibration-test estimate. Neither fact is evidence that the held-out5101–5105 results were used to fit or recalibrate the model.

No test recalibration was performed. The exact cause of the poorer selected-test risk calibration remains unproven without existing per-proposal feature/risk records. Archive shift is a source-supported mechanism hypothesis and protocol difference, not a standalone causal explanation.
