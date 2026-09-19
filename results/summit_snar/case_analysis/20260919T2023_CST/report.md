# SnAr positive/negative cases: local evidence analysis

This analysis uses the existing accepted 35 B50 episodes, seven arms and seeds5101–5105. It performs **zero new oracle, model, graph, training or remote calls**. It does not repeat the completed full-study acceptance audit. Input hashes and projection provenance are in [input_manifest.json](input_manifest.json); [recompute.py](recompute.py) regenerates the accompanying case tables using Python's standard library.

This is a new analysis of the [previously accepted seven-arm results](../../20260919T162548_CST/report.md), not a new experiment. Original source excerpts are linked through [source_contract.json](source_contract.json).

## What the cases establish

Full and UQ-only have the same five HV curves and final weight hashes. Full selected G0; independent ES-only also selected G0 and matches Qwen base's curves. Consequently there is **no demonstrated held-out selected-weight ES advantage under this common-prior design**. Any data-acquisition benefit of ES adaptation to the pooled550-row archive is not separately identified, because every arm receives that same archive. Raw proposal-sequence equality is not claimed from metric equality alone.

All numbers below are **incremental hypervolume ×10^6 over the same550-query prior**, not absolute HV or percentages. Query numbers start at1. Initial queries1–5 are the paired LHS design.

| Seed | Full/UQ gain | Qwen-base gain | GP-EI gain | Full−base | Full improvement queries | Full test Brier |
|---|---:|---:|---:|---:|---|---:|
|5101|6.964|0|189.891|+6.964|46|.72668|
|5102|96.430|50.671|432.298|+45.759|11,16,18|.61555|
|5103|5.033|74.091|409.145|−69.058|19|.67847|
|5104|5.033|140.740|361.468|−135.707|10|.70105|
|5105|0|0|442.521|0|none|.66722|

Full has two positive pairs, two negative pairs and one tie against base; **it loses to GP-EI in all five seeds**. The positive deltas sum to52.723 micro-HV, versus204.765 micro-HV lost in the two negative pairs. Mean full gain is22.692 micro-HV versus53.101 for base and367.064 for GP-EI. Full's mean-querywise-gain delta versus base is also negative, −1.348 micro-HV. These are five repeats of one deterministic reaction function and one fixed adaptation schedule, not five different reaction systems or independently trained models.

The two observed losses are large-gain misses, rather than evidence that the algorithm failed to produce any valid reaction condition:

- **Positive5102:** Full improves at queries11,16,18 and retains96.430 micro-HV. Base first improves only at33 and retains50.671. Full's earlier gains explain its larger querywise gain,66.572 versus18.242 micro-HV. Its final observed front includes `(tau=.55, equiv_pldn=1.2, conc_dfnb=.47, temperature=100)` with `sty=9644.223245625366, e_factor=9.115142185833085`. The local summaries do not establish which proposal index produced each improvement.
- **Positive5101:** A single small late gain at46 beats a flat base trajectory, but mean-querywise gain is only0.696 micro-HV. It is a genuine positive pair, with a limited optimization benefit. Its surviving new condition `(tau=.6, equiv_pldn=4.8, conc_dfnb=.5, temperature=32)` yields `sty=9615.08939247551, e_factor=9.700387163005304`.
- **Negative5103:** Full gains5.033 micro-HV at19 and then remains flat for31 queries. Base's larger gain at35,74.091 micro-HV, more than compensates for arriving later. Full has100% graph availability among valid proposals; missing graphs cannot explain this loss.
- **Negative5104:** Full gains the same5.033 micro-HV at10 and then remains flat for40 queries. Base gains140.740 micro-HV at37. Full has **zero invalid proposals and100% valid-proposal graph availability**, whereas base has one invalid proposal. A syntax/schema or missing-graph explanation is therefore insufficient. Both5103 and5104 end with the same additional Pareto condition `(tau=.55, equiv_pldn=4.9, conc_dfnb=.5, temperature=32)`, `sty=10501.983278613236, e_factor=9.72477974584034`. This proves identical surviving front points, **not repeated generation of that condition throughout either trajectory**.
- **Tie5105:** Both full and base add zero HV. Full has one unavailable valid-proposal graph, two invalid proposals and44 scored actions. GP-EI still gains442.521 micro-HV, and the always-two random controller gains141.810. Thus there was measurable remaining improvement available to other policies; this is not proof that an unexecuted full proposal would have achieved it.

GP-EI obtains22 HVI-positive query events across its250 calls. Full obtains6; base obtains3. Yet the sum of base's three gains exceeds the sum of full's six. The binary no-HVI target distinguishes any positive increment from none; it does not encode improvement magnitude. **This is an observed frequency-versus-magnitude mismatch and a plausible utility-alignment concern, not proof that risk ranking caused the missed large gains.** No unexecuted candidate is given an invented outcome.

## Shared prior and GP comparison

Every arm begins from HV **.8810117795428745**, inherited from550 completed adaptation observations. The final absolute HV is overwhelmingly inherited; reporting only approximately.8810 for each arm conceals the differences above. The550 calls are additional to1750 held-out calls, for2300 actual physical calls overall. All175 initial LHS queries across the35 episodes add zero HV over this prior in the available curves.

The registered information paths differ: GP-EI receives all550 raw prior rows, while each LLM receives a deterministic summary capped at four Pareto rows plus its own recent history. Random receives the front for scoring but does not condition its uniform samples on it. This is the **actual declared comparison**, not a new fairness correction. GP-EI is a fixed-product-scalarized GP acquisition, not EHVI or TSEMO. Its advantage cannot be uniquely attributed to uncertainty modeling independently of numerical fitting, acquisition optimization, and access to the uncompressed archive.

All full test fronts retain the strong maximum-STY inherited condition at `tau=.5, conc_dfnb=.5, equiv_pldn=5, temperature=30`. GP's additional surviving points have tau approximately.5004–.5087 and concentration approximately.486–.500, while varying equivalents/temperature to improve the STY/E-factor tradeoff. These are **observed final Pareto points**; they are not a reconstruction of the full search path. [terminal_pareto_points.csv](terminal_pareto_points.csv) preserves every arm's final points and marks objective pairs shared with the zero-gain base5101 reference front.

## Uncertainty was active, but its held-out risk calibration deteriorated

The source targets **no positive HV increment for the next executed query**, not a causal reasoning-error label. This target depends on the current archive/front.

Risk training used75 train and50 development actions, with70/75 and45/50 no-HVI labels respectively. Development AUROC is.608889 and Brier.097142. Development AUPRC.93746 should be read alongside the.9 positive-label prevalence; the five HVI-success examples in development provide limited support for distinguishing successful proposals.

The frozen development-matched controller probability is1.0: all50 available dev risks met the retry condition. At test, source logic attempts a second proposal when the first is invalid, graph-unavailable, or risk≥.5; it otherwise accepts the first valid proposal with risk<.5. The two-attempt cap allows exact count reconstruction from the episode summaries:

| Seed | Proposal attempts /45 decisions | Second-proposal queries | Invalid proposals | Missing valid graphs | First high-risk retry bound | First proposal accepted with risk<.5 |
|---|---:|---:|---:|---:|---:|---:|
|5101|50|5|2|0|3–5|40|
|5102|51|6|2|0|4–6|39|
|5103|50|5|2|0|3–5|40|
|5104|47|2|0|0|2 exactly|43|
|5105|50|5|2|1|2–5|40|
|Total|248|23|8|1|14–23|202|

Thus only23/225 decisions, **10.22%**, invoke a second proposal, versus the development-matched100% rate. At least14 and at most23 are triggered by a high-risk first valid proposal. These are rigorous **bounds**, because the summaries do not distinguish whether an invalid or unavailable proposal was first or second. In5104, both other causes are absent, so its two retries are exactly attributable to first-proposal risk≥.5. This demonstrates that the uncertainty threshold affected behavior; it does not establish that the second proposal was ultimately selected.

There are202 accepted first proposals with risk<.5, but only six HVI-positive test queries in total. Therefore **at least196/225 non-LHS executions (87.11%) both had predicted no-HVI risk<.5 and actually delivered no HVI**. This bound follows from the executed-episode counts and unchanged controller logic. It is not an invented list of per-query scores, and it does not assert the stricter p≤.1 overconfidence statistic.

The224 actually scored full actions have weighted test Brier **.677841**, versus.097142 on development. All six positive-HVI actions occur in seeds with45/45 available selected risks; the sole unavailable selected risk is in the zero-gain5105 episode. Hence the scored set contains218 no-HVI outcomes and six HVI outcomes. Two **descriptive calibration references** on this same set are:

- constant risk.9, the development no-HVI prevalence: Brier **.031429**;
- constant risk1: Brier **.026786**.

Neither reference involves new model execution, retraining or test-driven threshold selection. They are scoring references on the already executed actions, **not counterfactual predictions that a constant-risk controller would achieve better HV**. They show that the poor held-out Brier is not adequately explained by class imbalance alone. Almost all valid test proposal graphs are available (239/240), including every valid graph in both negative seeds. The observed problem is inaccurate action-risk probabilities under the test/controller distribution, not wholesale missing attribution graphs.

The label direction and Brier definition are consistent in the frozen source; see [label_prior_check.md](label_prior_check.md) for exact line references. The150 collected calls are five independent30-query episodes, **not a150-row initial prior**. The train/dev episodes use an empty initial archive; held-out episodes use the strong550-row prior. Test selection also conditions on the controller and minimum-risk choice, unlike the original collection policy. **These source-visible distribution differences are hypotheses for the calibration deterioration.** Local exports do not contain the per-proposal feature/risk vectors needed to distinguish feature shift, calibration extrapolation, controller selection effects, or a specific feature-construction issue. Successful representation fidelity checks alone do not establish a useful failure predictor.

The random controller is actually always-two at its frozen p=1.0:450 proposal attempts over225 non-LHS decisions, compared with248 for full/UQ. Its mean gain29.314 micro-HV is above full22.692, although full wins three paired seeds, loses one and ties one. This control therefore does not supply a realized-compute-matched isolation of risk ranking, nor evidence that retry alone is consistently superior.

## Why ES selected G0

The registered full development criterion is mean querywise HV minus.1×Brier minus.1×invalid-proposal fraction, over seeds4101/4102. It is evaluated without the550-row held-out prior. The following decomposition is recomputed from the six actual development episode summaries, without replaying weights or using test outcomes:

| Full generation | Mean HV | Mean Brier | Invalid rate | Fitness | Fitness change fromG0 |
|---|---:|---:|---:|---:|---:|
|G0|.664038905|.090172436|0|.655021662|0|
|G1|.664040182|.166972047|0|.647342977|−.007678685|
|G2|.606028655|.009042416|0|.605124413|−.049897248|

- **G1:** HV improves by only.00000127648. The Brier penalty worsens by.00767996110, explaining its entire net fitness decrease. It did not lose because of invalid proposals.
- **G2:** HV falls by.05801025051, primarily through development seed4102 (.8418516→.7258311). Better Brier contributes+.00811300206, insufficient to compensate. It did not lose because uncertainty calibration became worse; its measured calibration improved.
- The maximum-development-fitness rule correctly choosesG0. The two actual update histories/weight hashes are different fromG0, but neither updated checkpoint is used in the final full evaluation.

Independent ES-only has a different objective that excludes Brier. Its first population has exactly equal rewards.4108557965/.4108557965, giving normalized rewards0/0 and zero parameter update atG1. AtG2 all723 parameter tensors have nonzero deltas, but mean development fitness drops from.6075581568 to.6060310543. G0 andG1 tie, and the registered earlier-generation rule selectsG0. These facts explain both full=UQ and ES-only=base in the accepted metrics; they do not justify retuning against negative held-out results.

## What remains unknown and the minimum next read

The existing local projections contain35 curves/summaries, graph-availability fractions, final fronts, risk development metrics, and ES development decomposition. They **do not contain the full chronological test query/proposal records**. Accordingly this report does not claim exact duplicate-condition frequency, first/second risk distributions, the count of changed candidate selections, risk-triggered misses, wall-time attribution, or per-feature causes.

[minimal_readonly_extraction_plan.json](minimal_readonly_extraction_plan.json) specifies a bounded optional read: existing750 query records for full/base/GP across all five fixed seeds, the248 already-existing full proposal records, and optionally the125 stored train/dev feature rows from risk_fit.json. It reads and reduces existing evidence only; no graph/model/oracle is invoked, no unselected outcomes are requested, no test-guided parameter change is proposed. The records would resolve:

1. Exact first-trigger cause and whether risk ranking actually changed the executed proposal.
2. Repeated executed conditions versus repeated generated/unselected conditions.
3. Per-query risk calibration and whether low-risk errors cluster by stored graph/activation features.
4. Whether early small-gain choices and later missed large gains correlate with risk selection, while preserving the distinction between observation and counterfactual causation.

Within the recorded workflow, the source and prior acceptance support a frozen common adaptation archive rather than test-outcome leakage. Collection/ES development observations intentionally enter the test-time archive after adaptation; no held-out5101–5105 observations enter fitting, selection or the frozen prior. This remains a warm-start study on one known function. [label_prior_check.md](label_prior_check.md) gives the source guards and limits of this assessment.

No remote read from that plan has been executed in this task. No method, training data, threshold, checkpoint selection, evaluator or scientific registration was changed.
