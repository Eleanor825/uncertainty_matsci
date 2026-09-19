# SnAr sampling+hidden补充诊断：三次实际拟合，图特征增量仍不稳定

原新增诊断的prepare/run/audit均以rc0完成，实际记录结束时间为 **2026-09-19 22:10:18.444778 CST**。新增3个132维sampling+hidden风险NN共执行180个CPU优化步，原12个模型没有重拟合。该包只重算已保存预测，不调用模型、生成器、图或oracle。

**原all285模型相对sampling+hidden132的AUROC配对差为两正一负，平均+0.048930；Brier、NLL、ECE平均反而更差。** 新union模型自身的平均Brier为0.541182，仍差于常数0.5预测的0.25，更远差于训练类先验70/75的0.027659。新增这项对照没有证明可靠校准、不变的图增益或材料发现策略收益。

## 同一动作集合上的结果

两组比较都使用同一224个原full策略已执行且有完整特征的动作，其no-HVI标签为218个1、6个0。它们来自原五个policy seeds，而不是224次独立实验。与前12模型合计的3,360条预测仍对应这224个唯一动作；不新增物理轨迹。

| NN seed | Union AUROC | 原all AUROC | all−union AUROC | Union Brier | 原all Brier | all−union Brier |
|---|---:|---:|---:|---:|---:|---:|
|1729|0.722477|0.791284|+0.068807|0.388372|0.677841|+0.289469|
|1730|0.715596|0.808104|+0.092508|0.654742|0.729651|+0.074909|
|1731|0.713303|0.698777|-0.014526|0.580433|0.379225|-0.201208|

AUROC越高越好；Brier/NLL/ECE越低越好。因此Brier差为正表示原all更差，不能把正差误当收益。seed1731中原all的AUROC较低，但Brier也较低，排序与概率误差给出的比较不同。

## 三NN seed均值与样本方差

方差按三次NN初始化/训练seed计算，ddof=1；不是物理实验重复方差或置信区间。

| 指标 | Union均值 | Union方差 | 原all均值 | 原all方差 |
|---|---:|---:|---:|---:|
|auroc|0.717125|0.0000227955|0.766055|0.0034655005|
|brier|0.541182|0.0188936567|0.595572|0.0357758127|
|nll|1.418331|0.1640241087|1.726120|0.4643011617|
|ece|0.701792|0.0100041140|0.733040|0.0189507318|
|error_auprc|0.986707|0.0000005289|0.991456|0.0000127704|
|risk_coverage_auc|0.947488|0.0000151063|0.951663|0.0000323355|

| 配对差 all−union | 均值 | 样本方差 | SD | 三seed方向 |
|---|---:|---:|---:|---|
|auroc|+0.048930|0.0031603915|0.056217|2正/1负|
|brier|+0.054390|0.0605067943|0.245981|2正/1负|
|nll|+0.307789|0.5828286482|0.763432|2正/1负|
|ece|+0.031248|0.0340168327|0.184437|2正/1负|

**容量仍不相等：union为21,377参数，all为40,961参数。** 两组都使用hidden width64，但输入维度不同；相同NN seed只是配对编号，不代表它们的初始权重相同。训练期标准化、缺失指标和error prototypes均在选定特征后拟合，union不含graph输入。这个已注册补充比之前仅比较孤立families更直接地考察图相关输入的增量，但仍混有容量、优化和选择偏差，不能称作纯图信息的因果隔离。

## 实际训练与开发集

新注册固定3个seed、最大300优化步，实际为61+60+59=180；加原746步，15个NN累计926步。每个模型的75条训练数据为70个no-HVI/5个改进，50条development为45/5。development参与epoch选择和温度校准，相关分数为选择/校准内样本。

| NN seed | 选中epoch | 实际优化步 | 温度 | Dev AUROC | Dev Brier | Test p≥0.5数量 |
|---|---:|---:|---:|---:|---:|---:|
|1729|46|61|1.142770|0.568889|0.093427|24/224|
|1730|45|60|1.067842|0.564444|0.092166|17/224|
|1731|44|59|1.143410|0.582222|0.091172|18/224|

Union的raw Brier均值0.568541，经原development温度校准后为0.541182，误差略降但仍很大。三个NN在development的Brier约0.09，在所选test动作上为0.388372/0.654742/0.580433，全部大于0.25。固定0.5阈值下只有24/17/18个动作被判为高no-HVI风险，而实际有218个no-HVI结果；这些只是对既有动作的回看分类，不是新策略的retry次数。

原物理full及独立ES-only都选择G0/θ0，full=UQ、ES-only=base的物理HV曲线重合，本新增NN诊断不改变这件事。不能把ES权重漂移作为这里的分布变化解释。原训练collection的空archive与test的550点共同prior、控制器选样以及轨迹历史是可讨论的分布差异，但此处未通过干预证明任何一个是概率误差的原因。

## 完整分层与校准数据

[predictions.csv](predictions.csv)保留672条新模型预测（每条含原raw/calibrated/logit值）；[development_predictions.csv](development_predictions.csv)保留150个开发集预测及episode/prefix身份。新模型完整pooled指标、开发指标和五policy-seed分层分别见 [model_metrics.csv](model_metrics.csv)、[development_metrics.csv](development_metrics.csv)、[policy_seed_strata.csv](policy_seed_strata.csv)。policy seed5105没有改进类别，AUROC保持NA，未补成0或0.5。

[calibration_bins.csv](calibration_bins.csv)有420行：3模型×raw/calibrated×（pooled test+五policy-seed test分层+development）×固定10个等宽bin。空bin的均值保持缺失；不使用逐动作独立性置信区间。匹配比较在 [paired_models.csv](paired_models.csv) 和 [paired_summary.csv](paired_summary.csv)，原all值来自已公开的12模型冻结报告，不重新训练。

## 审计、原字节与复算边界

新增8份原始JSON共237,126bytes，以确定性gzip原字节保存，包括registration、3份fit metadata、modelset、report、completion、audit。新audit是实际原runner对保存模型重建完整报告后进行的严格publish-format字节比较，记录passed，未新增数值容差。其report SHA为`9838d2addb473446acc7a76dacf4cb9d1585915ba4e8fd54ea0629dac52eb61e`；audit SHA为`5fb6191c44f9a8ac8d9b38fbba4e0a08266f3d059252f9cee1f23f1532eebd9a`。

这与父12模型历史audit rc1是两回事：父失败及其独立byte-exact重建验收都保留；新audit自身rc0，不能混称两者全部原审计通过。

`python3 -B recompute.py`使用Python标准库独立重算保存概率的指标，共588项逐点指标/汇总比较通过，最大浮点差4.4409e−16，独立实现舍入上限1e−12。这个检查不加载`.pt`或重拟合，不声称重新执行了原checkpoint字节验收，也不放松其原始标准。`--write`仅重建本包派生CSV/JSON。源gzip、原科学runner/plan/manifest与父engine/runner/NN实现均有SHA；它们是历史出处，不应直接作为异机重跑入口。

完整记录见 [source_manifest.json](source_manifest.json)、[provenance.json](provenance.json)及 [MANIFEST.json](MANIFEST.json)。为了审查代码覆盖，包内保留新union的原runner、plan和source manifest；source manifest中涉及的外部验证/父重建/运行资产依赖仅作引用，不伪造旧工作区、收据或资源。复算指标无需这些模型权重；重新拟合/复现checkpoint则仍需明确的原输入、CPU ML运行环境、旧模型依赖及新的合法注册。

这是在观察原实验后追加的posthoc诊断。即使计算计划在3个新拟合之前冻结，也不能据此升级为前瞻策略试验。没有从这些test结果选择新阈值、family、checkpoint或部署controller。所有负差与单类/空bin缺失保留；本包经root审阅后独立公开，未覆盖主实验或此前诊断。
