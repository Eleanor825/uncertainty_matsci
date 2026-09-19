# SnAr 已完成正式结果：论文表与 48 小时内的取舍

**第二个 benchmark 已经完成：Summit SnAr 的 35/35 条正式 held-out 轨迹，7 个方法 × 5 个 evaluation seeds × B50。** 这不是 pilot 或 smoke。本次只复算已有结果，没有新模型、训练或 oracle 调用；它也不是整个 Summit 多任务套件。测试策略的基础模型是 Qwen3.5-4B，确定性 SnAr 反应函数只有一个。独立验收在 2026-09-19 16:25:48 CST 完成。

核心结论：**GP-EI 的平均发现增益最高；完整方法的平均增益低于 Qwen baseline。UQ-only 与完整方法曲线相同，ES-only 与 baseline 曲线相同；两个 ES 分支的开发集规则均选择 G0。现有 SnAr 结果不能支持 ES 提升或完整方法普遍优于 baseline 的结论。** 它可以作为论文中的第二个、包含负结果的正式 benchmark。

## 可直接用于论文的主表

每行 n=5，方差是 evaluation seeds 5101–5105 的样本方差（ddof=1），不是体系间方差，也不是重复训练的方差。表中**均值与 SD 的单位为 10⁻⁴ HV；方差的单位为 10⁻⁸ HV²**，不是百分数；全部原始未缩放值见 [table.csv](table.csv)。LaTeX 片段见 [table.tex](table.tex)，使用 booktabs 的 table*，可直接 input。

| 方法 | 最终 HV 增益：均值 ± SD | 样本方差 | 平均逐 query HV 增益：均值 ± SD | 样本方差 |
|---|---:|---:|---:|---:|
| Random | 0.2640 ± 0.4972 | 0.2472 | 0.1805 ± 0.3726 | 0.1389 |
| GP-EI (fixed scalarization) | 3.6706 ± 1.0385 | 1.0785 | 2.8679 ± 1.0423 | 1.0863 |
| Qwen baseline | 0.5310 ± 0.5867 | 0.3442 | 0.1627 ± 0.1676 | 0.0281 |
| UQ-only | 0.2269 ± 0.4130 | 0.1706 | 0.1492 ± 0.2892 | 0.0837 |
| Independent ES-only (G0) | 0.5310 ± 0.5867 | 0.3442 | 0.1627 ± 0.1676 | 0.0281 |
| Random controller | 0.2931 ± 0.6292 | 0.3959 | 0.1146 ± 0.2479 | 0.0615 |
| Full: UQ + ES (G0) | 0.2269 ± 0.4130 | 0.1706 | 0.1492 ± 0.2892 | 0.0837 |

所有方法先验相同：550 次已完成适配查询，初始 HV=0.8810117795428745。主指标为相对这份先验的最终 HV 增益；副指标为 50 个 query 后的 HV 增益均值。绝对 HV 的大部分来自共同先验，不能只报告约 0.881 的最终值来遮蔽增量差别。

测试支出为 1,750 次查询，加 550 次适配查询，共 2,300 次独立物理 oracle 调用。每条 B50 内含 5 次 paired LHS 初始设计。B10/B30 是这批 B50 的前缀，**不是独立运行的三个预算实验**。新推导表不增加支出。

GP-EI 是预先固定乘积标量化的 GP expected improvement，不是 EHVI/TSEMO。GP 接收全部 550 行原始先验，LLM 接收同一先验的固定至多 4 点 Pareto 摘要。这个输入表示差别属于原登记协议，不能把 GP 的优势唯一归因于 uncertainty。各 LLM 方法最大允许两次候选提议；random-controller 匹配开发集估计的预期触发率，而非每条测试轨迹实际的完全相同 token/推理开销。

## 逐 seed 的正负结果

以下仅显示最终 HV 增益，单位同为 10⁻⁴；[per_seed.csv](per_seed.csv) 保留全部 7 臂、两种指标和逐 seed 配对差，原 35 条独立执行记录也在 inputs/per_seed.csv。

| Seed | Baseline / ES-only | UQ-only / full | GP-EI | Full − baseline |
|---|---:|---:|---:|---:|
| 5101 | 0.0000 | 0.0696 | 1.8989 | 0.0696 |
| 5102 | 0.5067 | 0.9643 | 4.3230 | 0.4576 |
| 5103 | 0.7409 | 0.0503 | 4.0914 | -0.6906 |
| 5104 | 1.4074 | 0.0503 | 3.6147 | -1.3571 |
| 5105 | 0.0000 | 0.0000 | 4.4252 | 0.0000 |

完整方法 − baseline 的最终增益平均差为 −3.0408221000×10⁻⁵，配对差的样本方差为 5.1761060861×10⁻⁹；逐 query 增益平均差为 −1.3481530699×10⁻⁶，方差为 9.9791335257×10⁻¹⁰。两指标均为 2 胜、1 平、2 负，不能仅用两个正 seed 声称总体改善。五个 seed 共享一次固定的表示训练、NN 训练和 ES 适配，因此这些方差不包含适配/重训随机性。

## ES 训练与测试权重：避免把 G0 写成 ES 获益

| 分支 | G0 开发 fitness | G1 | G2 | 选中 | 真实权重变化 |
|---|---:|---:|---:|---|---|
| Full | 0.6550216616 | 0.6473429770 | 0.6051244131 | G0 | G1、G2 均改变权重 |
| ES-only | 0.6075581568 | 0.6075581568 | 0.6060310543 | G0 | G1 的两个 population reward 相等，归一化后为 0，更新为 0；G2 的 723 个参数张量均有非零变化 |

两条分支都实际运行了独立 ES 开发过程；不能写“没有做 ES”或“训练从未更新权重”。准确说法是：**最终测试没有采用改变后的权重，因注册的开发集选择规则选回原始 G0。** Full 与 UQ-only 在 5 个 seed 的 50 点 HV 曲线及权重 hash 完全一致，ES-only 与 baseline 同样一致；这些独立执行的重复表现不能充作额外有效样本。全体方法还共享 ES 收集数据贡献的 550 行先验，因此也没有隔离 ES 作为数据收集过程的边际价值。

G1 full 的 HV 比 G0 仅增加约 1.2765×10⁻⁶，Brier 惩罚却恶化约 0.007680，导致 fitness 下降。G2 full 的 HV 下降约 0.058010，较好的 Brier 也未能抵消。ES-only 的 G2 开发 HV/fitness 则下降约 0.001527。这些是已记录开发指标的损失分解，不是根据测试负结果重选 checkpoint。

## 完成、缺项和 48 小时内的最小补充

已完成：7 臂主比较；UQ-only、独立 ES-only、random-controller 消融；5 个 evaluation seeds；完整 B50 轨迹及 B10/B30 前缀；训练/开发选择证据；独立验收；固定动作上的 NN/特征族诊断。**不需要为了“补齐 SnAr 消融”再原样重跑 7×5 矩阵。** 这里没有第三个 benchmark 的结果，也没有更多模型尺寸结果；不能借 SnAr 的多个 seeds/多个臂计算为多个 benchmark。

未被证明：内部特征/NN 导致更好发现、ES 选中权重带来改善、跨反应函数或跨模型尺寸泛化、跨独立重训的稳定性。当前 UQ 测试 Brier 的按 seed 均值为 0.677794；224 个实际有风险值的动作加权 Brier 为 0.677841，不能混淆分母。开发集 Brier 为 0.097142，但这 50 行还用于 epoch selection/temperature calibration，所以不是独立的校准测试估计。原标签方向均为 P(no-HVI=1)，源码和计算没有发现反向标签或两倍 Brier 的不一致。

源代码显示明确条件差异：原风险收集/开发和 ES 开发从空外部 archive 开始，正式测试从完整 550 行 archive 开始，并且测试动作经过控制器选择。其影响是**机制假说**，不是本报告证明的因果解释；550 行在测试前冻结，已有验收支持未混入 held-out 测试结果。不能把旧收集行简单对包含其自身的 550 行先验重标并当作新的独立验证。

建议在 48 小时内按以下顺序取舍，均需先登记，**本次没有执行这些补充**：

1. 立即把现有表和负结果写进论文，并将已完成的固定动作风险/特征诊断放附录。以第二 benchmark 完整报告为准，不以“必须正向”作为完成条件。资源优先给第三个 benchmark 的真实核心比较和当前 MADE 未完成主矩阵。
2. 若仍要补 SnAr，首先只做一个**条件匹配的全新 development 诊断**：使用冻结的 550 行先验、G0 和既有风险模型，预先登记例如 3 个新 development seeds × B50=150 个 oracle queries，记录校准、正负类别支持及实际重试，保持原 35 条测试不变。这能定位在真实先验条件下概率是否仍失准；不能直接承诺改善。如果正例过少，报告支持不足，不继续挑 seeds 直到有效。
3. 只有上述开发证据支持进一步验证且时间允许，才登记一个小的 **G0 + 风险模型控制器 vs G0 + 风险屏蔽控制器** 配对比较，保持 schema/工具恢复/候选上限/预算/先验相同，明确其比较的是风险使用的总效果而非固定 token 成本。新的独立 held-out 3 seeds ×2 臂×B50为300次新增查询；若使用开发数据重校准，则必须先冻结方案，不能用原测试集调温度、阈值或选 feature family。该补充不证明 ES，暂不再跑 ES 大矩阵。

150+300 是候选的**新增明确调用预算**，不是已完成结果，也不是可信 GPU 耗时估计。若只能做一项补充，选择第2项而非重新进行无目标的整套七臂搜索。若目标是直接证明 NN 因果贡献，最终仍需要第3项独立的在线对照；固定动作 AUC 或 feature-family 重拟合不足以替代。

## 可复核证据与使用边界

- [原正式报告](https://github.com/Eleanor825/uncertainty_matsci/blob/736d96be6438992c59bc498754abc6dc7dcce3e1/results/summit_snar/20260919T162548_CST/report.md)、[35 条原值](https://github.com/Eleanor825/uncertainty_matsci/blob/736d96be6438992c59bc498754abc6dc7dcce3e1/results/summit_snar/20260919T162548_CST/per_seed.csv)、[完整方差](https://github.com/Eleanor825/uncertainty_matsci/blob/736d96be6438992c59bc498754abc6dc7dcce3e1/results/summit_snar/20260919T162548_CST/statistics.csv)。
- [原协议](https://github.com/Eleanor825/uncertainty_matsci/blob/736d96be6438992c59bc498754abc6dc7dcce3e1/results/summit_snar/20260919T162548_CST/protocol.json)、[先验/full 选择](https://github.com/Eleanor825/uncertainty_matsci/blob/736d96be6438992c59bc498754abc6dc7dcce3e1/results/summit_snar/20260919T162548_CST/prior_and_selection.json)、[ES-only 真正更新与选择](https://github.com/Eleanor825/uncertainty_matsci/blob/736d96be6438992c59bc498754abc6dc7dcce3e1/results/summit_snar/case_analysis/20260919T2023_CST/inputs/es_only_development.json)。
- [标签、Brier 与先验条件审计](https://github.com/Eleanor825/uncertainty_matsci/blob/736d96be6438992c59bc498754abc6dc7dcce3e1/results/summit_snar/case_analysis/20260919T2023_CST/label_prior_check.md)、[开发损失分解](https://github.com/Eleanor825/uncertainty_matsci/blob/736d96be6438992c59bc498754abc6dc7dcce3e1/results/summit_snar/case_analysis/20260919T2023_CST/development_selection_decomposition.csv)。

原报告 job 的 ModuleNotFoundError 仍被保留；后续独立 source-bound CPU wrapper 对同一科学证据验收通过，不代表旧 job 自己成功。原接受包的35轨迹/1750曲线点/2300调用/28项报告指标集合/16文件哈希已再次通过其 stdlib verifier。本包保留8个原字节科学输入和其来源 SHA；重算脚本从完整曲线重算两指标，再验证全部7臂均值/样本方差/SD、配对差、G0选择和曲线一致性，并逐字比对派生表。它不读取凭据/进程材料，不加载权重，不改变科学数据。

```bash
python3 -B recompute.py
```

本目录为经核验的新增公开报告和论文表；原35条正式结果没有被覆盖。
