# MADE 负结果：已测机制与未证实根因

本次汇总并发布既有证据，0 新模型、0 环境调用，原结果不改。**已能定位风险控制没有充分生效、部分搜索分布退化，以及一个可直接测试的 G0 选择缺口；尚不能把整个 1079 个有效结果的下降归于一个已证实原因。**

最新注册 1080：1079 accepted、1 baseline 技术失败、0 pending。失败未当零分；B10/B30/B50 配对分别 90/90/89。CSV 独立重算 Full−Baseline 平均 SUN 为 **−0.433333/−0.600000/−1.573034**，UQ-only 为 −0.155556/−0.333333/−0.449438。不同臂独立训练/轨迹，不能把相减直接解释为 NN 的因果贡献。[latest_report](../../../results/made_components/incremental/20260922T101850_CST/report.md)；[latest_pairs](../../../results/made_components/incremental/20260922T101850_CST/paired_results.csv)。

| 机制 | 已有直接证据 | 能下的结论与边界 |
|---|---|---|
| scalar 预测与控制弱 | 72 Full B10 中成功生成风险 .498467–.501268，阈值 .6，scalar 触发 0；527 个受支持直接材料选择 AUROC .492701、Brier .249796，实际失败率 .842505 | 概率范围与已测排名指标分别报告；不是由接近 .5 推出随机 AUC。此队列确实没有 scalar 控制收益，不能外推每一条新结果。 |
| 图缺失限制重排 | Full 1340 proposals：905 图支持、435 无支持；21 个 typed 追加候选中 20 个无图；实际选择非首个合法候选的受支持风险重排 0 | 未知保持未知的保护门真实减少了可比较候选；不能因此降低 FVU/数值门。21 个候选未执行，无稳定性/新颖性反事实，**“错拒稳定材料”未证实**。 |
| 历史错误格式拒绝 | 相同 G2 的一个 B50 开发配对中，合法 composition mapping 错拒 16→0，SUN 9→15、AUDC .2388→.3476 | 是已定位/修复的接口问题；当前 support-aware 1079 不是仍用旧错误规则，不能用它解释全部当前差距，也不能把 +6 SUN 全算作 NN 收益。 |
| 搜索集中、稳定新材料减少 | 独立旧 seed1 B50：3000 个 ORB 中两臂各 1500；stable-new 237→148，非新颖 183→417；每轨迹平均不同公式 21.63→12.73，重复字面 candidate hash 0 | 有探索分布集中证据，没有字面重复执行证据；公式重复也可能是新多晶型。权重、规则和随机路径共同变化，“NN 太保守”未隔离。 |
| 凸包重分类 | 旧 B50 stable-new 事件差 −89，后续移出凸包 baseline4、Full1，最终 SUN233→147，差 −86 | 重分类贡献 +3，方向反而有利 Full；无法解释主要下降。3000 步 CPU LP 复核最大 hull 差 1.33e−14。 |
| ES 选择缺失 G0 | 新 Full 原选择只比较 G1/G2；后验同 AlPdSm/B10 开发 seed 的 G0 AUDC .68 > G1 .62/G2 .64；实际选 G2 | 最具体可测试缺口：保留 incumbent G0。仅一个已用 dev seed，不保证跨 seed 改善，不回改旧选模。 |
| ES 目标/资源 | 实际 fitness 是官方 AUDC，uncertainty_reward_weight=0；6 个 train/dev trace 风险 retry/rerank 全 0；P2×G2，单 train/dev chemistry、B10 | 不成立“把 MACE 或 NN 分数误当 ES reward”。小种群、窄训练域、短时域是合理待检假设；不是已有因果结论。两臂 ORB 预算均完整用完，图慢并未使这批少做 ORB。 |

前两行来源 [mechanism_report](../../../results/made_components/mechanism_analysis/20260920T043843_CST/REPORT.md)（24 化学体系×3 seeds，固定历史144轨迹，非全1079）；历史格式 [schema_report](../../../results/development/schema_repair_seed2/20260919T074355_CST/report.md)；旧 B50 [old_B50_report](../../../results/budget_sweep/B50_mechanism_analysis/20260920T054512_CST/REPORT.md)；ES [ES_mechanism_report](../../../results/development/support_mechanism_audit/20260919T174119_CST/REPORT.md)、[G0_report](../../../results/development/G0_comparison/20260919T200818_CST/report.md)。

MADE 标签直接是 `not(stable and newly_discovered)`，合法但不新/不稳定的探索也记失败；执行的前序工具共享随后物理评估标签，代码明确它不是每一步推理错误的因果证明。格式、工具失败另有 typed 标签。见 [rollouts.py:380](code/historical_excerpts.json)、[rollouts.py:468](code/historical_excerpts.json)。目标和官方 SUN 有联系，但不等同“工具调用正确性”。

与 DW 的共同点是“可预测的失败”必须通过实际有用的动作变化才能改善终点。**差别**：DW 已测高失败 AUROC 仍可没有进展/有效改动作；这里 MADE 原风险网络已有低排名表现且 scalar 门不触发。不能把 DW .9867 阈值、即时 API 失败标签或末层选点证据直接套给 MADE。当前 MADE 数据不能证明图普遍无用；其有效训练只有 1 种 chemistry/3 episodes。下一最小方案见 [safe_incumbent.md](safe_incumbent.md)，训练细节见 [critic_audit.md](critic_audit.md)。

选择器现已实现并完成本地元数据复现：原规则 G2(.64)>G1(.62)；包含 incumbent 后 G0(.68)>G2>G1。新代码/原排名与新排名见 [选择回执](selection.json)、[原字节源码](code/selector.py)、[可移植校验](validate.py)。这不是新 heldout 实验，也没有替换生产模型或证明超过 Native。
