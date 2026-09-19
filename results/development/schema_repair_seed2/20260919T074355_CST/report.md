# Al–Pd–Sm 开发集 seed 2：schema 修复的机制核查

**这一个配对结果支持“修复了可复现的格式误拒绝”，尚不支持“NN 提供了有益的额外控制信号”。** 同 G2、同 Qwen3.5-4B、同 NN/TC、同环境 seed 2、各 50 次候选 ORB，原控制器 SUN **9** / AUDC **0.2388**，schema 修复后 **15 / 0.3476**。这是开发集诊断，不是测试集效果或跨 seed 结论。

原始结果快照时间为 **2026-09-19 07:43:55 CST**；目录中的 `20260920` 是任务命名，不是采样日期。本次只读这两个完成结果及其原决策、RPC、oracle journal、冻结源码；没有读取 30 个测试体系做选择。两份完成结果的 execution_profile 均与注册逐字段相等；实际 G2 state hash 为 `2e5adce2a0d3785a1e1df3c41888f42eed2a23b9e175e3e0b1a4e1d3ef534fff`。原始文件路径/SHA、选中 checkpoint、共同 profile 见 [summary.json](summary.json)、[identity_evidence.json](identity_evidence.json)。此次未重读大型模型权重。

| 对齐口径 | 原控制器 | schema 修复 |
|---|---:|---:|
| 候选 ORB：发起并成功返回 | 50 / 50 | 50 / 50 |
| 初始化 ORB，另计 | 26 | 26 |
| MACE surrogate，另计 | **0** | **31** |
| LLM proposal 数 | 88 | 84 |
| 决策组 / 产生第二 proposal 的组 | 75 / 13 | 79 / 5 |
| controller 请求 retry 的候选事件¹ | 20 | 6 |
| 本地 schema 拒绝事件 | 18 | 3 |
| 其中错误拒绝 adapter 支持的 Mapping | **16** | **0** |
| 真实 generation failure | 2 | 2 |
| 固定恢复动作 | 2 | 0 |
| 实际工具调用 / 工具错误 | **71 / 6** | **79 / 10** |
| 图成功 / 图不可用 | 67 / 21 | 79 / 5 |
| 其中原 fidelity gate 不通过 | 20 | 3 |
| 生成晶体候选 / buffer 接受 | 280 / 176 | 444 / 169 |
| 单次评估时稳定 | **39/50 = 78%** | **44/50 = 88%** |
| 官方结构唯一数 | 18 | 17 |
| 未增加结构唯一计数的评估次数 | 32 | 33 |
| 相对初始参考库 novel | 50/50 | 50/50 |
| 稳定且唯一且 novel，即最终 SUN | **9** | **15** |
| mSUN / AUDC | .18 / .2388 | .30 / .3476 |

¹候选事件与实际追加 LLM 调用不同：第二候选也记录 retry 请求，但最多两个候选的原上限不变。两条 RPC 均完整 close；100 次候选 ORB、52 次初始化 ORB、31 次 MACE 均与真实 journal 的唯一 started/returned attempt 对齐，未闭合 attempt 为 0。AUDC 按冻结公式 `2 × trapezoid_area / B²` 从各 51 个原始曲线点重算。

**提升体现为更好的稳定结构组成，不能说“找到了更多不同结构”。** 最终唯一数 18→17，稳定且唯一数 9→15；所有 50 次评估都相对初始参考库 novel，不等于每次在本轨迹内都不重复。官方稳定性 tolerance 为 0.1 eV/atom；唯一性按官方 StructureMatcher，novelty 的 reference 是 `initial_structures`，源码与字段依据见 [identity_evidence.json](identity_evidence.json)。buffer 拒绝 104→275 条只被记录为 `official_filter_rejection`，本审计不将这个合并原因强行分解为稳定性或新颖性。

三条可定位的机制证据（完整小记录见 [cases.json](cases.json)）：

1. **相同首个 proposal，局部错误分支可直接定位。** 两臂 `d0000000-c0`（各 decisions 第 1 行）prefix、解析 action、采样 seed、scalar 与 typed risk 全相同。action 是 8 个非空 composition Mapping 的 `generate_structures`，scalar **0.4991152585**、typed `generation_invalid≈6.25e−234`，图有效。原控制器仍以 `explicit_compositions_required` 拒绝；修复后保留，RPC **1** 实际生成并接受 32 个候选。冻结 adapter 本来就对每个输入调用 `Composition(formula)`，支持这种 Mapping；修复只纠正 shape 判断及相同 shape 的 advisory preference。原轨迹先执行空 buffer 查询、产生一次非法 formula 工具错误，并在 RPC **9** 走固定 random recovery。首个 ORB 分别为原 RPC **12**（不稳定，e_above_hull **1.61748**）及修复 RPC **4**（稳定，**0.06242**）。这证明局部门控原因；两个实际评估结构不同，不能将其当作同一晶体的反事实实验。

2. **schema 修复没有解决“有效 action + 缺图”误报。** 修复轨迹 `d0000021-c0`（decisions **25** 行）是有效 `select_for_evaluation`，但 layer 16 FVU **0.500933** 超出原 gate，图不可用。typed `generation_invalid=1`，尽管观测 generation-invalid 标签为 **0**、两版 schema 都无问题。它被替换为 `c1` 查询，RPC **43** 成功；随后 RPC **46**（第 13 次 ORB）是稳定但非本轨迹新结构，SUN 留在 5。被拒候选这一步未执行，不能推断其潜在 SUN。误报来自缺图情形的预测，不是发现真实 JSON 格式错误。

3. **仍有真实工具失败，NN 排序未避免。** 修复轨迹已完成 47 次 ORB 后，RPC **137,139,…,151** 连续 8 次报 `The selected hash is not in this composition buffer`，预算计数一直为 47。最后 `d0000075-c0/c1`（decisions **80/81** 行）都是有效 JSON、均缺图；两者 `generation_invalid=1`，给出格式修复反馈。排序保留 c0，但 RPC 151 仍失败。其 tool-failure 预测 .9173 确实对应真实工具错误，不能据此说所有 head 都没有信息；然而这次控制并未排除错误。最终 RPC **154**（第 48 次 ORB）完成，SUN 仍 15；最后一次 SUN 增加在第 46 次。

NN 的可支持结论很有限。有效 generation 的 scalar 原为 **.498940–.501051**、修复为 **.499115–.501027**，均低于固定阈值 **.6**。有效且图成功的 66/79 条 proposal 上，scalar 和所有可用 typed head 均没有达到阈值。实际 `predicted_failure_type` retry 是原 **2**、修复 **3** 个候选事件，全部在缺图时把成功 generation 误报为 `generation_invalid`。原图缺失的 typed generation-invalid 假阳性共 3 条，其中 1 条已由本地 schema 错误先触发；修复的 3 条均为该假阳性。两有效候选的风险排序只出现原 1 组、修复 2 组，均卷入缺图，未形成独立的正向信息收益证据。**这不证明 NN 在统计上完全无信息；它说明本配对的阈值/排序行为没有提供可归因的 NN 增益证据。**

成本必须保留：固定的 50 次候选 ORB 并不意味着等总计算量。修复多做 **31 次 MACE**（一条 `score_buffer` RPC）；prompt tokens 173,354→188,217，completion tokens 11,285→12,149，graph seconds 4,920.9→5,629.5，记录 wall 107.1→120.1 分钟。两臂在不同执行位/共享负载下运行，wall 只作为实际成本记录，不能作纯算法速度比较。随后历史、action、随机流都会因首处分支改变；不能从单条轨迹将 +6 SUN 分摊给 schema、MACE 或 NN。50 个步骤也不是 50 个独立重复；这里只完成 **1 个配对 seed**，run-to-run variance 不可估。

可用于论文的校准表述：*In one paired development rollout (Al–Pd–Sm, seed 2, 50 candidate ORB evaluations), correcting an adapter-compatible composition-schema rejection increased SUN from 9 to 15 and AUDC from 0.2388 to 0.3476. An identical first proposal isolated the erroneous local gate, while the unchanged neural risk outputs did not provide evidence of beneficial threshold-based control in this pair. Missing-graph false alarms persisted, and the repaired trajectory incurred 31 additional MACE screening evaluations; the result does not establish multi-seed efficacy or the independent value of uncertainty prediction.*

新 support-aware 方法针对“缺图应为 unknown”的处理是另一项已冻结候选，**效果仍待其独立主实验**，不能用本 schema-only 配对代替。只使用原 train/dev 的下一步信息诊断见 [nn_diagnostic.md](nn_diagnostic.md)。复算：`python3 analyze.py`。该命令只读取本目录缓存，不联网、不训练、不执行 oracle。
