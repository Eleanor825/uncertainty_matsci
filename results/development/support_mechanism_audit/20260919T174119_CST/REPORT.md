# 固定17体系四臂与新full六条训练/开发轨迹的机制审计

**本轮实际训练没有发生神经风险触发的重提或风险排序改选；ES确实更新了全部参数块，但优化的轨迹主要受模型自身工具选择、可观察的格式规则和无图中性处理影响。** 测试上的工具使用与图支持变化是具体线索。现有数据不能把full相对UQ的下降单独归因于“NN风险错误”，也不能证明更多MACE一定更好。

证据固定于17:26快照的17个B10/seed2完整四臂交集，实际读取于2026-09-19 17:41:19 CST；共68条测试轨迹和新full的6条train/dev轨迹。只读取既存JSON/JSONL及源码，没有模型权重读取、推理、拟合或Oracle调用。原始回传SHA为`2d1c1ff98a9f107289f039048f9430652d27b67d662da44d698e73d582db92fb`。完整来源在[原始提取SHA及来源证明](source_provenance.json)，汇总在[analysis.json](analysis.json)，全部68条同口径数据在[CSV](fixed17_all68_mechanisms.csv)。这是完成交集的事后描述，不是新的前瞻性测试集。

## 固定全体，不挑选案例

| 同17体系、B10、seed2 | SUN总数 | 平均AUDC |
|---|---:|---:|
| baseline | 34 | .202353 |
| UQ-only | 40 | .252941 |
| 独立ES-only | 24 | .209412 |
| full | 27 | .174706 |

full减UQ为SUN−13、平均AUDC−.078235。全部负例是Al-V-Zn、Au-K-Tb、Au-Tb-V-Y、Ce-Ir-Pt-Sn、Co-Dy-W、Ga-Ho-Lu、Mg-Sn-Sr；全部正例是Ce-Er-Pb-Rh、Co-Mg-Na、Ga-Pt-Tm、Hf-Ni-Zr；六个持平体系是Al-Li-V、Au-Cr-Cs-Dy、Ba-Be-Hf-Li、Ba-Nd-Ni-W、Ca-Pd-Sn-W、Co-Pd-Tl。两指标的符号分组在本交集中一致。完整名单及delta见[fixed17_cases.json](fixed17_cases.json)。

## 哪些控制来自NN，哪些只是规则

两臂的controller、NN参数/温度/阈值及TC bank相同；UQ-only保持theta0，full加载新独立ES选择的G2。`arm_runtime.py:50–78`明确此路径。不是把原旧G2换名字，也没有重新拟合NN。

- `support_control.py:39–73`先保存原NN诊断输出。图不支持时，将用于控制的scalar/type概率置为None，取消“risk_unavailable”重提。真实schema错误仍单独纠正。没有用缺图伪造高风险。
- `support_control.py:76–97`在任一提案无图支持时禁用跨候选风险比较，按schema有效性及原候选顺序选择。这是未知状态的规则，不是学到的低风险。
- `failure_controller.py:235–260`区分`local_schema_validation`、`predicted_failure_type`与scalar overall-risk。composition Mapping兼容、JSON/参数检查、工具限额及固定恢复不属于神经风险学习。
- `rollouts.py:298–377`只有剩余原候选槽可重提。请求retry不等于真正多生成一次；最后槽的请求不会增加预算。advisory tool preference只是下一次prompt建议，并不直接执行工具。

| 17个测试体系 | UQ-only | full |
|---|---:|---:|
| proposal / decision group | 390 / 359 | 323 / 306 |
| scalar达到原.6阈值 | 0 | 0 |
| NN typed retry请求（全部not_new） | 5 | 8 |
| 实际NN触发额外提案 | 1 | 5 |
| 实际local-schema额外提案 | 30 | 12 |
| 全部候选有支持、至少两个合法候选的风险比较 | 1 | 1 |
| 风险排序实际选择非首合法候选 | 1 | 0 |
| 有效生成且有图 / 全部有效生成 | 340/372（91.4%） | 223/313（71.2%） |
| 有效生成但无图 | 32 | 90 |

UQ唯一一次风险排序改选发生在Ba-Nd-Ni-W，该体系两臂最终都是0；不能由此声称避免了失败。full的实际NN额外提案位于Au-K-Tb两次、Ba-Be-Hf-Li一次、Co-Dy-W一次、Co-Mg-Na一次，既包括负例、平例，也包括正例。拒绝提案没有反事实Oracle结果，不能计算“被NN挽救的材料数”。

## 工具轨迹变化有实证，单向因果解释不成立

| 17体系的真实RPC/物理计数 | baseline | UQ-only | ES-only | full |
|---|---:|---:|---:|---:|
| score_buffer工具调用 | 57 | 36 | 27 | 12 |
| MACE物理attempt | 2234 | 1419 | 718 | 381 |
| generate_structures工具调用 | 112 | 109 | 100 | 93 |
| select_for_evaluation请求 | 199 | 189 | 222 | 192 |
| 重复成功选择同一显式structure_hash | 0 | 0 | 0 | 0 |

每臂依旧是170次候选ORB。MACE是额外筛选工作，不是主Oracle预算；工具次数不等于所评结构数。成本字典缺少surrogate字段的地方保留None，表中的0只来自已完成实际journal无MACE attempt，未把缺字段直接改成0。显式hash不重复也不代表材料一定具有官方科学新颖性；index-only选择无法由此判定是否同一结构。

- **Mg-Sn-Sr**：UQ score_buffer4次/MACE89，10次都稳定且新，SUN10/AUDC1；full没有MACE，前4次稳定且新，后6次仍稳定但不新，SUN4/.64。两臂均没有神经retry或风险排序改选，因此这里不能把差异称为NN识别了/没识别推理错误。full图为19/20有效提案有支持，单纯“全部缺图”也解释不了后6次停滞。
- **Au-K-Tb**：UQ MACE52、SUN3/.25；full MACE0且10次全部不稳定，SUN0。full有2次not_new神经重提，但没有风险排序改选；需保留“稳定性失败”与“新颖性风险反馈”的区别。
- **Co-Mg-Na正对照**：UQ MACE32仍SUN0，full MACE0却SUN2/.08。更多筛选不保证更好。
- **Ga-Ho-Lu反向筛选对照**：UQ MACE0、SUN6/.72；full MACE20、SUN3/.45。它进一步反驳“full只因不调用MACE而变差”的单因解释。

因此，可验证的机制是**ES改变了实际工具/候选分布，并与相同固定controller相互作用**；测试数据尚不能估计MACE的独立因果贡献。以后若比较筛选策略，必须另行在train/dev预注册，不可用这17个结果决定工具频率。

## 新full训练：学到了参数更新，未观察到神经信号驱动训练决策

下表的train G1是theta0加第一代扰动；train G2是theta1加第二代扰动，并非无扰动的同代dev。

| 原注册请求 | AUDC | 有图/有效生成 | 实际NN重提/排序改选 | schema额外提案 | MACE |
|---|---:|---:|---:|---:|---:|
| G1 train pop0 | .71 | 1/21 | 0/0 | 14 | 0 |
| G1 train pop1 | .19 | 3/18 | 0/0 | 2 | 0 |
| G1 dev | .62 | 17/17 | 0/0 | 1 | 0 |
| G2 train pop0 | .48 | 13/13 | 0/0 | 0 | 0 |
| G2 train pop1 | .26 | 20/20 | 0/0 | 0 | 104 |
| G2 dev | .64 | 15/15 | 0/0 | 0 | 0 |

第一代train另外有20/1个生成失败。其有效提案的缺图均记录为FVU门失败：20/15个；未降低门槛，走unknown中性契约。高奖励.71的第一代pop0在事件级最后执行select切片上0/10有支持，pop1为2/10。这说明ES奖励没有要求轨迹实际利用神经信号；不能反过来声称“缺图导致高奖励”，因为扰动同时改变了许多行为。

两个更新均对723个参数块记录非零delta，组合L2约23.82；sigma为.001/.0002，alpha=.0005，population奖励为[.71,.19]和[.48,.26]。这是实际full-parameter ES，不是空更新。该数值没有按权重范数归一，不能仅凭绝对L2断言过大。`uncertainty_reward_weight=0`，正式fitness为官方AUDC；NN只可能经controller间接影响轨迹。本次六条轨迹中的神经retry请求、scalar阈值越界、风险排序改选均为0。

## 事件级风险诊断，避免重复标签虚增样本

多个执行工具共享下一次科学结果。主要校准数字使用root独立去重的60个科学事件：每step优先最后executed select，没有该行时显式采用最后执行工具；无图仍缺失。见[事件CSV](train_dev_event_deduplicated.csv)与[事件汇总](train_dev_event_deduplicated_summary.json)。原analysis.json的逐行AUROC仅作诊断，不能替代事件级结果。

| dev，均为10个事件/全支持 | AUROC | Brier | 常数.5 Brier | >=.6 |
|---|---:|---:|---:|---:|
| G1 | .7500 | .250079 | .25 | 0 |
| G2 | .7083 | .249727 | .25 | 0 |

这些轨迹中存在少量排序信号；不能说NN完全没有信息。但概率仍约.499–.501，既未达到控制阈值，Brier也与常数.5几乎相同。10个事件、一个开发体系不能证明可靠校准或泛化。训练40事件总体失败比例.65，可作已声明的常数对照，不能据此直接重调NN。尤其不得把阈值降到.500x来人为制造干预，再把“触发了”称为“改善了”。

## 三个可验证机制及下一步边界

1. **无G0拒绝更新选项（确定的注册/实现事实）**。`es_training.py:326–340`计划从generation1开始，更新之后才dev；`full_es_training.py:373–378`强制只在G1/G2取最大。实际dev仅两个同seed请求，.62/.64，故选择G2。G0 checkpoint存在，但没有同controller的G0 dev轨迹。该规则可能选择两个都劣于G0的更新；当前还不知道G0是多少，不能把可能当结论。
2. **策略/工具分布与NN使用脱节（已有轨迹证据，因果分解未完成）**。训练早期高奖励可来自几乎无图支持的轨迹；当前测试full的筛选量及有效图覆盖下降，而神经重提很少。它解释了为何“实现了NN+ES”不等于“这轮ES受NN有效指导”，但不能单独分解参数、生成内容、图支持和工具策略的效应。
3. **有弱排序信息但校准/控制未证有效（事件级证据）**。G1/G2 dev有部分风险排序信号，却没有阈值干预，且Brier接近常数。需要预注册train/dev干预实验才能判断能否用于规划；未拟合新NN、未改温度/阈值，也未用17个测试例挑修法。

最小下一步已形成独立G0开发诊断预注册设计：保持原Al-Pd-Sm、driver/policy seed4289295197、实际环境seed4272278902、B10、同support/NN/evaluator/gates，仅新增theta0一条10candidate ORB，引用原G1/G2物理记录不重跑。初始化单列成本，旧selection不改。它仅回答该已用开发seed上的缺失对照；独立三新seed的theta0/G2配对另需6条/60candidate calls。当前只有预注册设计，没有资源或执行adapter准入，没有科学调用。


---

Publication note: this report preserves the 2026-09-19 17:41:19 CST observation. The G0 paragraph describes a proposal at that time, not a completed comparison. The original extraction remains private; [trajectory_evidence.json](trajectory_evidence.json) is a scientific projection with original source hashes. [recompute.py](recompute.py) independently checks all 74 trajectory summaries and the 60 deduplicated train/dev events. This does not load weights or rerun a physical evaluator. Repeated labels from multiple tools are not independent samples.
