# SnAr 风险失败诊断：标签一致，概率对执行条件的迁移失败

截至这次本地审计，**没有找到标签反转、Brier 公式、未实际拟合、错载 NN 或温度方向的实现错误**。已证实的是：同一原 NN 在冷启动开发集和共同 prior 下的候选分布之间，概率发生大幅变化；它在原开发集也没有超过简单训练先验常数的概率质量。不能据此宣称归一化代码“有 bug”，也不能把 AUROC 大于 0.5 当作可用的概率校准。

本报告使用已接受的 75 train / 50 dev 特征、原 12+3 个小 NN 拟合诊断、原 224 个有风险分数的已执行 test 动作，以及刚返回的四支实际 ES history。全部计算仅用已有数值；0 新模型/图/oracle/拟合。test 只用于描述已发生的失败，不用于选择阈值、特征、epoch 或修复赢家。

## 1. 定义与实际拟合没有发生方向错误

原 `pipeline.py:222–223` 的标签为 `int(HV(prior + past + current) − HV(prior + past) <= 1e-12)`。1 是**没有增加 HV**，不等于 ODE 失败、参数非法或推理错误。repaired pipeline 从实际 query 继承这个标签，BCE 对 1 拟合正 logit；推理为 `sigmoid(logit / positive_temperature)`。episode Brier 与 NN 指标均为 `(p − no_hvi)^2`，没有二分类乘二或反向概率。

原 all-feature/seed1729 真实执行了 **59 个 AdamW 步**，选 epoch44，development raw BCE 0.3429369330；温度 **1.1583745448** 以同一 50 dev 的 NLL 拟合。温度后 dev NLL 改善到 0.3392514346，但 Brier 从 0.09622942 略升到 0.09714218——NLL 校准不保证 Brier 同时改善，这是目标区别，不是计算错误。相同 dev 同时用于 epoch 选择、校准、报告，不能把它称独立验证。

之前真实 reproduction gate 完全重现原模型 tensors、预处理、history、epoch、温度，224 个原分数最大差 **0.0**。它有力排除了该读取路径上的模型加载/特征顺序/保存恢复错误，但不证明这种训练设计具备泛化能力。保存的 15 个模型仍保留全部负结果，没有替换 controller。

| 集合 | no-HVI | 原 NN 平均风险 | ≥0.5 | Brier | NLL |
|---|---:|---:|---:|---:|---:|
| 原 train | 70/75 | 本次未重推理 | — | — | — |
| 原 dev | 45/50 | 0.867484 | 50/50 | 0.097142 | 0.339251 |
| 原已选 test 动作 | 218/224 | 0.175444 | 10/224 | 0.677841 | 1.937815 |
| 固定 train prior=70/75，评 dev | 同上 | 0.933333 | 50/50 | **0.091111** | **0.332899** |
| 同一 train-prior 常数，评 test | 同上 | 0.933333 | 224/224 | 0.027659 | 0.139682 |

原 dev AUROC 0.608889 只由 5 个 HVI 改进支持。即使不看任何 test 坏案例，开发集已经不能证明该 NN 的概率优于上述零拟合常数。test 的 97.32% failure 并不是完全新的高基率：train 本来已有 93.33%。纯类别先验变化不能单独说明为什么预测均值反而掉到 17.54%。

## 2. 三个可检验机制，及目前证据边界

**A. archive 条件与目标、输入同时变化。** 原 collection 五条 B30、两支 ES 的 B20 population/dev 都从空外部 archive 开始；最终 B50 从同一 **550 条 prior、HV=0.8810117795** 开始。标签代码相同，但被比较的 frontier 不同。同一物理点可能改善一个冷启动 front，却不能改善已成熟的 pooled front。LLM 只收到最多 4 个 Pareto prior 摘要和最近 6 条本轨迹结果；285 个 NN 特征是 sampling/hidden/graph，没有显式全 archive/当前 HV/完整重复历史特征。不能把这种部分观察条件称成“必然不可预测”，但训练未覆盖实际 warm-start 使用条件是明确的协议差距。

**B. whole-prefix 特征支持范围变化，温度无法修复 logit 符号。** `policy.py:925–929` 的 hidden mean/RMS 对整个原 prefix 的 token×hidden 轴取平均；它们不是只取新动作 token 的表示。prior 摘要、history 和预算变动会同时改变该输入。按原 train75 的 mean/std 直接复算，以下均无新学习：

| hidden128 支持统计 | dev50 | 原 selected224 |
|---|---:|---:|
| 超出 train min/max 的单元 | 153/6400 | 4360/28672 |
| `abs(z)>3` 单元 | 11 | 1866 |
| `abs(z)>5` 单元 | 0 | 350 |
| 最大 `abs(z)` | 3.675 | 9.165 |

所有这些 hidden 列训练 std 均超过原 1e-8 下限；没有 hidden 分母被置 1、NaN 或推理临时重拟合。graph 中 103 个近常数列按原代码将 std 置 1，也并非除零。全部 selected224 的285输入有限；失败生成/缺图另计，不填造特征。较大的 z 是外推证据，不是数值损坏证据。正温度不改变 logit 正负，因此不能让这些负 logit 越过原 0.5。后验 family 诊断中 hidden/all/union 均有较差概率，sampling/graph 接近高失败常数；这不许可从 test 选择删 hidden/graph。

**C. 控制器选样和奖励尺度增加了诊断歧义。** 已有 test 首候选 219 个有效分数均值 **0.190787**；239 个有分数候选均值0.210174；最终224个均值0.175444；15个未选候选均值0.728811。选最小风险进一步降低了均值，但低风险在首候选上已经出现；不能把全部下降归于最终 rerank。与此同时，没有未执行候选的反事实 oracle 标签，不能推出取消重试会提高 HV。该集合上 all NN 的 AUROC0.791284 仍只是所选动作排序，只有6个 HVI 改进。

原 full 的真实开发奖励进一步显示概率与发现质量并不等价：

| checkpoint | mean querywise HV | Brier | `HV − .1 Brier − .1 invalid` |
|---|---:|---:|---:|
| G0 | 0.6640389052 | 0.09017244 | 0.6550216616 |
| G1 | 0.6640401817 | 0.16697205 | 0.6473429770 |
| G2 | 0.6060286547 | 0.00904242 | 0.6051244131 |

G1 的 HV 只增加0.0000012765，fitness减少0.0076786846几乎全来自 Brier项；G2 的 Brier很好但 HV明显较低。原公式确实如此，不是执行错目标。不能用“风险更准”代替发现效果，也不能从这个观测认定模型有意利用奖励漏洞。

四支均 P=2；不平奖励的 z-score几乎总为±1。在 N=4,539,265,536、alpha=.0005 下，独立各向同性噪声近似的更新范数为 `alpha*sqrt(N/2)=23.82033`，与实际约23.82一致。sigma=.001/.0002 的候选扰动典型范数分别67.374/13.475，更新/扰动比约0.354/1.768。**第二代步幅大于探索半径是可检验的 overshoot/高方差假说，不是已证算法 bug 或失败原因。** 原 SnAr ES-only 第一代同奖励的更新为0，两支最终都选G0，因此这里的 heldout 特征变化不能归因于已选策略的 ES 权重漂移。

## 3. 最轻量的下一步，先证实预测价值再做新 ES

1. **零调用诊断优先。** 已提供 `observe_archive_labels.py`。它重新核150个原 collection query，再对每条 train 使用另外两条 train 的60点 prior、对 dev 使用90个 train 点 prior，复算固定物理结果的 no-HVI 翻转。它不使用test；不能用包含拟合样本自身的550点 prior重标训练（那会泄漏）。这些旧 prompt 没看到新 prior，所以该复算仅检验目标条件变化，**不能直接冒充新的 warm-start 训练数据**。
2. **唯一建议进入新注册的数据/NN候选：匹配 warm-start 条件，保留原数学先做可证伪实验。** 若零调用检查支持该缺口，用90条既有 train-only 实测结果冻结共同 prior，新采3条 train B30+2条 dev B30（**150 calls，约75/50个非LHS特征**），G0、原解码/FD/FVU/.5阈值/最多两候选保持；采集不使用NN选样。新 seeds 必须从未用清单确定后注册，原test seeds及结果不得用于设计。3个train episode留一组 OOF选择原100epoch内训练时长，OOF logits拟合温度，之后1次全train重拟合；最多4个小NN fits/400优化步。新dev不再参加选择/校准。保留class-prior、step0和schema/no-NN对照；若某fold/dev缺一个类别则报告不可验收，不自动增加budget或挑成功样本。只有在新dev上NLL/Brier超过train-prior常数、且有可复核的排序证据，才进入controller试验；不以更多threshold crossings作为成功标准。
3. **完整行为验证另计，不能用NN离线分数收尾。** 新dev预先固定3 seeds，旧NN / 新NN / 同schema无NN三臂各B20，9条=**180 calls**，同G0/prior/RPC口径并报告HV gain、重复、LLM/图成本、Brier。合计候选最小330新calls，当前未注册/未执行。若预测和行为均不过，保留否定结论，不默认进入ES。ES后续须包含G0 dev参照，并继续采用匹配的warm-start train/dev；不要复用旧冷启动G2选择当新方案。

alpha随sigma缩放可作为之后**独立**的训练/dev对照：保留第一代alpha=.0005，第二代预设alpha=.0001（不改sigma），对照原fixed-alpha；不得与NN数据变更同时混为单一归因。P增加是另一成本更高因素，先不加；SnAr只有一个反应函数，不能把多seed称多化学泛化。此处只列下一注册候选，不改原history、checkpoint或已运行队列。

## 证据与限制

`analysis.json` 绑定全部真实输入和源码SHA；`model_calibration.csv`、`normalization_support.csv`、`feature_support.csv`、`full_development_reward.csv`、`ES_update_scale.csv`保留复算值。`analyze.py`仅标准库，无 torch/LLM/图/oracle。固定train-prior的开发集对照在`constant_train_prior_controls.json`。

新seed5201的 Brier≈.7316/HV gain0目前来自root实时观察；本报告未将它冒充已独立读取的原5seed证据。只读collector同时提取其完整summary/query（仅当summary已存在），确认后可追加实际回执。当前概率崩溃的**单一因果原因仍未隔离**；这里证实的是标签同方向、训练真实发生、probability严重外推、冷/暖archive协议不匹配、及ES奖励/步幅的具体可检验缺口。
