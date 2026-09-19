# 冻结 NN 的原 train/dev 信息诊断

**没有发现稳健的总体失败/稳定性预测证据；`not_new` 与工具失败有有限的开发集排序迹象。格式失败的近完美识别不能作为 NN 或图的额外价值。** 本次没有重新拟合、选阈值、运行 LLM/oracle 或使用 GPU，也没有读取 30 个测试体系做选择。只加载原冻结小型 NN，在既有特征上做 CPU 推理。

数据是原 G0 collection：**Al–Au–Hf，seed 1/2/3，train 150 次 ORB**；**Al–Pd–Sm，seed 1，dev 50 次 ORB**。共 386 个 proposal（302 train、84 dev）。这与另一个同 G2 的 schema 修复 seed 2 配对报告不同，不混合两个 policy state。原 dev 已用于 NN 选 epoch 和 temperature 校准，因此下面是**已参与模型选择/校准的开发表现**，不是未见验证集泛化，更不是多个独立 dev seeds。

原 4 份 label sidecar 由冻结 verifier 从完整原决策、tool/step RPC 重新核验，386 个 feature/label 身份与原训练 provenance 相等；200 个科学事件都能定位最后一个实际执行的 select。两份冻结 checkpoint 与已登记 SHA 一致，原 15 项保存的开发指标重算差异均小于 `1e-6`。最终数据提取与小型 NN 推理耗时 15.2 秒，CUDA 被显式禁用。证据见 [evidence.json](evidence.json)、[saved_metric_reproduction.json](saved_metric_reproduction.json)；原文源码与 SHA 见 [source_semantics.json](source_semantics.json)。

**标签、mask 与支持域核查：**

| 项目 | train | dev |
|---|---:|---:|
| 实际执行 proposal / 未执行 proposal | 293 / 9 | 83 / 1 |
| scalar future-label 拟合行 | 293 | 83 |
| 真实科学事件 | 150 | 50 |
| 同一个科学事件最多赋给多少此前 proposal | 8 | 4 |
| 图可用 / 缺图 proposal | 294 / 8 | 83 / 1 |
| **有效 generation 且缺图** | **0** | **0** |
| 缺图且 generation-invalid | 8 | 1 |
| 未执行但具有 future/工具/科学结果标签 | 0 | 0 |
| train/dev 完整 prefix 重叠 | — | 0 |

future-risk 在这些成功返回的科学事件上等于 `unstable OR not_new`；它不是“每个此前工具导致失败”的标签。原监督将一次未来 ORB 的结果赋给多个此前已执行工具 proposal，293/83 行不能当成 293/83 次独立科学事件。原训练对 episode 加权，但没有消除同 episode 内一个事件的多重 proposal 标签。本审计同时保存原 proposal 视角、每事件预测均值视角及每事件最后 select 视角，未改任何训练数据或既有 fit。

未执行的 10 个 proposal 仅保留其真实 parsing 标签，后验标签均为 `None`，没有当成阴性。完整原标签/来源对齐未发现未来结果写入 feature 的证据：sampling、hidden、action 来自已生成但尚未执行的 action prefix；离线 graph 对齐相同 prefix/G0；科学标签随后从 RPC 写入。**不过 `generation_invalid` 已精确等于 `1 − sampling.valid_json_action`，386/386 行成立。** 这是当时已经可观察的接口状态，不是未来科学结果；以现成 parsing bit 直接计算即可得到相同的完美标签，不需要 NN 或 attribution graph。

scalar 实际只用有图的 293/83 行；typed 虽保留缺图行，但缺图与 generation-invalid 完全重合。于是 **“有效 action + 缺图”在原 train/dev 完全没有监督支持**。这可以解释为什么应谨慎对待后续 G2 缺图时的高风险预测，但不能仅凭相关性证明模型具体依赖了哪个特征。原输入共 298 列：153 graph、130 hidden、9 action、6 sampling；121 列在 train 中为常数。没有做特征消融或新拟合，不能据现有结果识别 graph 的独立增量。

**固定 NN 与 train-only 常数的 dev 对照：** 常数概率先在相同标签、相同聚合视角、相同图支持域内求每个 train episode 的发生率，再对 3 个 episode 等权平均。Brier/logloss 同样先 episode 内平均再 episode 等权；当前 dev 只有 1 个 episode。AUC 为描述性的 dev 排序，不附独立样本显著性结论。

| 预测目标与观察单位 | dev n / 正例 | train 常数 p | NN Brier / 常数 | NN logloss / 常数 | NN AUC |
|---|---:|---:|---:|---:|---:|
| future failure；每 ORB 最后 select | 50 / 25 | .713333 | **.249988 / .295511** | .693124 / .793621 | **.4928** |
| unstable；每 ORB 最后 select | 50 / 16 | .486667 | **.253871 / .245378** | .701056 / .683901 | **.5110** |
| not_new；每 ORB 最后 select | 50 / 14 | .253333 | **.192842 / .202311** | .565645 / .594792 | **.5853** |
| tool failure；实际工具 proposal | 83 / 4 | .099591 | **.051679 / .048512** | .198196 / .211016 | **.6962** |
| generation-invalid；每 proposal | 84 / 1 | .031217 | ≈0 / .012136 | ≈0 / .072608 | **1.0000** |

future NN 胜过 train 发生率常数，**不能解读为有效区分失败**：它在这 50 个事件的概率仅 .499650–.500915，AUC .4928；其 Brier .249988 几乎等于固定无信息 `.5` 参照的 .25。这主要体现原 train 失败率 .7133 与 dev .5 的发生率差异，以及在该 dev 上校准后接近 .5 的输出。scalar 选中 epoch 1、temperature 54.5978（原搜索上界接近 `exp(4)`），亦与输出被压平一致；未据此改参数。

unstable 的 Brier、logloss 均不及 train 常数，AUC 近 .5。not_new 的分数与排序有弱正向迹象；tool failure 的 AUC 高一些，但只有 4 个正例，Brier 反而较差。它们都来自一个已参与选代和校准的 dev episode，尚不足以断言可靠信息或控制收益。generation-invalid 的 AUC 1 只验证识别了已编码的 parsing 状态。

原 proposal 视角的 future/unstable/not_new AUC 分别 **.5163/.5175/.5768**；去重后分别 **.4928/.5110/.5853**。主表没有把多条同事件标签当作额外样本。其他视角与图可用/缺图分层的全部数值见 [metrics.csv](metrics.csv)、[label_support.csv](label_support.csv)。

**固定 .6 的离线触发检查，不是声称原 baseline collection 实际执行了这些 retry：**

| head / 主表观察单位 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| future failure | 0 | 0 | 25 | 25 |
| unstable | 0 | **1** | 16 | 33 |
| not_new | 0 | 0 | 14 | 36 |
| tool failure | 0 | 0 | 4 | 79 |
| generation-invalid | 1 | 0 | 0 | 83 |

三个原 typed head 正确保持不可用，概率记 NA：candidate-generation failure 的 train/dev 为 **0/86、0/23 正例**；scientific-evaluation failure 为 **0/293、0/83**；screening unavailable 为 **2/28、0/6**。不将这些 head 的缺失预测记成 0，也不宣称其已经学会异常识别。

本阶段已完成标签时域、未执行 mask、图支持域、事件重复、episode 聚合与冻结预测/常数对照。**没有合法的 OOF/CV 结果可报告**：当前唯一冻结模型已用全部 3 个 train episodes 拟合；把它再评估其中一条不能叫留一验证。本任务禁止新拟合，因此仅保留每 episode 描述。当前结论支持继续审慎验证 unknown 处理，并要求新增控制收益证据；不支持“现 NN 已可靠预测科学失败”或“提高模型容量即可改善”的结论。下一次真 OOF 必须另行冻结按完整 episode 分组的新拟合协议，不能用本次 in-sample 结果冒充。

复算：在本目录运行 `python3 analyze.py`。该脚本只读缓存并生成统计；无远端调用、模型拟合、LLM/oracle 或 GPU。逐 episode 计数见 [episodes.csv](episodes.csv)，逐 episode 分数及 in-sample/已校准标记见 [per_episode_metrics.csv](per_episode_metrics.csv)，逐事件含原 RPC 行号见 [scientific_events.csv](scientific_events.csv)，所有 proposal 预测和 mask 见 [proposal_predictions.csv](proposal_predictions.csv)。没有改写原数据、冻结 checkpoint 或当前运行实验。
