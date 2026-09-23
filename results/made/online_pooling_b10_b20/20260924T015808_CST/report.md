# MADE B10/B20 在线池化对照：2/20 已完成

固定快照时间：2026-09-24T01:58:08.613330+08:00。

**目前唯一完成配对是双候选公共控制 Common2 对单候选 Native1。它不是内部特征、不确定性模型或 ES 的收益证据。**

本批固定 2 个曾被考察的化学体系 × B10/B20 × 5 组，共 20 条轨迹、300 次候选 ORB 预算；策略和环境种子均为 501，全部使用相同未更新的 Qwen3.5-4B G0。当前已完成 2 条、已启动未完成 2 条、待运行 16 条；未完成指标用“—”，不填 0。此为开发性描述结果，没有显著性或新 holdout 泛化主张。

| 化学体系 | 预算 | 组别 | 状态 | SUN | mSUN | AUDC | 整条任务秒数 |
|---|---:|---|---|---:|---:|---:|---:|
| Mg-Sn-Sr | 10 | Native1 | 已完成 | 8 | 0.800 | 0.860 | 657.727 |
| Mg-Sn-Sr | 10 | Common2 | 已完成 | 9 | 0.900 | 0.930 | 761.670 |
| Mg-Sn-Sr | 10 | Public2 | 待运行 | — | — | — | — |
| Mg-Sn-Sr | 10 | FullPool2 | 待运行 | — | — | — | — |
| Mg-Sn-Sr | 10 | ActionPool2 | 待运行 | — | — | — | — |
| Au-K-Tb | 10 | Native1 | 已启动，未完成 | — | — | — | — |
| Au-K-Tb | 10 | Common2 | 已启动，未完成 | — | — | — | — |
| Au-K-Tb | 10 | Public2 | 待运行 | — | — | — | — |
| Au-K-Tb | 10 | FullPool2 | 待运行 | — | — | — | — |
| Au-K-Tb | 10 | ActionPool2 | 待运行 | — | — | — | — |
| Mg-Sn-Sr | 20 | Native1 | 待运行 | — | — | — | — |
| Mg-Sn-Sr | 20 | Common2 | 待运行 | — | — | — | — |
| Mg-Sn-Sr | 20 | Public2 | 待运行 | — | — | — | — |
| Mg-Sn-Sr | 20 | FullPool2 | 待运行 | — | — | — | — |
| Mg-Sn-Sr | 20 | ActionPool2 | 待运行 | — | — | — | — |
| Au-K-Tb | 20 | Native1 | 待运行 | — | — | — | — |
| Au-K-Tb | 20 | Common2 | 待运行 | — | — | — | — |
| Au-K-Tb | 20 | Public2 | 待运行 | — | — | — | — |
| Au-K-Tb | 20 | FullPool2 | 待运行 | — | — | — | — |
| Au-K-Tb | 20 | ActionPool2 | 待运行 | — | — | — | — |

唯一完成配对 Common2−Native1：SUN +1、mSUN +0.10、AUDC +0.07，整条任务耗时增加 103.943 秒。仅一对轨迹；不能据此推广至其余体系、预算或模型。

完整成本保存在 [snapshot.json](snapshot.json)；下表区分完成轨迹的物理初始化、候选评估、生成及推理成本。时间项是不同边界，RPC 时间包含初始化，不应重复相加。

| 已完成轨迹的成本 | Native1 | Common2 |
|---|---:|---:|
| 冷启动秒数 | 110.952 | 105.377 |
| 整条任务秒数 | 657.727 | 761.670 |
| 初始化 ORB 次数 | 96 | 96 |
| 初始化秒数 | 184.048 | 176.597 |
| 候选 ORB 次数 | 10 | 10 |
| MACE surrogate 次数 | 4 | 1 |
| 候选循环秒数 | 334.679 | 452.863 |
| 全部 RPC 秒数 | 225.881 | 221.352 |
| 生成返回数 | 21 | 28 |
| prompt token 数 | 51709 | 52973 |
| completion token 数 | 2911 | 4180 |
| 工具尝试数 | 21 | 14 |
| 生成意图数 | 22 | 33 |
| context 预检拒绝数 | 1 | 5 |
| hidden capture forward 数 | 0 | 0 |
| logistic selector 调用数 | 0 | 0 |
| 内部风险实际选用数 | 0 | 0 |
| graph 调用数 | 0 | 0 |
| ES 更新数 | 0 | 0 |

Native1 每次生成 1 个候选；Common2 每次生成 2 个，使用固定公共反馈与公共 hash 选择。Public2、FullPool2、ActionPool2 分别使用已冻结的 P、P+B、P+C logistic 诊断探针，本快照尚无这三组结果。所有组均无 ES、图调用或在线拟合；未执行候选不补反事实标签。

物理评价仍为固定 CPU ORB-v3-conservative-inf-omat，单 ORB worker、MACE 4 workers、Chemeleon generator 在 CUDA；FIRE、fmax=0.02、最多500步、晶胞松弛及0.1 eV/atom稳定阈值保持不变。初始化 ORB 调用单列，不冒充 B10/B20 候选预算。

两次观测的完成回执一致，且从回执值按原序列化方式重建字节后 SHA256 与原生文件引用一致。报告依赖原运行器已接受的 RPC/物理证据重建；公开包不含原始 token、候选结构、模型权重、机器身份或私有路径。科学注册、模型/evaluator/方法源 hash 与全部成本见 [snapshot.json](snapshot.json)。`python3 -B validate.py` 可独立校验快照算术、状态与 [manifest.json](manifest.json)。

此前离线诊断和历史正负结果均保留；本次仅新增独立时间快照，不修改论文主表。
