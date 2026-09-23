# MADE B10/B20 在线池化对照：4/20 完成，2 条技术失败

固定快照：2026-09-24T02:40:19.773670+08:00。20 条 = 2 个既有化学体系 × B10/B20 × 5 组；策略与环境 seed 均为 501，G0 固定。当前 4 条已接受、2 条技术失败、14 条未领取。

**已完成结果均为 Native1 与 Common2 对照。Common2 不使用内部风险；其差异不能归因于不确定性或内部表征。**

| 体系 | 预算 | 组别 | 状态 | SUN | mSUN | AUDC |
|---|---:|---|---|---:|---:|---:|
| Mg-Sn-Sr | 10 | Native1 | 已接受 | 8 | 0.800 | 0.860 |
| Mg-Sn-Sr | 10 | Common2 | 已接受 | 9 | 0.900 | 0.930 |
| Mg-Sn-Sr | 10 | Public2 | 技术失败 | — | — | — |
| Mg-Sn-Sr | 10 | FullPool2 | 技术失败 | — | — | — |
| Mg-Sn-Sr | 10 | ActionPool2 | 未领取 | — | — | — |
| Au-K-Tb | 10 | Native1 | 已接受 | 0 | 0.000 | 0.000 |
| Au-K-Tb | 10 | Common2 | 已接受 | 3 | 0.300 | 0.390 |
| Au-K-Tb | 10 | Public2 | 未领取 | — | — | — |
| Au-K-Tb | 10 | FullPool2 | 未领取 | — | — | — |
| Au-K-Tb | 10 | ActionPool2 | 未领取 | — | — | — |
| Mg-Sn-Sr | 20 | Native1 | 未领取 | — | — | — |
| Mg-Sn-Sr | 20 | Common2 | 未领取 | — | — | — |
| Mg-Sn-Sr | 20 | Public2 | 未领取 | — | — | — |
| Mg-Sn-Sr | 20 | FullPool2 | 未领取 | — | — | — |
| Mg-Sn-Sr | 20 | ActionPool2 | 未领取 | — | — | — |
| Au-K-Tb | 20 | Native1 | 未领取 | — | — | — |
| Au-K-Tb | 20 | Common2 | 未领取 | — | — | — |
| Au-K-Tb | 20 | Public2 | 未领取 | — | — | — |
| Au-K-Tb | 20 | FullPool2 | 未领取 | — | — | — |
| Au-K-Tb | 20 | ActionPool2 | 未领取 | — | — | — |

SUN 为官方 stable–unique–novel 计数，mSUN = SUN / 预算；AUDC 使用原官方归一化发现曲线面积。破折号表示无已接受科学结果，不是 0。

| B10 完成配对，Common2 − Native1 | ΔSUN | ΔmSUN | ΔAUDC |
|---|---:|---:|---:|
| Mg-Sn-Sr，seed 501 | +1 | +0.100 | +0.070 |
| Au-K-Tb，seed 501 | +3 | +0.300 | +0.390 |

Mg-Sn-Sr 为 SUN 8→9、AUDC 0.86→0.93；Au-K-Tb 为 SUN 0→3、AUDC 0→0.39。只有两组既有体系各一个种子的 B10 配对，属于描述性开发结果，不作显著性或未见体系泛化主张。

Native1 每轮生成一个候选。Common2 使用两个候选、固定公共反馈和公共 hash 选择，不调用风险模型。Public2、FullPool2、ActionPool2 分别是冻结的公共特征、公共＋全序列 MLP、公共＋动作值预测位置 MLP logistic 风险；本快照没有这三组的已接受结果。

Mg-Sn-Sr B10 的 Public2 与 FullPool2 均报 `Wrong action-position boundary`。源码和纯 CPU dataclass 检查表明：`to_dict()` 保留 tuple，而检查将其与 list 比较，导致类型边界错误。这是技术失败，不能记成 SUN=0、不能据此判断方法优劣。原失败记录保留；失败或未执行候选没有补反事实标签。

完成轨迹共实际 40 次候选 ORB 评估。初始化调用、生成、捕获和时间成本另见 [snapshot.json](snapshot.json)，其成本只覆盖已接受轨迹，不代表失败任务未消耗资源。四条已接受轨迹的内部风险选用、捕获 forward、图调用、ES 更新和拟合次数均为 0。

[provenance.json](provenance.json) 保存固定观察文件、科学注册、原方法、完成与失败记录的 SHA 依据及类型错误诊断。六个终态回执均按原 canonical JSON 重建并匹配 SHA；公开包不含原始提示、token、结构、RPC 日志或机器信息。`python3 -B validate.py` 可重算状态、配对差值与 AUDC。

这是新增固定快照；未混入之后 core 扩展结果，历史报告和论文均未修改。
