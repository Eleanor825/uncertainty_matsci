# 保留 G0：已实现的元数据选择器与未来接入边界

[selector.py](code/selector.py) 为原字节源，tinyGit `61a5877cace4585f52ff0a5263e136d70a517141` 在元数据执行前封存。公开 projection SHA `09fd7ed68d68407c54212be0446c1595b03308759377cab5b169852e7c2a731f` 与原执行输入相同；[selection.json](selection.json) 也是原输出字节。两套完整排名均保留：旧规则 G2(.64)>G1(.62)，新规则 G0(.68)>G2(.64)>G1(.62)。

这是 AlPdSm/B10 的一个已用开发 seed。此次没有新采样、ORB、模型拟合、测试集执行或生产权重替换。不能由该开发选择宣称泛化提高、物理因果关系或超过 Native。汇总审计引用已公开测试 aggregate；选择函数只接收开发输入，不读取测试路径或据测试选模。

纯函数 `select_incumbent(rows, expected_seeds=...)` 要求完整 G0/G1/G2×固定开发 seed 网格，按等 seed 平均官方 AUDC 排序，精确平局选较早 generation，包含 G0。拒绝重复身份、test split、非 AUDC、非有限分数；缺失/失败保留，返回 `selection_incomplete`，不会当零分。此时保留 G0 只是部署默认，不是一次测得胜出。

未来新的 ES runner 应在训练前注册 G0 的真实开发评估和成本，所有候选使用相同 controller/NN/bank/parser/资格门、预算和 seeds；核验原 scientific completion 与精确 state hash 后调用函数。原数值门、.6 阈值与 baseline 都不改。这个纯选择器不代替科学资格、GPU 生命周期或 checkpoint 身份检查。原 G2 选择与 1079 个有效结果不会被回写。

运行 `python3 -B results/development/made_underperformance_20260923T012020_CST/validate.py` 可用本仓库历史 dev projection 复现两种选择。原 CLI 的 `run_audit` 还要求原 tinyGit/project 布局；公开 validator 直接调用同一原字节函数并核 SHA，不假造原目录。原本地10项测试已通过，公开校验另外检查真实 producer 字段、两套排名、平局、未知、重复及 test 输入拒绝。
