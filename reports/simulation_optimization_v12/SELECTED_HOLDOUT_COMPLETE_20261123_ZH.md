# 3200回合新布局：所选 Reach 和 Payload 模型在困难几何复现优势

[查看 Piper 仿真视频：Reach、Payload 和失败反例](piper_videos_20261123/README.md)。

2026-11-23（UTC+8），Reach 于11:10、Payload 于11:11结束，两任务退出码均为0。按[预先提交的冻结协议](../../configs/paper_crax_payload/selected_seed_frozen_holdout_20261123.json)完成四个固定模型、八个任务条件、每模型每条件200回合，总计3200回合；没有训练、调参、选中间检查点或按结果换布局。所有模型都用原目标 actor 和无噪声部署。模型和训练 seed 选自历史开发证据，本批新布局未参与选择，所以这里只验证所选模型，不证明全 seed 稳定或随机部署也有相同表现。

|任务／几何|SAFEFlow 安全成功|同 seed Metric-only|成功率差|违规：SAFEFlow／Metric|
|---|---:|---:|---:|---:|
|Reach 41029，57.5mm|200/200（100%）|196/200（98%）|+2.0pp|0／4|
|Reach 41029，60mm|200/200（100%）|191/200（95.5%）|+4.5pp|0／9|
|Payload 41027，55mm|191/200（95.5%）|182/200（91%）|+4.5pp|9／18|
|Payload 41027，57.5mm|180/200（90%）|156/200（78%）|+12.0pp|20／44|

两任务的 nominal 和 execution scale 0.85，两方法每格均为200/200。全部3200回合没有超时或停滞，原安全距离、horizon 和首次违规终止规则均不变。Reach 所选 P5+2mm 达到两个几何阈值；四条件合计800/800，Metric-only 为787/800。在两种困难几何下分别比 Metric 多4和9个安全成功布局，没有只被 Metric 成功的布局。

Payload 所选 P5+4mm 在55mm新增12个只有候选安全成功的布局、失去3个只有 Metric 安全成功的布局，57.5mm对应27和3；两处合计少33次违规。57.5mm平均截尾完成用时1.622秒对1.924秒，仍有20次违规，不能写成无风险。55mm是已有训练覆盖几何；57.5mm是超出该训练覆盖的几何，两者的评测布局都来自新 seed 区间。旧独立80%／63%与这一批不是同布局对照，不能把两批差值写成直接因果改善。

在名义条件和执行缩放条件，成功率打平，SAFEFlow 截尾完成用时稍长；不能声称所有指标都严格领先。推理计时保留在CSV，测的是共享 GPU 下各回合动作调用分位数的均值，不是实机。

## 未改变的负结果和取舍

- 所选 Reach 训练后缀违规50对268，但成功4870对5033，候选多1次训练超时、1个训练停滞回合。所选 Payload 的对应指标为违规108对197、成功2730对2832、训练超时均0、停滞2对0。新布局无超时不抹去训练代价。
- 另两个 Payload 父42028、42029完成各自300K谱系仍零训练成功；固定P5在第三个 Reach seed 终点崩塌。全部保留在[三 seed 汇总和证据说明](EVIDENCE_SCOPE_AND_SELECTED_HOLDOUT_20261123_ZH.md)。
- 本轮支持将两个有效 Piper 案例作为论文的**选定模型新布局证据**。不再为每个 seed 调到最优，也不根据本批结果调参后复用同一批布局声称新独立验证。SafeReach 前100K违规与任务表现的共同改进，以及 Panda 最终配置回归，仍是未完成项。

原始终点与紧凑数据：[队列](selected_holdout_complete_20261123/queue_terminal.json)、[Reach 收据](selected_holdout_complete_20261123/Reach/receipt.json)、[Reach 全回合](selected_holdout_complete_20261123/Reach/episodes.csv)、[Reach 配对比较](selected_holdout_complete_20261123/Reach/paired_against_metric.csv)、[Payload 收据](selected_holdout_complete_20261123/Payload/receipt.json)、[Payload 全回合](selected_holdout_complete_20261123/Payload/episodes.csv)、[Payload 配对比较](selected_holdout_complete_20261123/Payload/paired_against_metric.csv)。轨迹原件保留4080数据盘，不发布检查点。
