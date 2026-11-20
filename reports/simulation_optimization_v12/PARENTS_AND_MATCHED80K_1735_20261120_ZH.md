# 17:35进展：第三Reach父恢复，Payload仍未学会，A100同80K对照已齐

2026-11-20 17:31–17:35（UTC+8）读取实际进程和完整已保存前缀，四条训练均仍运行，没有重启、换seed、追加候选或修改中途配方。所有结果为DEV/训练诊断，不是最终冻结独立评测。

## 三个父的完整已观察学习情况

|主机 / 原生父seed|实际步数 / 预算|安全成功|违规|timeout|最近20K速度|训练步数外推剩余|
|---|---|---:|---:|---:|---:|---:|
|4080 Reach41030|162K / 300K|2292|360|846|19.88步/秒|1.93小时|
|4080 Payload42029|116K / 300K|0|85|1118|19.88步/秒|2.57小时|
|A100 Payload42028|60K / 300K|0|86|569|10.05步/秒|6.64小时|

ETA使用完整最近20K日志墙钟窗口，不采用随机预热阶段的高速度；窗口已包含过去发生的评测，未来保存/评测和共享负载变化未保证覆盖，不能当作完整研究结束时间。A100 Reach Metric在17:31新增97K/150K，约9.18新步/秒，剩余53K的训练外推约1.60小时，最终DEV还有开销。

第三Reach父首次成功出现在67K日志节点之前或当时，不伪称精确首次成功transition。60K固定DEV仍为nominal0/30、55mm0/30、execution0/30（违规1、timeout89）；120K为30/30、21/30、30/30（违规9、timeout0）。这是真实从早期任务失败中恢复，但55mm仍只有70%，不是60mm≥90%跨seed确认。完整0–162K曲线及全部DEV均保留，未删60K负结果或挑120K代替最终父。

Payload42029到116K仍零训练成功，已保存60K DEV三个条件均0/30、合计90timeout、0违规；42028到60K仍零训练成功，快照时首个60K评测尚未原子完成，明确缺失，不拿另一seed补作本seed结果。两条actor最近日志均继续更新，数值有限，属于任务学习困难而非已证实进程挂死。低违规来自大量timeout，不能当安全收益。

Payload42029当前安全乘子约0.0247、policy Q约−0.317，alpha约1.33e−5；42028乘子约0.255。它们与早期任务学习停滞相容，不能确定唯一原因或宣称只需放松安全约束。alpha是kinetic-energy权重，不等于动作方差同比下降。继续各300K完整预算，不因零成功换seed或提前停止。

## A100同80K三方法：困难几何较好，联合代价仍在

同seed41028、同一400K完整父、同主机、相同新增80K节点，每条件100个相同DEV布局，sampled部署：

|方法|57.5mm / 60mm成功率|后缀违规|后缀安全成功|timeout / stall episode|累计return|
|---|---|---:|---:|---|---:|
|固定P5|93% / 78%|42|2586|1 / 1|27425.33|
|P5+2mm|98% / 90%|47|2570|1 / 1|27198.88|
|Metric-only|90% / 72%|198|2627|0 / 0|26385.10|

该节点候选达到57.5mm≥93%/60mm≥90%，训练违规对Metric少76.26%，但训练成功少57、多1timeout/stall；对固定P5违规反而多11.90%、成功少16。不能只报任务成功率或减少违规而称所有指标领先。nominal/execution均100%，所有1200个DEV回合无timeout/stall，训练代价与DEV代价分开保留。

同时补齐A100同60K三方法节点，保留全部20/40/60/80K及原P5后续节点；Metric尚未完成150K，不计算完整双seed相对Metric最终优胜。已发布的双seed终点Reach候选91%/97%仍是固定150K DEV，不改写为全程达标或独立测试。

## 资源、记录与下一步

17:31快照A100两张GPU利用率20%/21%，各有原生进程；4080整卡88%，共享负载包括其他项目，不全计本实验。数据盘A100约48G、4080约431G可用，A100仍执行40GiB底线+既有未写入输出+Payload完整增长预算，4080写4T盘。没有删除文件、向home放模型或暂停外部项目。

- [A100父诊断收据及真实进程证据](parent_progress_1735_20261120/a100/receipt.json)、[全部0–60K训练前缀](parent_progress_1735_20261120/a100/training.csv)。该快照首个评测尚未完成，因此没有DEV CSV，不填零假造结果。
- [4080父诊断收据](parent_progress_1735_20261120/rtx4080/receipt.json)、[两个父完整训练前缀](parent_progress_1735_20261120/rtx4080/training.csv)、[全部9个DEV条件节点](parent_progress_1735_20261120/rtx4080/development_nodes.csv)、[全部270回合](parent_progress_1735_20261120/rtx4080/development_episodes.csv)。
- [A100同80K三方法来源收据](crossseed_dev80K_41028_20261120_1735/receipt.json)、[1200回合](crossseed_dev80K_41028_20261120_1735/development_episodes.csv)、[全部0–80K训练曲线](crossseed_dev80K_41028_20261120_1735/training.csv)、[同60K收据](crossseed_dev60K_41028_20261120_1735/receipt.json)。
- [更新学习曲线PNG](crossseed_learning_curves_1735_20261120/reach_crossseed_progress.png)、[PDF](crossseed_learning_curves_1735_20261120/reach_crossseed_progress.pdf)、[输入SHA与部分完成标记](crossseed_learning_curves_1735_20261120/provenance.json)。
- [只读紧凑父快照程序](../../scripts/paper_crax_payload/snapshot_independent_parents_20261120.py)：绑定三条实际源码/任务/seed/配方，忽略未完成append尾行，校验全部episode与原生汇总一致，记录文件SHA，不调用策略、不下载模型或修改训练。两项针对性检查通过。

半小时资源监测和900K双seed终点本地自动归档进程仍live；下次按各自1800秒周期记录，不高频盯盘。继续四条完整预算，等原生真实终点后绑定SHA；不以父准备替代Payload优化，Panda回归、SafeReach早期目标、第三seed方法适应及新冻结独立测试仍未完成。
