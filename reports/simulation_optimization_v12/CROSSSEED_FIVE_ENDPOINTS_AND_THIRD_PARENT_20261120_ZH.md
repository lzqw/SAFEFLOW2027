# 双seed四个P5终点已齐：Reach目标达到，联合优胜尚未成立

2026-11-20 15:38（UTC+8）更新。4080的三方法各新增150K、合计450K已正常完成，原生汇总保留全部9600个DEV回合。A100的固定P5与P5+2mm也已完成各150K；最后一条Metric-only在15:34实际进程376702仍运行，原生进度新增32K。当前900K批次已完成五个完整终点750K，不能把未完成的Metric记成完整双seed确认。

## Reach的数值目标已在两个真实训练seed的固定终点达到

所有结果均取预登记新增150K/总550K终点，每条件100个相同DEV布局；不是最好中间模型，也不是独立测试。两seed来自各自独立原生EA父，后续方法共享同seed父，继承400K不重复计作新训练。

|方法/模式|41028：57.5mm / 60mm|41029：57.5mm / 60mm|等权均值：57.5mm / 60mm|60mm最差 / 样本SD|
|---|---|---|---|---|
|固定P5 sampled|97% / 86%|100% / 98%|98.5% / 92%|86% / 8.49pp|
|P5+2mm sampled|98% / 91%|99% / 97%|98.5% / 94%|91% / 4.24pp|
|固定P5 zero_noise|100% / 99%|100% / 100%|100% / 99.5%|99% / 0.71pp|
|P5+2mm zero_noise|100% / 97%|100% / 100%|100% / 98.5%|97% / 2.12pp|

P5+2mm sampled两个seed分别达到57.5mm≥93%、60mm≥90%，57.5mm也均超过理想95%；60mm最差91%、跨seed SD4.24pp。其57.5mm跨seed SD0.71pp。nominal/execution085在这四个P5终点、两模式全部100%，终点DEV均零timeout/stall。这支持困难几何成功率与稳定性进展，但不能证明训练没有timeout，也不把两种部署模式混合。

固定P5在sampled的seed41028仍低于90%，且SD超过5pp。zero_noise下固定P5的60mm均值反而高于+2mm，因此不能声称扩大激活余量在所有模式都更好。此前40K和100K等中间节点也有候选落后固定P5的情况，全部保留在曲线中。

## 4080已有同期完整Metric，任务安全改善但训练存在代价

|41029方案|新增150K训练违规|安全成功|训练timeout / stall episode|累计return|终点sampled 57.5 / 60mm|终点zero_noise 57.5 / 60mm|
|---|---:|---:|---|---:|---|---|
|固定P5|60|4892|0 / 0|52082.54|100% / 98%|100% / 100%|
|P5+2mm|50|4870|1 / 1|51936.12|99% / 97%|100% / 100%|
|Metric-only|268|5033|0 / 0|51659.08|93% / 85%|100% / 95%|

该seed中，候选相对Metric训练违规少81.34%，但训练成功少163次，并多1次timeout/stall；候选对固定P5违规只少16.67%，安全成功少22次、return少0.28%。不能仅凭低违规晋升。终点四条件评测无timeout/stall；nominal/execution成功均为100%，不是严格超过100%的基线。失败按原horizon计入capped时间，未修改安全距离、timeout或停滞规则。

## 双seed训练后缀的完整代价

|seed/方法|违规|安全成功|timeout|stall episode|累计return|
|---|---:|---:|---:|---:|---:|
|41028固定P5|78|4862|2|2|51584.56|
|41028 P5+2mm|70|4831|2|2|51313.33|
|41029固定P5|60|4892|0|0|52082.54|
|41029 P5+2mm|50|4870|1|1|51936.12|

候选合计违规120对138（下降13.04%），安全成功9701对9754，timeout/stall各3对2，return合计下降约0.40%。这些是400K–550K训练后缀；不能拿来证明SafeReach从头前100K≤331.5，也不能抹去父训练历史风险。A100的Metric未完成，禁止跨seed提前计算相对Metric最终优胜。

## 继续利用已释放4080：第三个真正独立父已启动

15:37（UTC+8）在4080启动seed41030、原生EA Reach固定300K从头训练，补第三seed稳定性证据。它复制前两个独立父的同一300K配方，只改变训练seed；网络、优化器、replay、dual和环境RNG均重新初始化，不加载任何父模型或Panda权重。仍为原名义训练域，不把后续扩域准备冒充已完成。

- 源码：`58bf769dceb1eabd2eec07c627decf68d810c2aa`；queue PID2203197、原生训练PID2203200、半小时监测PID2203203。
- 15:38原生进程已核实存活，实际新增2K，仍处原配方随机预热；7违规、0成功、17timeout全部保留。这是父模型早期采集，不是P5方法改进或稳定吞吐估计。
- 15:41启动验收已到12K、learner更新10000次、actor更新4500次，GPU上进程2203200占用338MiB，已进入真实学习；34违规、0成功、115timeout全部保留，尚未证明父训练成功。启动验收后恢复半小时监测，不高频轮询。
- 使用原共享GPU0，不停止其他项目；输出、缓存、临时文件均在4080的4T盘，启动前盘可用约447GiB，保留40GiB+2GiB预计输出预算。
- 只启动这一条300K；后续100K扩域必须取得真实300K检查点后另行绑定，P5/Metric适应阶段也不能在父缺失时自动启动。不按中期表现换seed、选最好父或提前停止。
- 两项必要配方测试在4080实际实验环境通过，确认只改seed并拒绝改预算/复用seed；本地主机环境缺少panda_gym，未在本地安装额外训练依赖。

A100最后一条Metric保持原900K源码77a517d5与完整150K预算；按15:34约9.45新步/秒、尚余118K估计还需约3.5–4小时，包括后续DEV开销，共享负载变化会影响时间。该估计只针对本批Metric，不是全部研究目标完成承诺。A100数据盘15:21可用48.64GiB，不再往home写大文件；本轮已登记的40GiB加剩余输出预算继续生效。

## 证据、图与下一步

- [完整sampled学习曲线PNG](crossseed_learning_curves_1538_20261120/reach_crossseed_progress.png)、[论文PDF](crossseed_learning_curves_1538_20261120/reach_crossseed_progress.pdf)、[来源SHA及已观测前缀](crossseed_learning_curves_1538_20261120/provenance.json)。图含所有已归档固定节点和完整1K训练前缀；A100 Metric曲线仅到已归档20K，15:21新增25K计数用孤立x标记，不连接成完整曲线。
- [4080完整三方法收据和真实终点SHA](crossseed_complete_41029_20261120/receipt.json)、[全部96条件/模式节点](crossseed_complete_41029_20261120/development_nodes.csv)、[全部9600回合](crossseed_complete_41029_20261120/development_episodes.csv)、[完整训练曲线](crossseed_complete_41029_20261120/training.csv)。
- [A100固定P5完成收据](crossseed_completed_receipts_41028_20261120/P5-fixed.json)、[候选完成收据](crossseed_completed_receipts_41028_20261120/P5-activation2mm.json)、[双方法150K完整节点](crossseed_dev150K_41028_20261120_1535/development_nodes.csv)、[逐回合/来源收据](crossseed_dev150K_41028_20261120_1535/receipt.json)。其20/40/60/80/100K亦全部归档，没有只保留终点。
- [第三父实际队列](third_parent_execution_20261120/queue_third_parent_20261120.json)、[从头运行配置](third_parent_execution_20261120/run_config.json)、[配方来源与登记](third_parent_execution_20261120/registration.json)。
- [15:21 A100资源原始记录](crossseed_resource_1521_20261120/a100.json)、[完整未完成要求台账](REQUIREMENTS_STATUS_ZH.md)。

等待A100同预算Metric结束后，现有半小时自动归档会合并192条件/模式节点与19200个DEV回合，保留逐seed胜平负、训练风险及任务代价；它不自动冻结方法或上传GitHub。当前上传仅关键程序和紧凑结果，未创建PR、评论或发送邮件。第三父与两台资源监测每30分钟记录，不使用5080/实机。

全面优于Metric、训练起点始终领先、Payload55/57.5mm优化、最终配置Panda回归、第三seed适应结果和新冻结独立测试仍未完成。本次不据Reach数值达标宣布研究完成，也不追加offset/温度密集扫描。
