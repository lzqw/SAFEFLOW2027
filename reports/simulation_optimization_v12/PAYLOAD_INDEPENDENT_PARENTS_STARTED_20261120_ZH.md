# Payload双独立父已实际启动：利用A100空卡，继续完整预算

2026-11-20 16:06（UTC+8）。原900K Reach三方法实验仍继续，未重启或改源码。15:58实际读取A100 Metric进程376702：新增45K/150K、123违规、1467安全成功、零timeout/stall；4080第三Reach父进程2203200：41K/300K、72违规、0成功、382timeout，不能据正常更新判断该seed已经学会任务。

## 新增两条Payload原生父训练，各固定300K

之前Payload的方向项、激活余量及部署对照主要来自同一个41027父，独立训练稳定性仍缺。现在预先固定两个新seed42028/42029，各从头原生EA Payload300K，合计600K；不导入Reach或Panda模型、优化器、replay或随机流。这一步准备真实独立父，**不是新P5候选胜出，也没有新增最终成功率结果**。

仅在原始`piper_sim_v1.json`配方中替换训练seed、DEV布局和动作采样seed。训练仍是原nominal域，没有加入55/57.5mm训练半径；原horizon、15mm安全阈值、奖励、网络、更新次数和终止规则保持原定义。DEV为12010000起、动作12011000起，固定60/120/180/240/300K五个节点、原三条件各30回合；原验证/独立集不读取，不选最好中间父。

|主机 / 训练seed|实际GPU|queue / 原生 / 半小时监测PID|完整预算|
|---|---|---|---:|
|A100 / 42028|物理GPU1，进程内cuda:0|452060 / 452062 / 452068|300K|
|4080 / 42029|GPU0，与Reach第三父共享|2213975 / 2213979 / 2213982|300K|

两条均在16:04:11实际启动，原生源码固定`870e7e81ce29ab0b06a24fe546544d1f848ad666`。16:06启动验收读取了实际/proc命令、status和GPU进程：A100已到4K，learner更新2000次、actor更新500次；4080已到9K，learner更新7000次、actor更新3000次。两条训练成功均0，违规51/45、timeout27/72，全部保留；这些早期失败不能删除，也不是未来P5方法的独立评估。

A100原生Payload进程显存536MiB，原Reach Metric进程542MiB，两张卡快照利用率22%/20%；4080 Payload进程332MiB、Reach第三父338MiB，整卡利用率85%。整卡数字含共享负载，不将其他项目用量计成本实验，不暂停外部进程，也不把显存占用当作训练速度或方法优势。启动状态文件归档晚于上述验收，可能包含更长的真实前缀。

## 存储与工程保护

A100启动前数据盘可用48.03GiB，明确要求46.39GiB：40GiB绝对空闲底线、原900K三条A100输出的全部未写入预算、该Payload父完整2GiB预算。余量约1.64GiB已在这些预算之外。不能因为两条P5已经完成就忽略其仍可能替换的mutable存档；仅信用已物化的不可变milestone文件。

新Payload父在初始化及每次保存前再次检查该空间预算，原生训练增加了默认关闭的可选storage_guard回调；数学更新未改变，已有运行源码目录不修改。4080同时预留Reach第三父2GiB、新Payload父2GiB和40GiB底线。模型、replay、缓存、临时文件均写数据盘；本地只归档小JSON、CSV和图，不下载检查点。

两项针对性测试在4080实验环境通过：确认只改变登记的seed/DEV字段并拒绝预算或seed替换；确认A100三臂预算及新输出预算全部纳入，缺少旧臂时拒绝。没有追加测试全套、增加actor预热、重采样、余量扫描或新安全机制。新队列没有科学提前停止、自动重试或自动下游训练；后续扩域/P5/Metric比较须绑定真实300K终点及原生SHA后另行登记。

## A100固定40K已补上同期Metric，仍有任务—风险代价

同41028父、同一主机、相同新增40K节点、每条件100个相同DEV布局，部署sampled：

|方法|57.5mm / 60mm成功率|训练违规|训练安全成功|训练timeout / stall episode|训练累计return|
|---|---|---:|---:|---|---:|
|固定P5|92% / 86%|18|1289|0 / 0|13697.32|
|P5+2mm|95% / 86%|16|1284|1 / 1|13661.11|
|Metric-only|86% / 68%|112|1302|0 / 0|12942.47|

三个方法该节点nominal/execution085均100%，全部DEV无timeout/stall。候选困难几何成功多于同期Metric、训练违规更少，但安全成功少18次，并新增训练timeout/stall，尚非联合优胜。40K的60mm仅86%，不拿最终双seed91%/97%冒充从训练起点始终达标。400K父以前风险不从这些后缀计数中抹去，SafeReach从头前100K≤331.5仍是另一项未解决要求。

数据与程序：

- [A100实际队列](payload_parent_execution_20261120/a100/queue_payload_parent_42028_20261120.json)、[原生配置](payload_parent_execution_20261120/a100/run_config.json)、[来源/存储预算登记](payload_parent_execution_20261120/a100/registration.json)、[启动后真实状态](payload_parent_execution_20261120/a100/status.json)。
- [4080实际队列](payload_parent_execution_20261120/rtx4080/queue_payload_parent_42029_20261120.json)、[原生配置](payload_parent_execution_20261120/rtx4080/run_config.json)、[来源登记](payload_parent_execution_20261120/rtx4080/registration.json)、[启动后真实状态](payload_parent_execution_20261120/rtx4080/status.json)。
- [A100固定40K全部三方法收据](crossseed_dev40K_41028_20261120_1605/receipt.json)、[1200回合](crossseed_dev40K_41028_20261120_1605/development_episodes.csv)、[完整0–40K曲线](crossseed_dev40K_41028_20261120_1605/training.csv)。
- [更新全部已观测sampled曲线PNG](crossseed_learning_curves_1605_20261120/reach_crossseed_progress.png)、[PDF](crossseed_learning_curves_1605_20261120/reach_crossseed_progress.pdf)、[每个数据输入的SHA](crossseed_learning_curves_1605_20261120/provenance.json)。A100 Metric仅连接已归档0–40K，不外推成150K。
- [两个Payload父的固定计划](../../configs/paper_crax_payload/simulation_v12_payload_fresh_parents_20261120.json)、[原生入口与空间保护](../../scripts/paper_crax_payload/train_payload_fresh_parents_20261120.py)、[之前Reach五个终点与联合代价](CROSSSEED_FIVE_ENDPOINTS_AND_THIRD_PARENT_20261120_ZH.md)。

两机各自按1800秒保存资源/原生进度；旧900K终点汇总和本地自动归档仍在运行。下一步保持当前四条真实训练的完整预算，完成Reach双seedMetric汇总，并取得第三Reach父及两个Payload父真实固定终点。当前没有冻结最终新方法；Payload困难几何优化、Panda最终配置≤2pp回归、SafeReach全曲线及新独立测试均未完成，仍不使用5080/实机，不发邮件或创建PR。
