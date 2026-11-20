# 900K双seed三方法完整结论：Reach安全改善明确，联合最优仍未成立

2026-11-20 19:48（UTC+8）更新。两个真实独立训练seed41028/41029、固定P5/P5+2mm/Metric-only六条均正常完成各新增150K到总550K，共900K实际新增transition、192个条件/部署模式节点和19200个DEV回合。A100最后Metric于19:16完成；服务在19:37自动收齐两机紧凑汇总并合并，随后正常结束，不是监测崩溃。没有按中间DEV择优、跳过失败seed或使用独立测试选模。

## 固定150K终点：相同模式下Reach困难几何成功率优于Metric

每训练seed、每条件、每模式100回合，等权两个seed；nominal/execution085全部100%，保留成功率上限处的平局。

|方法|sampled 57.5mm均值|sampled60mm均值 / 最差 / 样本SD|zero_noise57.5mm均值|zero_noise60mm均值 / 最差 / 样本SD|
|---|---:|---|---:|---|
|固定P5|98.5%|92% / 86% / 8.49pp|100%|99.5% / 99% / 0.71pp|
|P5+2mm|98.5%|94% / 91% / 4.24pp|100%|98.5% / 97% / 2.12pp|
|Metric-only|92.5%|85.5% / 85% / 0.71pp|98%|94.5% / 94% / 0.71pp|

P5+2mm sampled两个seed57.5mm分别98%/99%、60mm91%/97%，均达到用户目标；60mm均值比同模式Metric高8.5pp。zero_noise固定P5比同模式Metric高5pp，+2mm高4pp。不能拿SAFEFlow zero_noise与Metric sampled比较，也不能忽略固定P5 zero_noise的60mm反而优于+2mm。固定P5 sampled的seed41028仅86%，仍低于90%，SD8.49pp也不满足≤5pp要求。

所有四个P5终点/两模式无DEV timeout/stall；Metric在41028 sampled57.5mm有1次timeout/stall，其他终点组无此问题。记录保留，不把Metric失败删成零。

## 全部训练后缀：违规大减，成功数和timeout/stall有代价

各方法合计两个相同150K后缀，总300K训练预算；原400K父历史风险保留，不重复计新训练。

|方法|训练违规|训练安全成功|timeout|stall episode|累计return|
|---|---:|---:|---:|---:|---:|
|固定P5|138|9754|2|2|103667.10|
|P5+2mm|120|9701|3|3|103249.45|
|Metric-only|575|9998|1|1|102192.53|

候选对Metric违规少79.13%、累计return高1.03%，但安全成功少297（约2.97%）、timeout/stall各多2；固定P5对Metric违规少76.00%，累计return高1.44%，训练成功少244，timeout/stall各多1。候选对固定P5违规少13.04%，成功少53、timeout/stall各多1、累计return少0.40%。**训练低违规并没有共同消除任务代价，因此不冻结成全面winner。**

逐seed结果和检查全部保留：[六终点计数](crossseed_complete_combined_20261120/training_endpoints.csv)、[逐seed对Metric比较](crossseed_complete_combined_20261120/training_comparisons_vs_metric.csv)。候选在两个seed都满足违规至少低20%及return不降超过1%，但两个seed训练成功数都较低、timeout/stall都较高，不能用合计掩盖失败seed。

## 完整学习曲线：仅固定DEV节点的任务表现，不能推成训练起点始终最优

按预登记同模式、每个seed成功不低于Metric、违规/timeout/stall不增加、每seed return不下降超过1%的描述性检查：候选sampled24/24个条件节点符合、zero_noise8/8符合；固定P5分别23/24、8/8。固定P5不符合的情况是100K sampled60mm在41028为76%对Metric77%。

这是有限节点的DEV表现，允许100%上限平局及1% return容差，不是所有transition、所有状态或独立集上的严格支配。候选在训练成功数和timeout/stall上仍失败；同批没有Iso，且该实验从已训练400K父开始，不能证明SafeReach从头前100K≤331.5或从训练起点始终领先。nominal的sampled capped时间候选1.2004秒、Metric1.1352秒，约慢5.74%，不能泛称所有效率指标优胜；推理延迟与任务完成时间也不能混为一谈。

## 第三Reach固定300K父已完成，实际接续原生100K扩域

第三真正独立seed41030正常完成300K，完整父训练计数：7064安全成功、778违规、846timeout。固定300K DEV为nominal27/30、55mm9/30、execution08521/30，合计57/90成功、33违规、0timeout。55mm从120K21/30、240K16/30进一步降到9/30；保留最终回退，不换成最好中间父，也不将EA父结果当作SAFEFlow第三seed确认。

原父checkpoint SHA256为`342de2dd71798dc69739b42b7d5ad8af42f9d20f03fb37fa1ffcac019cc1cd3e`；完成收据及run_config也绑定真实SHA。原父进程已退出、全部五DEV节点仍在，19:46:53已接续原定100K半径覆盖，目标总400K，训练域45/50/55mm，不训练57.5/60mm。原生fork保留网络、优化器、replay、alpha、dual和失败计数；按原训练seed重新seed随机流，明确不是精确RNG resume。

- 扩域源码`255b737b69c61481e094043789f1ac6ae53a04e2`，queue PID2715497、原生PID2715499、半小时监测PID2715500。
- 启动后已实际到301K，即新增1K：新增28安全成功、16违规、0timeout；累计7092/794/846。learner从298000到299000、actor从148500到149000，确认真实学习，不把启动命令或队列状态当作已采集transition。
- 仅共享4080 GPU0，未暂停Payload或外部项目。4T盘约422G可用，输出与缓存留盘，每次保存留40GiB及本任务/Payload各2GiB增长预算。
- 未启动第三seed P5/Metric方法对照；须等真实400K终点后另行固定方案，不自动补选模型或增加扫描。

## 当前剩余运行与未完成事项

A100 Payload父135K/300K、4080 Payload父272K/300K，均仍零训练成功。4080此前210K附近Q/违规骤增后，272K Q又回到约−0.664，失败并未因数值回落变成成功；全部阶段保留。两条继续完整300K，不中途改奖励、timeout或换seed。A100当前GPU0已由完成的Metric释放，保留共享资源，不为填满显存重复或启动尚无具体诊断依据的参数扫描。

当前运行三条：A100 Payload父、4080 Payload父及Reach第三seed扩域，分别有半小时资源监测。旧900K监测/终点汇总已正常结束；本地归档服务也在完成既定合并后正常结束，无需重启或制造重复训练。接下来完成第三seed扩域和两个Payload父，依据完整结果解决任务学习问题，再准备最终方法/第三seed比较、Panda回归及新冻结独立测试。全面最优目标继续未完成。

## 完整证据与本地归档

- [900K完整合并收据](crossseed_complete_combined_20261120/receipt.json)、[全部双seed等权结果](crossseed_complete_combined_20261120/equal_seed_development.csv)、[每节点同模式Metric比较](crossseed_complete_combined_20261120/comparisons_vs_metric.csv)。
- [A100完整9600个DEV回合](crossseed_complete_41028_20261120/development_episodes.csv)、[全部训练曲线](crossseed_complete_41028_20261120/training.csv)、[六方法原生来源及终点SHA中的本机部分](crossseed_complete_41028_20261120/receipt.json)；[4080完整9600回合](crossseed_complete_41029_20261120/development_episodes.csv)、[训练曲线](crossseed_complete_41029_20261120/training.csv)、[来源SHA](crossseed_complete_41029_20261120/receipt.json)。
- [完整双seed学习曲线PNG](crossseed_learning_curves_complete_20261120_v2/reach_crossseed_progress.png)、[论文PDF](crossseed_learning_curves_complete_20261120_v2/reach_crossseed_progress.pdf)、[输入SHA](crossseed_learning_curves_complete_20261120_v2/provenance.json)。图绘sampled；zero_noise100K/150K节点全部保留在原始表，不混模式。
- [第三父300K完成收据](third_radius_execution_20261120/parent_receipt.json)、[全部五节点及同时观察的Payload前缀](parent_progress_1948_20261120/rtx4080/receipt.json)、[全部DEV回合](parent_progress_1948_20261120/rtx4080/development_episodes.csv)。
- [第三扩域实际队列](third_radius_execution_20261120/queue_third_radius_20261120.json)、[三文件SHA及继承计数登记](third_radius_execution_20261120/registration.json)、[实际301K原生状态](third_radius_execution_20261120/status.json)、[新运行配置](third_radius_execution_20261120/run_config.json)。

本次只下载CSV/JSON/图，不下载模型，关键内容继续上传GitHub并归档本地。没有邮件、PR、评论、删除、5080或实体机器人动作。独立测试未用于选择、最终方法未冻结，不能将900K批次完成写成整个研究目标完成。
