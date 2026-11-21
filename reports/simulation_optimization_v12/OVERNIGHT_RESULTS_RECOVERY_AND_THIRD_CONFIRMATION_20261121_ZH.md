# 11月21日接续：第三Reach扩域完成、Payload完整失败与磁盘中断恢复

2026-11-21 15:00–15:08（UTC+8）。先核实原生完成收据、旧队列退出状态及/proc实际进程，再启动明确接续；没有把旧监测进程正常结束误当作仍在训练，没有重复启动完成的第三Reach扩域或Payload42029。

## 昨夜两条完成、一条工程中断

|原生任务|实际预算与状态|安全成功 / 违规 / timeout|最终DEV：nominal / 55mm / execution085|
|---|---|---|---|
|Reach第三父41030扩域|300K→400K，新增100K，完成exit0|新增3245成功、361违规、0timeout；累计10309 / 1139 / 846|30/30、24/30、30/30|
|Payload父42029|从头300K，完成exit0|0 / 270 / 2929|0/30、0/30、0/30；1违规、89timeout|
|A100 Payload父42028|观察到175K/300K，exit1工程中断|0 / 167 / 1678|只完成60K、120K两个节点；不能冒充300K终点|

Reach第三父55mm从300K的9/30到400K的24/30，扩域后任务恢复，但55mm已属于训练域，不可写作未见几何泛化。400K固定原生checkpoint SHA256为`9fe1abe406ff7a18dd4415018605e487039bfc7fdc8c95548a3f180201d8f0f9`；不选较好的中间父，也不把EA父结果写成第三seed P5确认完成。

Payload42029全五个DEV节点都零成功，共450回合0成功、32违规、418timeout；训练300K也0成功。中途Q/动作幅度曾明显增大，后又回落，均没有转化为任务成功，完整轨迹计数保留。这个新独立父为明确负结果，不换seed、不用历史41027父代替。继续优化Payload必须解决任务学习与安全的共同问题，不能将低违规的超时策略算作安全收益。

## A100中断原因与断点恢复边界

A100日志在175K保存时触发`Preserve40GiB plus all pending existing outputs and full Payload output budget`。异常处理再次保存中断checkpoint也触发同一保护，所以没有175K完整可恢复状态；status仍写着running仅是退出前旧文件，真实队列为`completed_with_failures`/exit1，原PID452062已不存在。11月21日15:00数据盘只剩约25G，低于原40GiB底线，不能继续往该盘产生大存档。

最后可恢复完整状态为150K的`latest_full.pt`（65.56MiB），SHA256两机一致：`6839e61e35f7b5ead69b1cf9f1c4b9d27b24141cb3f1795a33aed538fa4f0579`。原始150K checkpoint记录157违规、0成功、1429timeout，并含原网络、优化器、replay、dual、alpha、两个环境及partial episode、CPU/CUDA随机状态。检查点及原配置/日志已直接A100→4080流式复制，未在本地主机落模型。

显式在4080新目录从150K恢复到固定总300K，不从头重训、不换seed、不改变配方。原A100150K–175K的25K未保存片段保留在原始日志及独立失败汇总中，**不会与恢复后的150K–175K重叠段拼接成单一连续轨迹、抹去风险或重复记作独立预算**。恢复后若到300K，所选可恢复谱系是原0–150K+新150K–300K，实际计算还应另披露原丢失25K；原175K中断尝试始终为失败记录。跨GPU恢复不声称逐位等价，随机状态及原生任务定义按原入口恢复。

恢复源码固定`075f8f5ec117e2b331b3909ba1521c0a02981b7c`，queue743366、原生743368、半小时监测743369。15:07启动验收已实际到151K、learner149000、actor74000，较断点新增1K、1000次learner/500次actor更新；累计0成功、161违规、1439timeout。模型状态已成功加载、环境重建与原生随机恢复已走通，不能仅凭队列running宣称恢复成功。此后按半小时监测，原A100失败队列不改写或自动重试。

## 第三个真正独立seed的三方法150K确认已实际启动

同41030固定400K完整父，在4080进行固定P5、P5+2mm、Metric-only各新增150K，合计450K。沿用前两seed的原生数学更新、候选参数、LR、训练45/50/55mm分布、所有六个DEV节点（20/40/60/80/100/150K）、四条件每节点100回合；100K/150K三方法均同样评估sampled与zero_noise。仍使用12020000起DEV布局、12021000起动作采样seed，不用独立测试集选择。

这次是固定既有配方的第三seed检验，不根据昨夜新结果再调offset或挑种子。所有P5/Metric方法共享同一真实41030父，但41030父本身由独立从头训练产生；不同布局不是新训练seed。第三seed结果尚未产生，不能提前与前两seed均值合成“三seed确认通过”。

|任务|原生进程|15:07启动验收实际进度|成功 / 违规 / timeout|
|---|---:|---|---|
|固定P5|743373|总401K，新增1K|30 / 2 / 0|
|P5+2mm|743375|总401K，新增1K|30 / 3 / 0|
|Metric-only|尚未启动|等待任一确认槽释放|无结果，不填零|

确认queue743370、半小时监测743376，源码同`075f8f5e`。三方法150K从同预算起点、同一设备共享资源，Metric顺序接续；两P5和恢复任务当前共三条共享4080 GPU0。启动验收整卡利用率98%、显存1510MiB，包括其他共享进程，未停止任何其他项目；不能把显存少认作没有使用GPU。4T盘初始约415G可用，新存档、缓存、临时文件全部留盘，保存前保留40GiB加本次三臂各2GiB及恢复2GiB共8GiB增长预算。

两项针对性测试在4080实验环境通过，确认三方法所有科学配方字段与前两seed模板一致，拒绝变更父SHA/seed/预算；另实际读取转移检查点确认与A100原生配置、计数、格式相符。没有重复运行全套测试。已有两seed900K结果仍保留：[完整报告](CROSSSEED900K_COMPLETE_AND_THIRD_RADIUS_STARTED_20261120_ZH.md)。

## 证据与继续事项

- [昨夜完成/中断汇总收据](overnight_summary_20261121/receipt.json)、[端点计数](overnight_summary_20261121/endpoints.csv)、[完整观察学习曲线](overnight_summary_20261121/training.csv)、[全部24个DEV节点](overnight_summary_20261121/development_nodes.csv)、[720回合明细](overnight_summary_20261121/development_episodes.csv)。原始JSON/progress本地归档，收据保留逐文件SHA；大检查点仍留服务器。
- [A100真实exit1失败队列](recovery_confirmation_execution_20261121/a100_failed_queue.json)、[恢复谱系与失去25K明确登记](recovery_confirmation_execution_20261121/recovery/registration.json)、[恢复后真实status](recovery_confirmation_execution_20261121/recovery/status.json)、[原生恢复配置](recovery_confirmation_execution_20261121/recovery/run_config.json)。
- [第三seed确认实际队列](recovery_confirmation_execution_20261121/queue_third_confirmation_20261121.json)、[固定P5真实状态](recovery_confirmation_execution_20261121/fixed/status.json)、[+2mm真实状态](recovery_confirmation_execution_20261121/activation2mm/status.json)、[新配方登记](../../configs/paper_crax_payload/simulation_v12_third_seed_confirmation_20261121.json)。启动后归档的状态可能比15:07验收略晚，保留原文件，不强造同一时刻。
- [显式恢复入口](../../scripts/paper_crax_payload/resume_payload42028_20261121.py)、[第三seed原生确认入口](../../scripts/paper_crax_payload/train_third_seed_confirmation_20261121.py)、[昨夜只读汇总程序](../../scripts/paper_crax_payload/summarize_overnight_20261121.py)。

A100因数据盘25G暂不新增大任务、不往home回退写模型，也未删除证据；在4080继续授权的恢复和独立seed确认。三个实际进程及一个排队任务每30分钟监测。后续须完成恢复300K和第三seed三方法，再解决Payload任务学习/Panda回归/SafeReach早期目标并冻结新独立评测，全面最优目标仍未完成。不使用5080/实机、不发邮件或创建PR。
