# 四条原生训练已完整结束：残差后期有效但不合格，独立父仍缺几何泛化

2026-11-19 16:09 UTC（北京时间11月20日00:09）核查：4080两条250K和A100两条300K均有完整原生完成receipt，实际训练PID已退出；两项自动汇总和终点绘图也正常完成，不是观察超时或失联。新完成1.1M transition，其中SafeReach配对500K、独立Piper父600K；此前4.75M完成计数加本轮为5.85M（不同阶段/预算分开比较，不把历史参考1M重复加为新增）。

## SafeReach终点余量残差250K：不晋升

|完整训练方案|前100K违规|后150K违规|全程违规|安全成功数|超时|stall episode|累计return|最终DEV成功：nom/r65/exec|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|当期原P5|295|387|682|10844|361|511|116858.4|28 / 26 / 28|
|终点余量残差|394|220|614|11595|332|412|126315.8|29 / 29 / 29|
|历史同4080 Metric|384|337|721|11843|287|437|128318.0|29 / 29 / 29|
|历史同4080 Iso|345|397|742|11353|383|505|122250.5|29 / 29 / 30|

候选相对原P5全程违规下降10.0%、训练安全成功增加6.9%、return增加8.1%，timeout/stall均减少；后150K违规下降43.2%，最终DEV87/90对82/90，所有条件均无timeout/stall。这个积极结果保留，不能简单称方法完全失败。

但关键联合要求仍失败：前100K违规394高于331.5、P5295、Iso345和Metric384；全程违规比Metric低14.8%、比Iso低17.3%，均未达到20%；最终成功仅追平Metric、不超过Iso；对Metric训练成功低2.1%、return低1.56%（超过1%容许退化）、timeout多45。25K严重任务放弃、50K恢复和100K回退都仍存在，不满足从一开始始终领先。故该候选**不进入多seed晋升、不继续扫系数、不选较好checkpoint替代250K**。

完整报告逐条列387项描述性DEV/训练标准，其中82项未满足；该数字不是科学p值或独立验收总分。当前P5对照与历史同4080 P5全部250个训练节点的计数/return精确一致，三个最终DEV计数也一致；Iso/Metric明确为历史参考，不冒称新同期重跑。

数据：[完整早晚窗口](endpoint_residual_complete_20261119/training_windows.csv)、[全126个DEV条件节点](endpoint_residual_complete_20261119/development_nodes.csv)、[3780回合（含四条历史参考）](endpoint_residual_complete_20261119/development_episodes.csv)、[逐项要求](endpoint_residual_complete_20261119/development_criteria.csv)、[来源/完成收据](endpoint_residual_complete_20261119/receipt.json)、[论文PNG](endpoint_residual_final_figures_20261119/endpoint_residual_learning_curves.png)、[PDF](endpoint_residual_final_figures_20261119/endpoint_residual_learning_curves.pdf)。新增训练只有两条500K，不是报告中含历史参考的1.5M全部新增。

## 两个独立Piper Reach父300K：学习恢复但55mm仍差

|真正独立seed|训练违规|训练安全成功|训练超时|最终nominal成功/30|最终55mm成功/30|最终exec成功/30|
|---|---:|---:|---:|---:|---:|---:|
|41028|558|4767|1482|24|6|25|
|41029|735|8091|395|29|15|29|

两条各300K、五个DEV节点全部完成，最终共900个DEV回合；41028从126K才出现首次训练成功，完整早期失败不删除。最终两个seed等权nominal/55mm/exec成功均值88.3%/35.0%/90.0%，样本SD11.8/21.2/9.4个百分点，最差seed80.0%/20.0%/83.3%；55mm最终无timeout但违规率65%，不是靠超时保安全，也未达到泛化要求。41028的240K较好55mm16/30不替代300K的6/30；41029也不挑60K的29/30。

这里是原生EA父阶段，尚未训练45/50/55半径覆盖，更不是P5 vs Metric的多seed确认或57.5/60mm独立验收。两个真实固定300K父SHA已登记，不照搬旧41027：41028为`a1cb1a5860a8210fb126e640068ae685b0233b09cdfd4e10ac5b288eca929703`，41029为`26061e7111e3f55bc2fed96adc348be46fc93ebd5430e27f381822e99a4f1e04`；均保留原seed、完整网络/optimizer/replay/计数。

数据：[全部父训练曲线](reach_parents_complete_20261119/training.csv)、[30个完整DEV条件节点](reach_parents_complete_20261119/development_nodes.csv)、[900个完整回合](reach_parents_complete_20261119/development_episodes.csv)、[等权seed均值/SD/最好/最差](reach_parents_complete_20261119/equal_seed_development.csv)、[真实父SHA](reach_parents_complete_20261119/fixed_parent_checkpoints.csv)、[完整原生收据](reach_parents_complete_20261119/receipt.json)。`complete_native_campaign_verified=true`只证明完整预算/身份，不证明父任务合格或最终方法优越。

## 下一步不缩减总目标

先保留完整已失败残差分支，停止SafeReach该系数搜索。Piper按原独立谱系准备每seed相同100K的45/50/55mm有限覆盖训练，到固定400K，不删除较差父、不选其他父；它是已规划独立多seed路径的第二阶段，不是新安全机制候选。两条400K完成后，三臂150K方法确认必须另固定配置/父SHA，不能自动启动或用后期成功率消除父阶段失败。

全部任务/几何对Metric优势、前100K风险、timeout/stall/return、Panda≤2pp、最终3seed和新独立评测仍未完成，不能宣布全面最优。新大文件继续只写数据盘，A100当前约51GiB可用，启动前按真实增长保持40GiB底线；不使用5080或实体机器人，不停止共享外部任务，不新增删除。当前旧队列已终止，不能继续声称四条仍在运行；任何新训练另报实际PID/进度。
