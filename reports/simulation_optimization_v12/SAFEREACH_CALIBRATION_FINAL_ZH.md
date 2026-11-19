# 公共30K真实动作池：不支持“P5 margin明显劣于Metric”的假设

源码`19b9bd4f`，A100数据盘已完成原登记的冻结回顾性诊断。P5、Iso、Metric各提供自己的240001–250000全部实际transition，共30K；三者各自online及geometry EMA共六个模型均在同一公共池预测，共180K次模型—动作打分。零训练、零新动作生成、零环境控制、零optimizer更新；没有只挑本模型表现好的行为池。

|冻结geometry EMA模型|公共池MAE（mm）|38个真实危险动作中mean判安全|其中两个head都判安全|17个前3步危险动作中mean判安全|
|---|---:|---:|---:|---:|
|P5|0.338|2|1|2|
|Metric|0.375|4|2|3|
|Iso|0.355|5|1|4|

P5自身最后10K包含13个危险动作，其中前3步6个；终点EMA模型对这些危险动作没有mean假安全，整体MAE0.303mm。**这是终点模型读取已收集/可能已学习的数据，不能说违规发生时模型就知道危险。** 其他两种行为池的P5误差也纳入上表，因此不支持直接把当前成功率落后归因于更差的数值margin。2/38仍有漏判；训练分布误差不能当作新DEV覆盖或独立校准保证，ensemble差异不是概率。

已有整段DEV的直接失败模式：P5 zero_noise在nominal/radius065/execution085分别4/2/4次违规，其中前3步3/2/3次，合计8/10；Metric同模式2/1/1次违规。全部无timeout/stall。这说明值得检查早期路径安全与有限动作终点，而不应继续延长horizon。这里没有在相同状态下比较动作的真几何梯度，不能宣称已经确定因果。

下一项有限单因素实验另行登记：完整保留原P5方向损失、共享扩散/能量和EMA，增加同一个已有margin模型对最终生成动作的正余量残差。它测试“局部方向约束之外，显式终点余量惩罚是否改善任务—风险关系”，不是新critic、温度扫描、loss-only taper重复或在线oracle投影。当前诊断不证明候选会成功；完整对照终点、timeout/stall、训练风险和以后多seed必须实测。

全量输出：[108个模型/行为池/分层统计](safereach_common_calibration_20261119/summary.csv)、[全部38个真实危险动作的六模型预测](safereach_common_calibration_20261119/all_actual_unsafe_predictions.csv)、[先验诊断登记](safereach_common_calibration_20261119/registration.json)、[原生来源SHA与完成收据](safereach_common_calibration_20261119/receipt.json)。大型checkpoint仍在数据盘，没有下载本地或上传GitHub。
