# Piper方向冲突诊断：检验能否从SafeReach迁移解释

固定此前终止噪声诊断登记的四个原生100K完整端点：Reach固定+2mm、Reach half75K+2mm、Payload固定、Payload+4mm。沿用已冻结checkpoint SHA、严格原生loader，不挑最好checkpoint，不导入其他机器人的权重。两任务均为Piper34D，父训练seed仍41027，不虚构多seed。

每端点四批256保存replay状态：半均匀、半最近10K，索引RNG992700+batch，Actor噪声992800+batch。仅CPU推理/微分，零环境实例/控制、零优化器更新、零新训练数据、零checkpoint回写。原loss与taper loss共享完全相同生成动作/速度/几何，只改变额外方向损失权重；动作不由诊断结果选择。

记录原方向项、原Actor基础项（task/cost/energy）与任务梯度关系，taper的一阶影响，以及**每个flow注入点**EMA预测margin≤−10mm样本及损失占比。这里包含每个历史模型原登记的激活offset，明确是模型风险而非真实距离；Piper replay来自训练45/50/55mm，不能冒称是57.5/60mm外推失败状态的梯度。

Piper原程序使用批量head-VJP `UpdateGeometry`，增加与普通EndpointGeometry相同的已有margin诊断字段，以便直接记录、避免额外网络前向。sampling/energy/法向梯度计算不改。小型等价测试覆盖普通25D与此原生34D批量几何路径；它不是所有原生Piper任务收益已通过的证明。

如果Piper原方向项也存在深安全区任务冲突，才支持准备原生同父、同预算的loss-only taper候选；如不支持，则不直接推广SafeReach解释。该分析不自动排队新的Piper训练，不改变已有SafeReach有限队列，不增加A100大型存档、不使用5080/实机。全部结果写4080 4T数据盘，预留40GiB+1GiB增长余量。
