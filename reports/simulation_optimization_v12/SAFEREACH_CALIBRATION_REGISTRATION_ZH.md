# SafeReach失败后：固定原始transition上的margin诊断

只使用既有登记的SafeReach P5、Iso、Metric三个250K固定终点，检查点SHA与来源沿用`simulation_v12_panda_safereach_deployment_modes.json`，不重新挑checkpoint、不接触独立test。三者各贡献保存replay的完整最后10K真实transition（240001–250000），共30K；六个原生模型（各自online及geometry EMA）分别在全部30K公共state/action池预测，避免只用各自访问状态比较误差。

固定报告全体、真实危险margin>0、真实边界±5mm、安全内部margin<−5mm、每episode前3控制及其危险子集。预动作obs最后一维1/.99/.98表示前3控制；不是使用next_obs，也不是用预测margin筛选危险样本。记录所有真实危险transition的两个head预测，不因错判而丢弃。

动机来自已完成的新DEV整段轨迹：P5 zero_noise三个条件共10次违规，其中8次在前3步，全部无timeout/stall；Metric同模式共4次违规。先检查原生margin数值是否漏判真实危险，再决定新机制，不重做已经失败的温度、方向taper或warm-up扫描。

这是**回顾性训练数据诊断**，自己的训练池可能in-sample，公共池也不是独立集。不能把ensemble方差解释为校准概率；数值误差也不证明动作梯度/最终成功率因果。不会用这些诊断标签在线选择动作，不产生新动作、环境控制、optimizer更新或新critic。既有margin模块严格按原生state_dict加载，不改变任何训练。

在A100数据盘运行，遵守storage wrapper及40GiB+1GiB余量；GPU0共享小allocator，既有GPU1父训练不改变。输出是紧凑CSV/来源收据，大检查点复用已有数据盘副本；不新增checkpoint拷贝，不使用5080或实体机器人。完成后与原完整DEV配对失败记录一起解释，不把局部诊断算为新方法通过。
