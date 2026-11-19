# 小型定向诊断：数值margin准确是否意味着安全梯度方向正确

已有公共30K回顾性校准显示P5 margin数值误差并不比Metric大；终点正余量候选却在前100K违规394、未达331.5。因此不再追加惩罚系数扫描，先区分“数值拟合”和“动作方向梯度”的证据。该诊断不训练、不修改正在运行的代码或模型。

冻结原登记的P5/Iso/Metric三个seed2034完整250K检查点（沿用部署协议逐文件SHA，非当前残差候选100K）。公共池仍是各自240001–250000全部真实transition。固定保留其中全部38个危险动作，再从每种行为池真实安全边界[-5mm,0]抽10、内部<-5mm抽20，独立NumPy RNG993100；总128状态，不按任何模型误差/预测方向/成功筛选。三个geometry EMA模型均在同128个实际动作比较。

先用原25D预动作obs的q/dq、goal、obstacle、radius、remaining_time重建原生仿真状态。q/dq来自已保存float32观测；控制器下一真实控制会重置motor目标，隐藏warm-start并不假装bitwise精确。全部128原动作必须先重放原interval-min margin、最大误差≤2e-6m，重建obs误差≤2e-6；任一不满足则停止，记录失败收据，**不放宽容差、不丢状态、不继续导数探测**。

只有全部通过后，每动作三个轴各±0.002 normalized-action有限差分（边界按真实clip宽度），同一物理快照恢复，最后再重放原动作检查标签变化≤2e-7m。总上限1024个控制区间、20,480物理substeps，另明确计入一次初始化reset20 settling substeps；不产生新训练transition，不导入actor、不生成行为动作或进行在线oracle动作选择。

真实梯度norm≤1e-7作为flat保留，不计算或捏造方向cosine；其余报告signed cosine、axis cosine平方（扩散轴不敏感于符号、方向损失敏感）、反向数量、head agreement。按全体/真实危险/真实边界±5mm/安全内部/head agreement≥0.8固定分层；最后一种只是诊断，不是策略门控。数值margin和真实flat比例并列呈现。

这仍是历史训练分布，可能in-sample；不是当前残差候选的因果归因、不是当前DEV访问状态、不是sampler deterministic-tail pullback梯度，也不是独立校准。即使发现坏方向，也不能直接证明某个新方法更优；必须后续单独登记并实测。A100 CPU运行，原GPU父训练不变，storage wrapper/data root和40GiB+1GiB预留保持。无5080、无实机、无大文件本地下载、无新增训练队列。
