# Panda补充匹配参考，不宣称新候选已通过回归

原v11 Panda Payload四条件96–100%和训练违规−56.5%是P5相对EA control，不是显式Metric-only。这次仅补完整原生45D Panda参考，固定P5与Metric-only各从同一240K原生完整父状态新增100K。不是围绕Panda调参，不导入任何Piper网络。

父模型`payload-ea-41028-v1/full_240000.pt`，SHA256 `628b5be167895c22917a5fac9fd81d57ba2a89feb89e8e7d35d7d7c2d5caef2d`，训练seed41028。原60/67.5/75mm训练半径族、15mm真实安全距离、100步horizon、35mm payload质心成功阈值、原成本/奖励/更新次数保持一致。继承网络、优化器、replay、alpha/dual、RNG并同样重置/censor父部分回合。

固定P5沿用NormalShrink+方向项1，不加新激活/预热；Metric-only采用各向同性采样+方向能量、无方向损失，与其他任务显式Metric定义一致。Panda原生PayloadFlowLearner更新及45D诊断保留，不将PiperLearner改名成Panda。

新的DEV布局11880000起，动作11881000+条件索引×1000+episode，各条件100回合；nominal、70mm、wide payload、execution085，20/40/60/80/100K全节点。使用已有target Actor Polyak.001部署，评测前后RNG隔离；记录成功、普通终端任务成功（包括同时违规）、违规、超时、stall、return、完成秒数与单独action推理耗时。普通成功不允许违规后继续行动；stall按真实payload质心到目标10控制步净进展<1mm定义。

这两条只是未来新候选回归的同期方法参考，完成不等于新候选的Panda回归通过。之后明确选择的最终新配置仍需同条件≤2pp回归、两个训练seed确认；新独立集冻结与三seed主候选稳定性也不能省略。新DEV不与旧独立96–100%直接做变化差值。

队列4080有限两槽，等待E4两完整100K收据再运行，预计新增200K。所有大型存档4T数据盘，无新/home训练目录，不抢占/终止外部项目。主Piper及E3/E4未完成任务继续，Panda不提前占用额外槽。代码快照独立，新增可选Payload evaluator钩子的旧默认不变，不热改其他运行源。

若原生父缺失或身份不同则拒绝运行，不用Piper或其他seed权重代替。本轮不接机器人、不使用5080、不宣称物理响应gate通过（用户已明确豁免）。
