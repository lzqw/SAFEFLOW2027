# Piper：标准型号已确认，工程续训通过，候选队列已准备

## 本轮实际完成的范围

官方模型来源固定为 [`agilexrobotics/piper_ros`，ac41fcbc](https://github.com/agilexrobotics/piper_ros/tree/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc)，
SDK 源固定为 [`pyAgxArm`，d90e234b](https://github.com/agilexrobotics/pyAgxArm/tree/d90e234ba8e408cfa964b38aa9120b3131fc9f85)。
使用官方标准夹爪 URDF `src/piper_description/urdf/piper_description.urdf`，
实际下载校验源 SHA256 为
`c1a5c29ae9b91160557ed02bddd47365a71f0eb1653abb4997eb34f124021486`。
mesh 和完整官方仓库在外部缓存，不上传大文件。仿真只解析 package URI，不修改官方几何。

已实现并实际运行官方六关节 Piper 的 PyBullet DIRECT **Reach 与 Payload**。
Payload 开始时以固定约束持有 50g 轻质盒体，固定姿态、三维末端位移控制；
每个控制区间20个2ms子步全部检查 arm–obstacle 和 payload–obstacle 最小距离。
允许的夹持接触单独处理，障碍碰撞检查不豁免。34维共享观测包含关节、TCP、目标、
障碍、剩余时间、载荷位姿、尺寸和质量；不是把 Panda 模型或权重改名。

两任务最初各做3个零动作 smoke 控制区间：共6个新模拟控制区间、120个物理子步。
随后补齐每次reset显式恢复手指位置，各任务只重确认1个控制区间；旧凭证保留。
**总计8个新模拟控制区间、160个物理子步，0训练步、0真实交互、0生成的学习策略**。
初次smoke最小区间距离分别为 Reach34.429mm、
Payload33.597mm，均未违规，也均未成功完成任务。这不是策略性能评测。
初始 TCP IK 误差15.327mm被原样记录；不声称已完成真实TCP标定或高精度部署。
官方 URDF 固定底座 dummy_link 无惯性数据，PyBullet 有非致命warning。
实际本地环境Python3.11.5、NumPy1.26.4、PyBullet3.2.7、Gymnasium0.28.1；
这些原始记录只覆盖CPU模型smoke，当时未启动PyTorch策略训练或GPU性能测试；
后续原生工程训练单独列在下文，不把旧记录改写为训练性能。

后续训练接入没有把零动作smoke当作可学习性证明：静态目标检查发现基础200次IK迭代
在跨工作空间目标上收敛不足。独立训练wrapper使用同一真实IK求解器的1200次上限、
所有方法共同的安全可达端点reset筛选，并以真实载荷质心定义Payload完成。
[构造性可行路线记录](piper-constructive-feasibility-v2.json) 中 Reach、Payload 各58个
真实控制区间成功绕行，最小距离44.6mm、0违规；另外116个控制区间/2320子步由该
独立探测计账，不重算进上述8个准备smoke，也不是训练策略结果。
基础仿真CLI只用于模型工程smoke，训练必须使用 `PiperTrainingEnvironment`。

离线接口使用 **robot-base坐标，位置m、角度rad、时间s、四元数xyzw**。
SDK 的单位转换只做纯算术：位置1单位=0.001mm，角度1单位=0.001度，依据
[固定SDK源说明](https://github.com/agilexrobotics/pyAgxArm/blob/d90e234ba8e408cfa964b38aa9120b3131fc9f85/pyAgxArm/protocols/can_protocol/msgs/piper/default/transmit/arm_end_pose_ctrl.py)。
没有SDK导入、CAN连接或真实指令发送。

已实现 dry-run、逐条记录、只读回放、观测超时/时间倒退/推理deadline拒绝、
速度和工作空间修正，以及原始动作与实际执行动作的区分。实际synthetic dry-run
为3条记录：1个可接受离线提案、2个被拒绝提案（含1次deadline miss），没有执行动作。
4个聚焦契约测试通过，0失败；不重跑旧历史测试。

`bundle.py` 提供 native Piper 策略导出/完整性检查/冻结推理：Actor、EMA margin、
normalizer、动作缩放、采样方差、版本和训练凭证一同保存。几何保留输入梯度，
不使用全局 inference mode。拒绝 Panda／未训练来源和伪造 Iso-Low 标签。
**当前没有已训练的 Piper Iso／Iso-Low／EA候选，导出接口不等于候选已生成。**

另一个已明确批准的原生CPU工程实验在固定源码 `70269ac` 上完成 Payload Iso-Low
3210→3220步；Replay、优化器、RNG、夹爪/关节/载荷状态和部分回合真实恢复。
实际方差0.01、能量目标75、初始alpha0.0002同步匹配，不是把Iso标签改成Iso-Low。
这3220训练步计入Piper的3M预算；工程策略不是部署候选，未消费独立测试、未导出假候选。
完整检查点已存4080的4T硬盘且本地/远端校验和一致。详见
[原生训练实现与完整计数](PIPER_TRAINING_IMPLEMENTATION.md)、
[工程凭证](piper-engineering-70269ac/ENGINEERING_ACCOUNTING.json)。

## 标准型号已由用户确认，仍需现场标定信息

用户随后明确确认“piper标准版，夹爪也是标准的”。据此选用已固定的官方标准型号
`piper_description.urdf`，不再把型号／夹爪身份记为未知。
`deployment/piper/hardware_pending.json` 仍保留待填项目：实测开口、固件、TCP与底座标定、
真实控制周期、负载/速度/工作空间限制、急停和操作者确认。
型号对应不等于实体几何／控制标定已验证；当前默认TCP和动态参数仍是仿真fixture。
机器记录中的物理匹配未验证，不应再被读作“用户没有说明型号／夹爪”；两项身份均已确认。
仿真示例0.04s周期、0.01m动作缩放等是软件fixture，不能直接用于实体。

型号已确认，因此可在CRAX安装完成且共享槽可用时直接推进标准Piper原生仿真训练，
不必等待实体运动授权；Piper仿真预算3M。后续TCP标定变化仍需更新独立版本。
相同候选数、checkpoint数及验证规则用于Iso、Iso-Low、EA。
冻结所有参数和normalizer，在未见条件评价；不转换旧Panda权重冒充Piper训练。
候选、完整state/replay和部署bundle分开保存；大文件遵守4080 4T策略。

## 已注册的有限仿真候选队列，尚未启动

`configs/paper_crax_payload/piper_queue_4080_v1.json` 注册
Reach/Payload × Iso/Iso-Low/EA，共同seed41027，每项300k、合计1.8M。
加已发生工程3220步，总训练预算预留1,803,220，低于Piper总3M；剩余额度不是自动续训授权。
所有方法有相同五个checkpoint机会和共同验证规则，独立71027测试保持未使用。

只有两个训练槽都释放时使用合并双槽队列；槽分别释放时，可依次启用互不重复的
`piper_queue_4080_slot0_v1.json` / `slot1_v1.json` 单槽分区，每个只含三项900k。
这两种启动方式互斥，不能把同六项重复运行。Payload待运行项目始终优先；没有Payload
待派任务后，核实真正释放的槽即可补Piper，不要求等待另一条仍占槽的Payload结束。
Piper＋Payload总原生训练槽不超过两个，其他项目不暂停，不用5080。

截至2026-11-12 13:53的实际检查，4080仍有两条Payload训练、两项待派，GPU利用率99%。
因此没有启动Piper候选，不把“配置就绪”写成“运行中”。协调者还须在实际启动时重新
核对原生PID/资源/输出路径及准确源码版本；这里不新建多层watchdog或故障自动重启。

官方固定模型现已预取到4080的4T缓存，源URDF和mesh校验仍与固定版本一致；
[4080模型凭证](piper-assets-4080-ac41fcb.json) 保留实际绝对路径及解析后digest。
预取未导入机器人SDK、未连接硬件、未执行仿真步，也没有改变正在运行的Payload源码。

## 实体实验只生成协议，不启动

`python -m deployment.piper.protocol` 生成独立seed72027的可复现顺序：
Reach/Payload × Iso/Iso-Low/EA × Nominal/新几何/新执行或载荷 ×20回合，
共360个**计划**回合；每个任务/条件/重复块内交错方法，不按方法整批执行。
当前真实执行回合数为0。

所有方法使用同一保护设置，记录 safe success、无介入safe success、违规/碰撞、
保护介入率、动作修正幅度、最小距离、capped时间、p50/p95推理和deadline miss。
保护层拒绝的动作不能视为已执行数据；原始提案与实测执行分开存储。
Reach≥90%、Payload≥85%及干预减少约20%仍是努力目标，不是现有结果。
零观测碰撞不等于理论零风险。

若论文保留适应效率主张，协议保留单任务共同预训练起点、三分支各50个真实训练回合、
尽量两次独立重复、0/10/25/50独立评估与冻结参照。**没有启动真实微调**。
本包物理transport函数始终抛错；实际运动需要现场操作者授权及另行评审实现，
不能因填写配置或本报告完成就自动启用。

机器可读的实际状态/成本见 `deployment/piper/preparation_receipt.json`；
使用命令与组件见 `deployment/piper/README.md`。本项不阻塞CRAX/Payload主训练。
