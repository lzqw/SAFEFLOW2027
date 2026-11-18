# v12 SafeReach：直接比较真实前100K风险

用户已解除总额度和接触响应gate，允许Piper仿真；本组是另外的Panda SafeReach从头任务，不是Piper400K父状态的适应训练。数据和科学结论分开记录，不能将本组改善当成Piper泛化提升。

为了准确回答“前100K是否低于Iso/Metric或比固定P5少15%”，把原建议的75K粗筛延长到同轮100K，六臂共600K实际步，单开发seed2033从头初始化，未用旧2031/2032测试seed选模。

|分支|唯一方法变化|
|---|---|
|Iso|原native Iso对照|
|Metric-only|原native Metric对照|
|P5-fixed|原P5方向权重固定1|
|E1-critic20k|P5基础上Actor从约20K才更新（随机2K后1更新/真实步，critic_only_updates=18000）|
|E2-quarter50k|方向权重从.25按0–50K线性升1|
|E2-quarter75k|方向权重从.25按0–75K线性升1|

所有臂初始种子相同、原生25D任务、两环境、batch256、UTD1、真实安全距离15mm、horizon100、成功距离35mm保持一致。两次reward更新、四target样本、margin EMA .005、原几何10K–30K ramp不改；保留Normal-shrink和原critic/alpha/dual。P5的Actor LR150K–200K衰减仍保留，但本组100K不会触发，不从本组宣称后期LR机制已经被隔离或改善。

E1的18K learner预热对应随机2K之后的真实20K Actor起点，不排除这期间的违规、任务失败或超时。环境继续采样完整预算；没有科学早停、不自动重试、不将较低训练违规自动当作成功。

25/50/75/100K均固定DEV：nominal60mm、radius65mm、execution085各30回合，布局11840000起、动作seed11841000起；不打开旧v10独立集。Piper57.5/60mm没有混入此任务验收。所有轨迹及终止结果留数据盘，记录成功/违规/超时/return/终端普通成功及10控制净进展<1mm的stall诊断。推理延迟单独计时，不用任务完成秒数替代。

完整100K后先比较同轮控制和所有节点；才选有证据的候选进入150K×2确认及最后三seed，不把六臂全部自动扩大。后期100K–250K优势、最终独立任务成功和Panda Payload回归仍是后续要求，本组完成不代表整个目标达成。E3近期/边界采样以及M系列的单因素任务约束需结合本组证据另登记，避免重复v8/v9负结果。

A100新数据根`/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12`，所有启动经过run-with-storage.sh，两共享槽各绑一张A100，显存上限每进程2.5%。预计增长<8GiB，保留40GiB加增长；不停止外部任务。已有GPU共享满载会使完成时间可变。每30分钟记录进度/资源，队列有限退出。
