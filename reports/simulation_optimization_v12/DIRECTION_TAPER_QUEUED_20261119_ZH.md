# 2026-11-19 11:23（UTC+8）：候选已排队，尚未训练

单候选方向损失taper已完成实现并部署4080。正式环境两项针对性测试通过（2 passed）：原sampler/能量/全部旧trace/RNG精确保持，损失端点与反向有限，参数/评测预算与原250K一致、checkpoint拒绝改解释、完整分析依赖不能以工程smoke代替。

队列PID3915728，状态`pending=1`、`jobs=[]`、`live_native_pids=[]`，原生原因`waiting_for_registered_predecessor_receipts;no_extra_GPU_slot`。这是正常依赖等待，不能写成候选已跑或已有收益。源码`61bd30f0`，单槽，不改正在运行的两条P5/Iso或其源码；两条原250K收据与严格同硬件分析收据全部到齐才自动接续。

候选队列路径`/home/carla/4T/SAFEFLOW2027_SIMULATION_V12/queue_direction_taper.json`，半小时监测`half-hour-monitor-direction-taper/`，日志`logs/monitor-direction-taper.log`。已有原对照的半小时资源监测和完整分析进程继续。没有自动改权重、重试失败任务或改变样本定义的后台逻辑。

本次A100只运行了两次小型CPU离线分析：原方向损失与候选方向损失，同四个checkpoint/同四批replay，共各4096个模型状态，无环境控制或优化器更新。结果原始逐批JSON、摘要CSV及checkpoint哈希已收回本地并上传结果分支。不能把这些重复状态当8192条新真实transition或两个训练seed。

当前80K窗口P5违规252对Iso268、成功1534对1799、timeout354对378、stall493对497。P5任务量已经落后，明确修正50K的乐观早期片段，不声称始终共同领先；仍待完整100K与250K。

新候选是直接检验“深安全区额外方向约束的任务代价”，尚不代表Piper优化或Panda回归通过。若丢失后期安全优势或以更多timeout/stall降低违规，完整保留结果、不晋升。全部目标仍未完成。
