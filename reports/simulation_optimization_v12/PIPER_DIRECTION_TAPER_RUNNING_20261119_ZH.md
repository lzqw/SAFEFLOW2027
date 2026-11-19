# A100原生Piper两组Reach已实际运行，Payload两组待接续

2026-11-19约11:57（UTC+8），A100有限队列PID3829716，原生Reach control PID3829730、候选PID3829732都已实际跑完新增3000/100000 transition；不是仅提交队列。后面只有已登记的两条Payload同硬件control/candidate，没有扩张其他扫描。

源码`d904b9c8`；两个训练进程均绑定物理GPU1、通过storage wrapper继承数据盘cache/TMP设置，allocator fraction .025。GPU0其他项目保持运行，未发送信号或调整其资源。4080现有两条SafeReach原生进程也仍活跃，GPU整卡99%（含共享项目），其待启动候选不变。

新增3000这个很早的窗口，Reach control违规3/安全成功95，候选违规10/成功94，两者训练timeout/stall均0。候选此时风险更高，不能因为离线梯度冲突减弱就预告任务更好；保留完整100K与五DEV节点后判定，不科学早停、不挑最好中间点、不自动重试。

实际全父CPU装载/本机原生34D动作等价/dual保持/严格恢复与错臂拒绝测试已通过，2 passed；没有启动额外工程训练或仿真smoke。单个父seed41027仍只是开发，不代表独立训练seed或最终确认。

数据存`/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12/runs-piper-direction-taper/`，约55.6GiB可用时启动，按同类实际存档四组预留4GiB，预计结束≥51GiB。启动门槛52GiB及原checkpoint50GiB保护均保留，不回退home，不删除额外文件。

每30分钟资源/原生状态监测已启动，目录`half-hour-monitor-piper-direction-taper/`，日志`logs/monitor-piper-direction-taper.log`。另有有限只读收尾程序：只有四条原生100K、所有100个训练节点、全部五DEV节点/条件/100原布局齐全，才自动生成`piper-direction-taper-analysis/`。不执行新独立test，不选winner，不写GitHub；后续收回紧凑结果发布并本地镜像。

预计这批四组约2–4小时，按当前两条早期约48–50 steps/s及同类完整运行经验给范围，实际包含评测与共享负载波动。SafeReach和Piper是并行研究线；任一单项胜出都不等于所有任务/几何或Panda回归完成。
