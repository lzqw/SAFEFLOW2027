# 资源接续v4b：只转移尚未开始的任务

2026-11-14，原六臂科学配置、训练seed2027、源码b31878c0、各300K终点和
开发／独立布局均保持不变。此变更只重新安排机器和共享槽释放顺序，不增加科学预算。

原安排两个参数任务都等待A100当前臂结束，而4080完成Metric-only后无后续任务。
新安排为4080原Metric-only完成真实300K收据后，顺序接续rho0.1、rho0.5，始终一共享槽。
A100 Full／Noise-only原worker不中断；Full完成后释放原GPU0槽给独立CVPO队列，
不再等待已搬到4080的rho0.1。CVPO训练／评测／选择规则、三个seed、30M预算和输出身份不变。

前提是两个参数任务从未开始、CVPO没有训练worker；控制器交接前逐一核对原生PID／命令／状态。
只退役A100两个组件调度器和空等CVPO调度器，不给训练worker或其他项目发信号，
不使用进程组kill。旧状态在A100数据盘`resource-handoff-v4b/`保留。
原A100组件queue文件成为退役控制器的历史快照，不用于推断剩余任务位置。

4080新参数队列配置为`component_pending_4080_v4b.json`；
A100新资源依赖配置为`cvpo_queue_a100_v4b.json`。
4080参数worker仍从原不可变source目录执行；单独的新控制器仅负责收据依赖和有限队列。
CVPO继续从原`formal-source-v1`执行，正式源码7abb6ebb不变。
盘余量40GiB（CVPO启动56GiB）与显存门槛、allocator上限、共享GPU、不科学早停及不自动重试均保留。

原本地收集器替换时保留原输出及已收Iso，新增`--pending-parameters-on4080`只改变
两个尚未启动参数臂的读取位置；六臂完整性、300K收据、source与初始网络SHA检查不变。
已有自动收集结果及所有原生训练不被删除或重跑。

机器差异披露：Iso／Metric-only／两个参数点在4080，Full／Noise-only在A100。
CPU/GPU数值和共享负载不是受控的跨机器吞吐对照；不得将实际用时差直接归为纯算法收益。

## 已实际部署与核对

执行前登记提交effa8d4a。17:02 UTC完成[只退役调度器的交接收据](resource_handoff/receipt.json)，
17:05 UTC再次读取`ps`确认以下句柄在关闭SSH启动连接后仍存活：

| 对象 | 原生PID | 状态 |
|---|---:|---|
| A100 Full训练 |1825421|原进程保留，未发信号|
| A100 Noise-only训练 |1825547|原进程保留，未发信号|
| 4080 Metric-only训练 |3348196|原进程保留，未发信号|
| 4080参数调度器 |3377002|[pending2，等待Metric-only收据](resource_handoff/parameter-queue.json)|
| A100 CVPO调度器 |1842989|[pending4，等待Full收据](resource_handoff/cvpo-queue.json)|
| 本地只读收集器 |673280|保留原收集目录及Iso，使用新位置读取参数臂|

收据的`replacement_controllers_not_yet_launched=true`是17:02第一阶段状态，
随后两个新调度器已启动并经上述实际进程核对，不表示当前仍未启动。
退役的只有调度器1825419、1825545、1831775及旧本地收集器646574；
没有中断或重启任何科学训练，没有停止其他项目，没有删除原结果。
原CVPO队列状态已完整移到`resource-handoff-v4b/formal-queue-v1.retired.json`，
新调度器继续写原稳定state路径，既有观察器1831776未受影响。

最后原收集器实测为16:56:22 UTC：Full／Noise-only125K、Metric-only75K，
三worker命令匹配、errors为空；[完整清单](runtime/check_20261114_1656/collector-status.json)保留。
新收集器首次读取17:26:16 UTC，此后每1800秒；不是又完成一轮远程采样的声明。
数据盘实际剩余A10087GB、4080643GB，均高于本次接续门槛。
未来训练任务仍未开始，不能把调度接纳当作参数实验或CVPO完成。
