# 独立半径覆盖两条已实际到301K：原300K父全部保留，新增100K而非重跑

2026-11-19 16:18 UTC（北京时间11月20日00:18），A100有限队列PID93881、源码`10ba914c`已实际运行两条原生EA分布训练；不是仅准备配置。首个原生日志两条均301000，initial_steps300000，各已新增1000真实transition，actor计数由148500至149000（新增500次）。

|seed|native PID|物理GPU|总步数/固定目标|本段已新增|本段新增违规|本段新增安全成功|本段新增超时|
|---|---:|---:|---:|---:|---:|---:|---:|
|41028|93891|0|301K/400K|1000|15|27|0|
|41029|93893|1|301K/400K|1000|21|28|0|

原生run_config明确`initialization=Piper_Reach_finite_radius_adaptation`、initial300K，保留各自真实父SHA（41028 `a1cb1a58…`、41029 `26061e71…`）、seed和既有网络/optimizer/replay/计数；resolved config仅比本seed父增加45/50/55mm radius round-robin字段。初期违规是实际记录，不隐去或据此提前停止；这里不是新的P5/Metric比较或独立测试。

A100两个slot各绑定一个GPU、小allocator .025，共享其他任务不更改。当前首段各10.15/9.95 transitions/s，按该段外推剩余99K约2.7–2.8小时，含保存/评测和共享负载粗估**约3小时**；只有首1K实测，这仍是低置信运行估算，不用GPU显存容量夸大速度。卡瞬时利用率各约20%，小网络/CPU调度与原生未批处理路径会限制吞吐，两个GPU不等于双倍sps。不改活跃科学配方以追求利用率数字，也不重复失败分支填空。

输出`runs-independent-radius/Reach-EA-41028/`和41029、源程序`source-independent-radius/`、resolved配置`independent-radius-configurations/`、log/cache/TMP均在本用户storage0；启动剩余50.96GiB、两条保守预留6GiB，保持40GiB底线，不新增删除或home大文件。7项针对性测试在A100原环境通过；pytest有两项缓存目录不可写警告，没有科学测试失败、未安装额外依赖或重复大规模校验。

半小时资源监测PID94020真实存活，首快照unix1795104993.197797确认两个native_handle_live=true；首快照早于1K所以progress=null，不能误判训练不运行。下一轮约16:46 UTC（北京时间00:46）。既有4080/A100四条旧训练、旧monitor与closure均已正常结束，完整终点已另行上传，不能再报告“原四条仍在运行”。4080暂未新增任务，不使用5080或实机。

这只是原已规划独立训练谱系第二阶段的固定200K有限队列，无自动重试/科学早停/自动下游。完成400K后再单独固定真实父SHA及方法确认配置，不照搬旧41027或index1；原生distribution fork按原seed重新初始化随机流，不是父RNG完整resume。完整要求（前100K安全、全任务几何、Panda护栏、最终多seed/独立test）仍未达到；残差候选不因该新队列而晋升。参见[固定登记](INDEPENDENT_RADIUS_REGISTRATION_ZH.md)与[四条旧训练完整结论](FINAL_RESIDUAL_AND_PARENTS_20261119_ZH.md)。
