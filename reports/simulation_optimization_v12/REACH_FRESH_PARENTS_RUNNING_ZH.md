# 2026-11-19约14:25（UTC+8）：独立300K父模型已实际训练

有限队列4014800已启动两个原生进程：seed41028 PID4014810、seed41029 PID4014812，源码`22d4966f`。两个都已实际保存5000/300000训练步，不是仅准备配置；随机前缀分别违规35/16、超时43/44、安全成功均0，是新鲜早期训练，不据此评判最终方法优劣。

两条均从独立网络/optimizer/replay/dual/环境RNG开始，原生manifest登记`fresh_Piper_no_Panda_transfer`、initial_steps0、父路径为空；不是导入41027模型后重标seed。wrapper生成的逐seed精确配置在数据盘`fresh-parent-configurations/`，登记SHA、来源、cuda RNG index0。原始300K参数/2env/batch256/UTD1保持；仅seed和开发评测随机流改变，三项针对性配置/收据字段/槽分配测试通过。

父训练输出`runs-reach-fresh-parents/Reach-EA-41028/`与`Reach-EA-41029/`，source/cache/TMP/所有checkpoint写storage0。启动前52.6GiB可用，两条按实测历史目录保守预留2GiB；不删证据、不写home、不启动完整1.7M方法确认。GPU1共享两个allocator .025槽，GPU0的4200回合冻结部署评测继续，4080候选也不变。

每30分钟只读监测已启动，状态目录`half-hour-monitor-reach-fresh-parents/`，日志`logs/monitor-reach-fresh-parents.log`；原生进程/完成收据是是否完成的依据，不凭锁文件或队列字样。无自动重试/科学早停/下游分支自动启动。预计父模型这两条约8–16小时，待真实完整更新速度重新估算，不用最初随机前缀的40steps/s外推。

这只是独立训练基础，不能宣称已完成跨seed的P5/Metric对比、困难Reach/Panda/训练风险共同目标或独立测试。后续100K扩域和三臂150K确认必须各自登记本seed真实父SHA，不照搬旧双GPU父RNG位置或旧配置guard。此前“尚未启动”准备文件保留为历史，总入口以本记录为准。
