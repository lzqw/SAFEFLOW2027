# 两个独立原生父模型300K：仅共用训练基础，不晋升候选

此前准备记录要求先完成Payload与SafeReach候选再决定整套确认。现在Payload/Reach loss-only候选已完整失败，SafeReach25D候选仍运行；A100 GPU1空闲。独立父模型的生成不依赖哪种P5适应方案胜出，因此提前只执行共同基础两条300K，避免后续串行等待。并未提前启动原计划每seed后续400K扩域父或三臂150K确认，也未将P5+2mm冻结为全部任务最终配置。

固定原生Piper EA Reach，seed41028、41029，各从全新网络/所有optimizer/replay/dual/环境RNG开始训练300K，总新增600K。不导入任何旧父，不只改适应阶段随机种子。原始`piper_sim_v1.json`SHA256锁定84ff1e94…；每seed仅改`development_seed`及新DEV布局11990000/动作11991000，其他网络/目标、2env、batch256、原UTD1/两reward更新、原critic预热、方差/安全距离/奖励/horizon/终止均完全不改。不是沿用当前8env/batch1024的适应执行预算去假装复现原300K父。

通过独立wrapper生成每seed精确resolved配置交原生`piper_train.train`，不全局放宽旧resume/parent loader，也不增加原生runner算法开关。两个seed共用同GPU1/同Python和源码，独立训练RNG；每进程仅一个可见GPU，实际保存cuda RNG index0，后续不能照抄旧双GPU父index1。完整300K预算、五原生DEV节点60/120/180/240/300K与所有定期checkpoint保留，固定300K父端点，不按最好的DEV选模型或重启失败seed。

仅执行原生DEV评测各条件30布局，不调用旧validation/independent_test，即使旧配置保留那些字段。这里DEV只是父训练跟踪，不宣称已确认P5跨seed/Reach困难几何。所有原生训练违规和前缀成本都计入；600K不能在后续150K适应表里消失。后续400K扩域、原固定P5/+2mm/Metric等三臂确认还需独立协议、各父实际SHA和原生成功完成收据，当前队列没有这些任务。

资源与空间按实际旧300K EA目录467MiB/组核算，两组连新增source/config保守预留2GiB；启动前storage0可用52.6GiB，预计完成≥50.6GiB，保留≥40GiB余量及当前部署评测的小输出增长。队列52GiB启动门槛、wrapper40+2GiB最低余量检查；数据盘若外部大幅变化则不新增任务，不回退home或删证据。所有source/cache/TMP/输出经storage wrapper，仅GPU1共享两槽allocator .025，不修改GPU0正在进行的4200回合部署评测或4080候选。

历史原生EA300K实际耗时29311秒（8.14h），本次两条并行共享一张A100，保守估计8–16小时，不能套用8env短适应的约1小时。后续完整确认时间另算，没有承诺本轮结束就达成全面最优。每30分钟监测、不科学早停、不自动重试、不动外部项目、无5080或实机。
