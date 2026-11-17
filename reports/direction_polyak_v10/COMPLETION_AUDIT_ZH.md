# 连续调优与冻结确认完成核对

当前目标是获得同时兼顾任务与训练风险的候选，达到原DEV联合区域后冻结并完成新seed确认。
不将目标改成“每项指标都严格第一”，也不将有限两seed证据夸大为普遍保证。
本轮已完成该流程，结果支持保留冻结P5；独立任务结果接近对照，训练风险改善，全部局限如实报告。

| 原要求 | 完成证据 | 结论 |
|---|---|---|
| 单Flow Actor、原critic、Endpoint-Pullback、Normal-shrink；不另起方法体系 | 冻结配置的selected_candidate、共享learner及deployment；科学源码019833be | 保留；无v8/v9追加机制 |
| 完整开发结果后选择，不删失败、不挑较好中间节点 | probe全部6条、各250K结果与REVIEW_ZH.md，历史v7/v8/v9引用哈希 | 全部保留，P5原规则入选 |
| 保持原联合目标及超时≤6规则 | COMPLETION_AUDIT.json逐seed核对 | P5两个新seed均满足最低联合目标 |
| 达标后停止参数搜索，先冻结方法 | frozen.json、方法摘要及pretraining registration | 冻结早于新训练与独立评测 |
| 三方法×seed2031/2032×从头250K | 六条run_config、receipt及queue exit0记录 | 150万步全部完成，无历史前缀冒充新训练 |
| 同seed网络初始状态相同，三臂同部署规则 | manifest初始网络SHA、config deployment | 两seed均核对一致，全部sole target Actor |
| 50K/100K/150K/200K/250K完整曲线 | 全30个DEV文件、2700回合、1500个训练进度记录 | 全部保留含早期负面结果 |
| 报告0—250K和100K—250K训练代价 | training_intervals.csv及完整receipt | 两窗口分报，无预热风险隐藏 |
| 高风险几何诊断 | progress中的既有top20%权重诊断、FINAL_INTERPRETATION_ZH.md | 分两seed报告，不冒充最终独立状态分析 |
| 六模型固定250K身份先冻结，再一次独立评测 | policy_freeze.json、配置摘要、六checkpoint SHA256 | 测试前冻结，测试后六文件哈希未变 |
| 每模型每条件100回合、保留三条件 | 独立6文件、receipt、1800逐回合CSV | 全部完成，仅100个不同配对布局 |
| 不因独立分数重选方法、换seed或调参 | 原配置未改、分析只读、最终报告保留两seed | 无重选、补测或追加训练 |
| 不科学早停；工程失败不自动重试 | 六原生exit0及receipt250K，队列记录 | 无早停、失败或重试 |
| GPU共享、不停止其他项目，数据盘≥40GiB保留 | 队列2槽、运行记录、最终磁盘约153GiB | 本项目2.57GiB；无其他项目进程信号 |
| 本地归档，GitHub和邮件暂停，不用5080/实机 | 本地镜像、无push/PR/send操作、全部计算路径4080数据盘 | 本地完成，未对外发布 |
| 合理收尾而非永久监控 | 队列completed/pending0，观察器与收集器退出，训练/评测PID均结束 | 没有剩余本轮训练或周期检查 |

自动核对记录见COMPLETION_AUDIT.json。该核对只读已完成输入，没有运行额外科学训练或仿真。
程序改动此前的两项针对性CPU检查通过，不重复完整测试套件。

结果边界：只有两个新训练seed；P5独立587/600与Iso相同、略低于Metric-only589/600；
P5并非所有条件第一，没有预注册统计非劣/等价界限，不能称统计显著、统计等价或实机安全保证。
整体方案包含方向机制与Actor LR调度差异；当前确认不能单独分离方向损失、协方差或EMA的因果贡献。
