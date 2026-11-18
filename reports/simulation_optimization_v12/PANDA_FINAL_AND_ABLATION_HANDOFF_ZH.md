# Panda显式Metric参考完整完成；仅转移未启动的消融任务

两条同原生EA240K完整父、seed41028、新增100K均已原生退出并有完整收据。新DEV每条件100回合，不打开v11独立集，不选择最好中间节点。

|方案|训练违规|训练成功|超时/stall|累计return|
|---|---:|---:|---:|---:|
|Panda固定P5|184|5143|0/0|57236.02|
|Panda显式Metric-only|446|5200|0/0|55254.77|

P5训练违规少58.7%，累计return高3.6%，但累计安全成功少1.1%；不能声称每项严格占优。

|最终DEV|P5安全成功/100|Metric安全成功/100|P5/Metric封顶完成秒数|
|---|---:|---:|---:|
|nominal|100|100|0.774/0.751|
|70mm|100|97|0.766/0.842|
|wide payload|96|96|0.900/0.879|
|execution085|100|100|0.898/0.866|

最终DEV均无超时/stall，失败均违规。P5合计396/400、Metric393/400（99.0%/98.25%）；宽载荷都96%，并非所有几何更高。除70mm外P5稍慢，nominal/wide/execution的return也略低，不能用平均优势掩盖。这里是单训练seed开发结果，不是最终新候选≤2pp回归验证，更不是多seed独立结论。

## 复用释放的4080实验槽，避免待机与串行等待

Panda两条训练已正常结束，4080项目槽释放后，仅将A100队列中尚未启动的A-no-lateLR-2034、A-Metric-matchedLR-2034转移到4080。A100已运行的A-full-P5-2034(PID786854)、A-no-direction-2034(PID786855)原样继续，未发训练信号、未重放步数。四组总预算不变，仍各fresh250K。

确认原队列controller786815命令、pending2与两个原生训练身份后，仅冻结/终止该controller以防待启动任务重复；两个原生子训练没有信号。保存queue_matched_ablation_before_handoff.json及matched_ablation_handoff_registration.json；A100 read-only adopted observer现在检查原生PID+启动ticks和250K收据。旧queue_matched_ablation.json为历史快照，不再作为当前状态，也不允许重启旧queue。

4080新队列PID692611，原生训练PID692613/692615；同科学source2f09231b快照和相同配置/seed，已看到真实首批progress。仅路径/allocator cap(.025→.04)/GPU硬件不同，原始指标保留。GPU数值内核差异可能导致浮点和学习路径差异；前2000随机前缀相同不表示GPU更新bitwise一致。本轮不将跨GPU差别冒充严格因果或跨seed稳定性，解释完整曲线时保留该限制；若发现结论依赖微小差异，后续固定硬件配对确认，不悄悄重跑选择结果。

各机有限两槽、共享GPU，不暂停外部项目；全部大型checkpoint/轨迹在A100storage0与4080的4T盘。半小时只读监测已转向正确队列，失败不自动重试。总计26个完整100K端点，新的四条250K消融尚未完成，不计入完成表。

后续仍应检验方向项、lateLR机制，冻结新版本后多seed/独立测试；Reach训练风险+任务联合目标、Payload困难几何、初期始终领先和新Panda回归尚未全部解决，不能标记总体完成。
