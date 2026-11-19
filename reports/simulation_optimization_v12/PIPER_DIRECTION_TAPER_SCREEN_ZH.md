# Piper同硬件loss-only taper筛选：两任务各一对100K

原生Piper四端点保存replay诊断已完成，零训练/物理控制。Reach固定2mm与half75K两设置四批全部任务/基础Actor梯度冲突，深安全flow点原方向损失贡献70.5%/66.6%；Payload固定/4mm同样四批全部冲突，贡献84.2%/71.3%。taper仅减小相同动作上的方向损失，不能据此宣称真实安全成功提高。

这给出独立于SafeReach结果的原生Piper依据，登记同一个单因素而非继续多参数搜索：

|任务|同硬件control|唯一候选|共同激活offset|每组新增训练|
|---|---|---|---:|---:|
|Reach|固定P5+2mm|同设置，仅方向损失taper|2mm|100K|
|Payload|固定P5+4mm|同设置，仅方向损失taper|4mm|100K|

合计四组400K。每任务从原生完整EA父同样fork，网络/所有optimizer/replay/双margin/dual/RNG保持，Reach400K父SHA17f3b355…、Payload120K父SHAb4410cb9…。两臂使用原固定λ=1，不叠加ramp，协方差/energy/20mm温度、奖励/15mm安全阈值/成功距离/原horizon/首次违规终止不变。训练半径均45/50/55mm，Reach57.5/60与Payload57.5仍未训练；不把DEV困难布局加回训练。

新候选只用已登记cubic taper：EMA预测margin≤−10mm零额外方向损失、≥0原强度、中间连续变化，包含原2/4mm激活offset；法向扩散不乘taper，无新critic/投影/候选动作/二元门控。控制臂也使用新注册元数据，但算法仍原NormalShrink固定方向损失。

control与candidate都在A100同一个GPU1/同Python源码同期，避免跨4080/A100归因混杂；先Reach两臂，两个槽释放后Payload两臂。原4080历史P5/Metric完整结果保留为外部参考，不拿它们代替本批同硬件control或虚构同seed因果对比。当前GPU0其他项目负载约100%，不暂停它；GPU1空闲，采用两个共享小显存槽，每进程allocator cap .025，不独占卡。

DEV沿用原任务固定100布局/原动作seed、20/40/60/80/100K五节点：Reach nominal/57.5/60，Payload nominal/55/57.5/execution085。它们都是已用于开发的DEV，不称独立集。完整100K终点评判，所有失败与中间点保留，不科学早停或自动重试。不凭一个适应seed晋升最终模型。

共同验收仍要求Reach60≥90%、57.5≥93%理想95%，相对原固定P5训练违规≤110%，不增加timeout/stall换安全；Payload两困难条件同时改善且nominal/execution不退化>2pp，同时审查训练代价/return/完成时间。候选必须与本轮同期control及原固定/Metric参考分别报告，不能以选择某个弱control改变目标。最终还缺真正独立多seed和Panda≤2pp回归，不在本批冒称完成。

存储重新按**实际同类原生运行大小**核算：A100此前Reach完整100K目录698MiB、Payload492MiB，四组约2.4GiB；本批保守预留4GiB，当前数据盘可用约55.6GiB，预计结束仍≥51GiB，超过40GiB底线及原checkpoint函数50GiB保护线。此前24GiB估算适用于保留逐步大轨迹的fresh250K，不应套用于当前原生Piper目录。队列启动门槛52GiB、每次保存仍保留旧storage_check保护，外部磁盘变化不回退home。

所有source/cache/TMP/结果经A100 storage wrapper写storage0，实际父路径和环境不迁移，不删除任何额外存档。预计四组总计约2–4小时，随共享负载调整；每30分钟监测。5080与实机均不使用。此登记只授权这四个固定任务，不自动扩张额外扫描。
