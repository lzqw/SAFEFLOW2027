# Piper Payload：coverage最终端点诊断与下一组

旧第一阶段Payload的约10mm风险低估来自未扩域端点，不能直接推断目前coverage模型同样低估。新诊断固定coverage100K终点（同120K父），两臂、55/57.5mm、各12个预定新DEV布局，无替换，包含全部终止成功/失败，零训练。

|条件|方案|成功/违规|违规中false-safe|违规均值预测误差|
|---|---|---:|---:|---:|
|55mm|EA coverage control|8/4|2|−0.625mm|
|55mm|P5 coverage|11/1|0|−0.374mm|
|57.5mm|EA coverage control|8/4|1|−0.217mm|
|57.5mm|P5 coverage|8/4|1|−0.064mm|

误差已明显小于旧未扩域模型，不支持盲目增加critic容量；P5危险端点梯度余弦约0.890–0.988。样本很少，11/12不是成功率从80%提升到92%的结论，57.5mm也仍失败；独立80%/63%保持原记录，不修改。

下一组固定100K预算、同EA120K完整父（sha256 b4410cb9…）、同45/50/55mm训练分布、同LR和目标Actor部署，对比固定P5、P5仅提前2mm激活、显式Metric-only。2mm只作用EMA几何值，沿用已验证实现，仍为待检验单因素；方向损失固定1。Metric-only为isotropic sampling+directional energy、无方向损失，保留所有原网络、优化器与replay，不能称旧EA control为Metric-only。环境15mm真安全阈值、原horizon及终止保持不变，无接触后继续运行、无动作候选筛选。

新DEV布局11870000起、动作11871000起，每条件100回合，20/40/60/80/100K完整曲线；nominal、55mm、57.5mm、execution085。统一evaluator明确将action_scale用于实际执行，Reach原条件scale1行为不变。57.5mm仍为未训练半径，不添加57.5/60mm训练后再称外推。

合格候选要求55/57.5mm成功同时优于同期固定P5与Metric-only，训练违规不高于固定P5的110%，无增加timeout/stall换低违规，保留return/完成时间；nominal/execution不降超过2pp。单DEV seed只用于开发；最终还需多seed、新独立集及Panda的≤2pp回归检查。并未宣称所有任务已全面领先。

4080有限两槽，等待当前Reach组合与Metric完成收据后自动接着三臂；不抢占外部项目，不使用5080或实体机器人，数据存4T盘。A100同时做E3重采样。未设总计费额度上限，仍不把工程smoke或诊断当正式训练结果。
