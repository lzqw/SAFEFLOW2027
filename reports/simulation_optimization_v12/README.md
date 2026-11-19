# v12仿真优化

用户已允许Piper仿真、解除总预算限制，并取消接触响应gate。26个100K开发端点和四组250K匹配消融均已完成；仍未全部达标。方向项产生后期安全收益，但完整P5未全面胜过Metric。下一项补同硬件确认与同期Iso，未继续扫已失败的权重/温度。最终新版本尚未确定。

- [最新：四组250K完整结果与后续硬件确认](MATCHED_ABLATION_FINAL_20261119_ZH.md)
- [最新：Panda完整结果与未启动消融的双机加速交接](PANDA_FINAL_AND_ABLATION_HANDOFF_ZH.md)
- [局部化完整结果，拒绝继续扫温度](LOCALIZED_FINAL_REVIEW_ZH.md)
- [新seed250K匹配方向项/lateLR机制实验](MATCHED_ABLATION_PREREGISTRATION_ZH.md)
- [250K完整消融的分析口径与硬件限制](ABLATION_ANALYSIS_RULES_ZH.md)
- [E4完整结果与A100几何局部化历史登记](SAFETY_REPLAY_FINAL_AND_LOCALIZED_LAUNCH_ZH.md)
- [Reach半强度与Payload4mm完整终点](FOLLOWUP_REVIEW_1357_ZH.md)
- [E3完整负结果：低违规来自大量任务放弃](BOUNDARY_REPLAY_FINAL_REVIEW_ZH.md)
- [最新26个完整终点训练指标](consolidated_endpoints_1632/training_endpoints.csv) 与 [逐条件DEV指标](consolidated_endpoints_1632/development_endpoints.csv)
- [统一指标口径与缺失值说明（首批18端点快照）](CONSOLIDATED_METRICS_ZH.md)
- [终止动作噪声归因：不同任务不宜统一降噪](TERMINAL_NOISE_REVIEW_ZH.md)
- [安全权重局部化的历史准备记录](LOCALIZED_GEOMETRY_PREPARATION_ZH.md)（已在E3/E4完成后另行正式登记启动）
- [Reach第三轮、Payload首轮完整结果](ROUND3_PAYLOAD_AND_FOLLOWUP_ZH.md)
- [Reach训练风险—任务成功关系图](piper_tradeoff_figures_v2/reach_task_risk_pareto.png) 与 [Payload配对关系图](piper_tradeoff_figures_v2/payload_task_risk_pareto.png)（单训练seed DEV，不是独立结论）
- [完整目标证据台账与未完成事项](REQUIREMENTS_STATUS_ZH.md)
- [Panda显式Metric匹配参考预登记（不是新候选回归通过）](PANDA_REFERENCE_PREREGISTRATION_ZH.md)
- [前一轮Reach与早期风险完整结果](ROUND2_AND_EARLY_REVIEW_ZH.md)
- [最近运行交接与仍未完成事项](OPERATIONS_20261118_ZH.md)
- [全部早期学习曲线](early_figures/early_training_tradeoffs.png) 与 [逐条件DEV曲线](early_figures/early_development_curves.png)
- [逐时间节点/条件与基线的描述性比较](early_pairwise/summary.json)
- [Payload最终coverage诊断和下一组三臂对照](PAYLOAD_NEXT_COMPARISON_ZH.md)
- [追加全部任务/几何目标与结论边界](OBJECTIVE_UPDATE_ZH.md)
- [首个DEV片段（历史运行记录，保留负结果）](RUNNING_STATUS_ZH.md)
- [Piper Reach三分支预登记](ROUND1_PREREGISTRATION_ZH.md)
- [SafeReach从头100K早期风险预登记](EARLY_SCREEN_PREREGISTRATION_ZH.md)
- [任务来源与准备核对（已被最新授权更新）](PREPARATION_REVIEW_ZH.md)
- [既有六模型真实CPU推理计时](baseline_cpu_latency/summary.csv)

两台机器每30分钟只读检查，不暂停外部任务；不使用5080，不连接或移动实体机器人。原始安全距离、成功定义、horizon和所有失败计数保持不变。旧独立集不用于新候选选择，当前全部训练节点评测仍为开发集。
