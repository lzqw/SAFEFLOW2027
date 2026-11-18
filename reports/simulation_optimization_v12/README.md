# v12仿真优化

用户已允许Piper仿真、解除总预算限制，并取消接触响应gate。Reach前两轮五臂、SafeReach早期六臂已完成；尚未同时达到全部目标。当前继续Reach组合与显式Metric-only、A100边界重采样及Payload对照，最终模型尚未确定。

- [最新完整Reach与早期风险结果](ROUND2_AND_EARLY_REVIEW_ZH.md)
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
