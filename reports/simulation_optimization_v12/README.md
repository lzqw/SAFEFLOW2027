# v12仿真优化

用户已允许Piper仿真、解除总预算限制，并取消接触响应gate。当前Piper Reach三分支和SafeReach早期风险六臂均已登记并运行，最终模型尚未确定。

- [当前运行状态及首个DEV片段（保留负结果）](RUNNING_STATUS_ZH.md)
- [Piper Reach三分支预登记](ROUND1_PREREGISTRATION_ZH.md)
- [SafeReach从头100K早期风险预登记](EARLY_SCREEN_PREREGISTRATION_ZH.md)
- [任务来源与准备核对（已被最新授权更新）](PREPARATION_REVIEW_ZH.md)
- [既有六模型真实CPU推理计时](baseline_cpu_latency/summary.csv)

两台机器每30分钟只读检查，不暂停外部任务；不使用5080，不连接或移动实体机器人。原始安全距离、成功定义、horizon和所有失败计数保持不变。旧独立集不用于新候选选择，当前全部训练节点评测仍为开发集。
