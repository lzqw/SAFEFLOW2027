# Panda Payload E01：最终独立测试

四条原生分支各新增100K、20候选共同验证及验证后审议完成后，冻结候选再一次性测试。
每条件100个新布局91027–91126，共1600回合；未依据测试换候选或继续训练。
训练seed41028；父年龄EA240K、Iso120K、SAC300K不同，仅新增适应预算相同。

| 冻结分支 | 新增候选年龄 | 条件 | 安全成功 | 违规 | 失败封顶平均用时 |
|---|---:|---|---:|---:|---:|
| EA-control | 100K | nominal | 100/100 | 0/100 | 0.7336秒 |
| EA-control | 100K | radius_070 | 91/100 | 9/100 | 1.0192秒 |
| EA-control | 100K | payload_wide | 75/100 | 25/100 | 1.5496秒 |
| EA-control | 100K | execution_scale_085 | 100/100 | 0/100 | 0.8508秒 |
| EA-repair | 80K | nominal | 100/100 | 0/100 | 0.7596秒 |
| EA-repair | 80K | radius_070 | 100/100 | 0/100 | 0.7492秒 |
| EA-repair | 80K | payload_wide | 100/100 | 0/100 | 0.7596秒 |
| EA-repair | 80K | execution_scale_085 | 100/100 | 0/100 | 0.8776秒 |
| Iso-repair | 80K | nominal | 100/100 | 0/100 | 0.7592秒 |
| Iso-repair | 80K | radius_070 | 99/100 | 1/100 | 0.7816秒 |
| Iso-repair | 80K | payload_wide | 100/100 | 0/100 | 0.7616秒 |
| Iso-repair | 80K | execution_scale_085 | 100/100 | 0/100 | 0.8608秒 |
| SAC-repair | 40K | nominal | 100/100 | 0/100 | 0.7436秒 |
| SAC-repair | 40K | radius_070 | 100/100 | 0/100 | 0.7352秒 |
| SAC-repair | 40K | payload_wide | 100/100 | 0/100 | 0.7456秒 |
| SAC-repair | 40K | execution_scale_085 | 100/100 | 0/100 | 0.8540秒 |

失败、违规或超时均计完整原生时域，不只统计成功回合。
零观测违规不是未来零风险：0/100的Wilson95%上界约3.70%。
这是共同验证选出的好候选，不是跨训练seed均值、训练稳定性或实机保证。
保留全部原生物理子步与碰撞检查；沿用原生逐条件动作随机流，
不声称在回合长度不同时仍逐布局逐动作噪声配对。70mm属于修复训练范围，
不叫未见半径外推；旧偏移条件不在本次最终测试，未证明合法替代路线。

[全部条件与风险区间](independent_summary.csv) · [全部1600回合](independent_episodes.csv)
· [冻结身份与测试前审议](freeze.json) · [原生完成收据](receipt.json)
· [来源哈希](report_receipt.json) · [测试前阶段决策](../E01_STAGE1_REVIEW_ZH.md)

完整恢复检查点留4080的4T数据盘；本报告仅整理已保存数据，没有新仿真／RL／实体动作。
