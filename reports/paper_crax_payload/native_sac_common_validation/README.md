# SAC 共同开发验证：两个配置各五个候选

实际执行：A100共享GPU0，原生PID32186正常退出。分析源以运行前SHA256固定为
`dd123845b2f3aaede0c6a3d4e74eebfc927fe90961210c12dbe35043f32e05a9`。
两个已完成的normalization=true SAC配置分别来自e4ad097与699c5be；唯一注册调参差别
为lagrangian_lr .01/.1，其余实际训练设置保留在 `dd12384/plan.json`。

每个配置保留全部五个预先记录checkpoint，各在**同一61027开发布局、50个episode**
验证，与当前Flow候选screen规则一致。未使用710xx独立测试，也没有用这些结果
代替固定3M终点。后续五个开发seed/独立测试仍需统一开展。

## 全部结果

Return与Goals/Episode在该原生稀疏reward配置数值相同，下表合列；成本是完整原生
episode累计成本，budget为25。到达一次目标之后继续原生目标刷新，并非立即终止。

| SAC dual LR | 实际checkpoint | Return /Goals | ≥1目标比例 | Mean episode cost | 成本≤25比例 | 预算内episode Return | Capped首目标秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| .01 |616192|.56|38%|113.972|6%|1.000|64.554|
| .01 |1212160|.86|52%|28.436|56%|.679|52.627|
| .01 |1808128|3.12|90%|18.573|74%|2.000|11.706|
| .01 |2404096|28.88|100%|88.062|0%|无合格episode|3.253|
| .01 |3000064|33.74|100%|87.198|0%|无合格episode|2.792|
| .1 |616192|.18|18%|34.205|50%|.120|70.640|
| .1 |1212160|.92|40%|2.767|98%|.898|55.392|
| .1 |1808128|5.24|88%|53.748|54%|3.296|20.557|
| .1 |2404096|9.80|98%|35.899|36%|4.389|8.096|
| .1 |3000064|30.08|100%|61.215|0%|无合格episode|3.106|

这里已经出现一个**有任务进展且均值成本合格的开发候选**：默认.01的1808128点，
平均3.12目标、90%至少一次目标、mean cost18.573≤25。它不是“所有episode都满足
预算”：仍有26%episode超过25。它也不是独立测试收益或三训练seed统计结论。

不能因为最终3M任务数更高就忽略其成本超标；也不能选.1的1212160点仅因其成本2.767
就称任务已学会，因为它只有40%episode到达目标、平均目标数.92。所有五点完整记录，
共同规则可用于之后选择，未修改全局MODEL_SELECTION或主结果表。

## 评测实现与开销

- 一次构造native环境，一次编译 `evaluate(params, reset_keys, noise_key)`，跨10个policy/
  normalizer输入复用，不把每个候选参数捕获成重新编译的闭包常量。
- Reset PRNG拆分与PointGoalTorch.reset完全一致，policy噪声单独流。首episode active
  mask保留原生终止/截断，全部native1000控制步和frame_skip4不变。
- SavedFullCheckpointManager的policy export带单设备pmap轴。新reader严格检查所有
  参数leaf的singleton轴后恰好移除一次，不改变保存值、不丢弃多设备分片；也支持将来
  已unreplicated的原生policy格式。每行明确记录 `stored_policy_replica_axis_removed=0`。
- 0新增训练、0optimizer/dual更新、0真实交互。实际**500000非训练控制区间**，
  500个首episode、10候选结果；一次编译78.329秒，总164.221秒。
- 第一个admission版本7a60095在`.lower()`阶段发现保存schema的额外pmap轴，尚未执行
  unroll，0控制区间。错误记录与trace保留在 `failures/`，没有重复已跑物理样本。
- 必要的seed隔离、指标和单设备schema检查共3项CPU测试通过，不替代实际上述GPU数据。

全部逐episode原始结果、候选SHA256、plan、receipt和紧凑CSV在 `dd12384/`。
分析代码未编辑原生SAC训练文件、其他项目进程或root汇总器。
