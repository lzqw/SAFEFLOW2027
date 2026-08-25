# SAFEFlow 单 GPU 论文核心实验：正式结果与进度

更新时间：2026-08-25 13:45（Asia/Shanghai）

## 当前结论

PointGoal 四方法正式比较已经完成；PointCircle 原始 `300k` 单 seed 筛选也已完成并按冻结规则判定为未通过。PointGoal 所有方法使用训练 seeds `2027–2029`、每 seed `200k` 环境步，并对每个最终 checkpoint 使用 `5` 个固定基础种子、每个 `20` 个连续 episodes，共 `100` episodes。独立统计单位是训练 seed（`n=3`），不是合并后的 episodes。

| 方法 | Return | Success | Mean Cost | CVaR-20 | 训练累计 Cost | 高风险 V_N/V_T |
|---|---:|---:|---:|---:|---:|---:|
| C0 / isotropic FLAC | 0.4621 | 22.7% | 13.567 | 67.750 | 7305.3 | 0.9982 |
| SACLag | **25.2192** | **100.0%** | 53.190 | 108.933 | 12874.3 | n/a |
| SHAPO-PIDLag | 1.5040 | 51.3% | 24.020 | 108.383 | 9655.5 | n/a |
| P4.7 / Ours | 0.6545 | 25.3% | **9.433** | **46.800** | **5582.3** | **0.8744** |

结果支持以下判断：

1. P4.7 相对其严格等预算对照 C0 保持三 seed 聚合 Pareto 改进：return `+41.6%`、success `+2.7 pp`、mean cost `-30.5%`、CVaR-20 `-30.9%`、训练累计 cost `-23.6%`、高风险 `V_N/V_T -12.4%`、危险外向速度 `-16.9%`。
2. P4.7 是四种方法中 mean cost、CVaR-20 和训练累计 cost 最低的方法，也是唯一平均评测成本低于预注册 cost limit `10` 的方法。
3. SACLag 取得最高 return 和 success，但 mean cost 为 `53.19`、CVaR-20 为 `108.93`；SHAPO-PIDLag 的任务性能也高于 P4.7，但 mean cost 和 CVaR-20 分别为 `24.02` 和 `108.38`。因此 P4.7 的优势是安全侧，而不是对所有方法的无条件性能支配。
4. 结论边界不变：P4.7 对 C0 是跨三个训练 seed 的聚合 Pareto 改进，不宣称每个 seed 都 Pareto。SACLag 与 SHAPO-PIDLag 的正式结果没有参与 P4.7 的选择或修改。

## 当前运行状态

PointCircle 唯一一次几何增强正在服务器上正常训练。截至 `2026-08-25 13:44`，增强版 P4.7 已完成 `35k/300k` 环境步（`11.7%`），累计吞吐为 `7.92 FPS`，未出现 NaN 或 OOM。按当前累计吞吐估算，剩余训练约 `9.3` 小时；计入 `150k/300k` 两次固定评测后，预计在当日 `23:00–23:30`（Asia/Shanghai）形成冻结的 `300k` 筛选结论。服务器重启后的复查确认 learner 与全部条件控制器均在位，当前只有这一条科学训练占用 GPU。

训练日志在 `20k` 的参考批次上记录到 `Reference/VNVT=0.8743`、危险外向速度 `0.0342`。这些数值仅用于运行监控，不是高风险样本上的正式 checkpoint 评测，不能提前替代 `300k` 冻结门槛。当前没有科学提前停止，也没有根据该临时数值改变配置。

后续正式矩阵、prior baseline、机制控制和 checkpoint-only 鲁棒性均已部署为条件队列，但仍处于等待状态，没有启动任何下游科学训练：只有增强版在 `300k` 通过原封不动的门槛，才会进入 PointCircle 三 seed `500k` 正式阶段；若未通过，队列会记录 `screen_not_passed` 并停止 PointCircle 方法修改与权重搜索。

## Ant 二级扩展的已复用证据

服务器上已有的 Ant/P4.7 seed `2027`、`2M` checkpoint 及其 `100` episodes 评测已经完成身份核对并同步到主机，因此后续不会重复训练该任务。该单 seed 结果为：return `5823.8252`、mean cost `0.11`、CVaR-20 `0.55`、训练累计 cost `509`、高风险 `V_N/V_T=0.7006`、高风险危险外向速度 `0.3270`。它支持高维环境中方向各向异性确实激活，但当前只是一个训练 seed，不能替代后续三 seed 的 Ours/C0 正式比较。

Ant 剩余矩阵已在任何新 Ant 训练前锁定：新增 P4.7 seeds `2028/2029` 和 C0 seeds `2027–2029`，每条 `2M`，共 `10M` 新步数；最多两个 learner、`vector_env_nums=1`、`UTD=1`。新训练将使用与已复用 seed 2027 完全相同的历史训练代码 commit `62ca7399357c10645a955dcbead7ed2153a82c69`。P4.7 的科学参数保持不变，C0 仅关闭 `directional_loss_enabled` 和 `safety_aware_cost_geometry_enabled`。条件控制器已经部署到服务器，当前状态为 `waiting_for_point_core`、`scientific_training_launched=false`；只有 Point 核心达到 `complete` 或预注册的 `screen_not_passed` 终态后才会启动，因此目前不占用 GPU。

- [Ant/P4.7 seed 2027 的 2M 审计](shapo_p47_four_env_artifacts/milestones/step_2000000/ant-2027/milestone_audit.json)
- [Ant/P4.7 seed 2027 的 2M 评测](shapo_p47_four_env_artifacts/milestones/step_2000000/ant-2027/artifacts/evaluation_step_2000000.json)
- [Ant 单 GPU 补充预注册](single_gpu_ant_preregistration.json)
- [Ant P4.7 配置](../../configs/star/single_gpu_ant_p47.yaml)
- [Ant C0 配置](../../configs/star/single_gpu_ant_c0.yaml)

Walker2d 按协议只保留为 seed `2027` 的附录案例，不新增训练。其已有 `2M`、`100` episodes 结果也已同步：return `5810.7547`、mean cost `0.01`、CVaR-20 `0.05`、训练累计 cost `2312`、高风险 `V_N/V_T=0.7185`、危险外向速度 `0.2186`。由于正式 cost budget 为 `0` 且该 checkpoint 的 100 episodes 中出现 `1` 次 failure，它只能作为单 seed 方向机制与学习曲线证据，不能表述为多 seed 正式安全结论。

- [Walker2d/P4.7 seed 2027 的 2M 审计](shapo_p47_four_env_artifacts/milestones/step_2000000/walker2d-2027/milestone_audit.json)
- [Walker2d/P4.7 seed 2027 的 2M 评测](shapo_p47_four_env_artifacts/milestones/step_2000000/walker2d-2027/artifacts/evaluation_step_2000000.json)

## 正式证据

- [四方法正式报告](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/POINTGOAL_FOUR_METHOD_REPORT.md)
- [四方法聚合主表](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/pointgoal_four_method_summary.csv)
- [逐训练 seed 指标](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/pointgoal_four_method_seed_metrics.csv)
- [完整 episode 记录](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/pointgoal_four_method_episode_records.csv)
- [终态 manifest 与证据哈希](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/pointgoal_four_method_manifest.json)
- [全实验完成状态审计](single_gpu_paper_core_completion_audit.json)

主机同步目录只包含 JSON、CSV 和 Markdown。checkpoint、训练目录、工作目录和日志均未纳入 GitHub 证据包。

## PointCircle 原始 300k 筛选结果

PointCircle `SafetyPointCircle1-v0` 的 C0/P4.7 配对筛选已于 `2026-08-25 12:17` 完成。两种配置都使用训练 seed `2027`、`300k` 环境步；`150k` 和 `300k` 均使用相同的 `5 × 4 = 20` 个固定评测 episodes。该实验是开发筛选，统计单位只有一个训练 seed，不能替代三 seed 正式结果。

| 里程碑 | 配置 | Return | Mean Cost | CVaR-20 | 训练累计 Cost | 高风险 V_N/V_T | 危险外向速度 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 150k | C0 / isotropic FLAC | 3.4453 | 17.0 | 85.0 | 8913 | 0.99908 | 0.09655 |
| 150k | P4.7 / Ours | **28.7077** | **0.0** | **0.0** | **6068** | 1.00000 | **0.01738** |
| 300k | C0 / isotropic FLAC | 11.3807 | 0.0 | 0.0 | **9788** | 1.00000 | 0.10258 |
| 300k | P4.7 / Ours | **13.9828** | 0.0 | 0.0 | 11076 | 1.00000 | **0.03673** |

在正式筛选点 `300k`，P4.7 的 return 比 C0 高 `22.9%`，危险外向速度低 `64.2%`；但两者的 mean cost 和 CVaR-20 都已经落到 `0`，无法形成预注册要求的相对安全改善。P4.7 的高风险 `V_N/V_T=0.999999`，也未达到 `<=0.9` 的几何条件；训练累计 cost 反而高 `13.2%`。

冻结门槛的逐项结果为：

| 条件 | 结果 | 依据 |
|---|---|---|
| Return 下降不超过 10% | 通过 | P4.7 return 高于 C0 |
| Mean cost 或 CVaR-20 改善至少 10% | 未通过 | 两者均为 0，相对改善记为 0 |
| `V_N/V_T <= 0.9` 且 outward 更低 | 未通过 | outward 更低，但 `V_N/V_T=0.999999` |
| 总门槛 `return_ok AND (safety_ok OR geometry_ok)` | **未通过** | safety 与 geometry 分支均未满足 |

因此原始 PointCircle 配置没有进入三 seed `500k` 正式矩阵，prior、机制控制和鲁棒性队列也没有启动科学训练。`150k` 的明显安全优势作为学习曲线证据保留，但不能覆盖冻结的 `300k` 终点判定。这个结果说明原始配置在该 seed 上降低了危险外向速度，却没有让各向异性指标充分激活；它不是 PointGoal 正式结果失效的证据，也不能被表述为 PointCircle 上的方法成功。

- [PointCircle 300k 筛选报告](single_gpu_paper_core_artifacts/pointcircle/POINTCIRCLE_300K_SCREEN_REPORT.md)
- [机器可读冻结决策](single_gpu_paper_core_artifacts/pointcircle/screen_decision.json)
- [C0 300k 评测证据](single_gpu_paper_core_artifacts/pointcircle/screen_development_evidence/C0/evaluation_step_300000.json)
- [P4.7 300k 评测证据](single_gpu_paper_core_artifacts/pointcircle/screen_development_evidence/P47/evaluation_step_300000.json)

## 唯一一次几何增强

原始 P4.7 在 `300k` 的高风险 `V_N/V_T > 0.9`，触发了总协议事先允许的唯一一次增强。增强已在任何新训练开始前锁定，只修改：

```yaml
safety_gradient_scale: 0.005 -> 0.001
safety_ensemble_scale: 0.05 -> 0.10
```

增强实验固定为 seed `2027`、`300k`，复用哈希锁定的原始 C0 `300k` 评测，不重跑基线、不增加第二个候选、不做科学提前停止，且使用完全相同的筛选门槛。如果未通过，停止 PointCircle 方法修改和权重搜索；如果通过，才冻结增强配置并进入三 seed `500k` 正式矩阵。

- [增强预注册](single_gpu_pointcircle_geometry_enhancement_preregistration.json)
- [增强配置](../../configs/star/single_gpu_pointcircle_p47_geometry_enhanced.yaml)

## PointCircle 元数据修复

旧训练源生成的四个 `150k/300k` evaluation JSON 都漏写了 `environment` 标签。元数据修复现已全部完成：只添加 `"environment": "SafetyPointCircle1-v0"`，分别保留修改前后 SHA-256 以及排除该标签后的完整科学 payload SHA-256。审计终态明确记录 `scientific_metrics_changed: false`；该修复没有停止、修改或重跑 learner。

- [C0 150k 数值证据](single_gpu_paper_core_artifacts/pointcircle/screen_development_evidence/C0/evaluation_step_150000.json)
- [P4.7 150k 数值证据](single_gpu_paper_core_artifacts/pointcircle/screen_development_evidence/P47/evaluation_step_150000.json)
- [元数据修复审计状态](single_gpu_paper_core_artifacts/pointcircle/screen_metadata_repair_state.json)
- 修复逻辑初始 commit：`3a48c26dbd5d933efa4ccd5ee6d9fdb867bdde89`
- 轻量守护与来源历史 commit：`7b708b8dbb5e741eb5882ba1e2ce6d0e02ad77ca`

## 可复现身份

- 协议 SHA-256：`0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8`
- PointGoal 训练源 commit：`061787a2b393041cd54b8ccd38beab9316af46f8`
- PointGoal 评测与汇总源 commit：`7c63ac6dff81ae0135456a474d967c8d7dbbccff`
- PointCircle 训练源 commit：`ac334bfbcbedf6641aa3b454ac59497c38f8ce0b`
- PointCircle 原始筛选决策 SHA-256：`62873e89e8584fbb6663a183f71dcca84d33ef7b63296308bb47e88af1221d86`
- C0 300k 评测 SHA-256：`ac6e54f6086c6718209d415d552b0b97c5ee4e0958a858cc1aefd993091fbbcc`
- P4.7 300k 评测 SHA-256：`4d141267e74c579c14b4f4881e364fa5d875cdca0d481bf4ae894f4256b712e0`
- 几何增强预注册 SHA-256：`fcde54b52e02c00f4c8997f676f5f963fd307dbae8d053559e55e1568c9ffc02`
- 官方 SHAPO commit：`1bb770437bbbe40a89938661343744243e164bf9`
- 当前开发分支：`codex/star-single-gpu-paper-core`
- 当前执行 PR：[PR #12](https://github.com/lzqw/SAFEFLOW2027/pull/12)
