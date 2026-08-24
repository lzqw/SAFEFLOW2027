# SAFEFlow 单 GPU 论文核心实验：结果与进度汇总

快照日期：2026-08-24（Asia/Shanghai）

## 总体结论

PointGoal 的 P4.7 正式实验已经完成，并在相同 `200k` 训练预算下取得聚合层面的 Pareto 改进：相对于各向同性 FLAC，P4.7 同时提高 return 和 success，并降低评测成本、尾部风险、训练累计成本以及高风险区域的法向探索。该结论来自 `3` 个训练 seed，每个最终 checkpoint 评测 `100` episodes；独立统计单位仍是训练 seed，而不是合并后的 episodes。

旧的 `150M` 步四环境长矩阵与单张 RTX 3090 的资源不匹配，现已安全归档，所有重要 checkpoint 均保留。当前执行的是冻结后的单 GPU 论文核心协议：先补齐 PointGoal 的 SACLag 与官方 SHAPO-PIDLag，再进入 PointCircle 的方法比较和机制实验。

## 已完成的正式 PointGoal 结果

环境：`SafetyPointGoal1-v0`；训练 seeds：`2027, 2028, 2029`；每个方法 `200k` steps；每个训练 seed 使用 `100` 个固定随机流评测 episodes。

| 方法 | Return | Success | Mean Cost | CVaR-20 | 训练累计 Cost | 高风险 V_N/V_T | 高风险外向速度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 / isotropic FLAC | 0.4621 | 22.7% | 13.567 | 67.750 | 7305.3 | 0.9982 | 0.2209 |
| P4.7 / Ours | 0.6545 | 25.3% | 9.433 | 46.800 | 5582.3 | 0.8744 | 0.1834 |
| P4.7 相对 C0 | **+0.1924（+41.6%）** | **+2.7 pp** | **-30.5%** | **-30.9%** | **-23.6%** | **-12.4%** | **-16.9%** |

由此可支持三项结论：

1. **整体安全—性能改进。** P4.7 的聚合 return 与 success 没有下降，同时 mean cost 和 CVaR-20 分别下降 `30.5%` 与 `30.9%`。
2. **训练期探索更安全。** 训练累计成本平均下降 `23.6%`，并在 `3/3` 配对训练 seeds 上降低。
3. **方向机制确实激活。** 在风险最高的 `20%` 样本中，`V_N/V_T` 从接近各向同性的 `0.9982` 降至 `0.8744`，危险外向速度也同步下降。

结论边界需要保留：这是**跨三个训练 seed 的聚合 Pareto 结果**，不是“每个 seed 都 Pareto 改进”。Return 在 `2/3` seeds 改善，success 在 `1/3` seeds 改善；mean cost、CVaR-20 和训练累计成本均在 `3/3` seeds 改善。

完整证据见 [P4.7 正式报告](p4_7_safety_aware_geometry_artifacts/final/summary/P4_7_SAFETY_AWARE_GEOMETRY_REPORT.md)、[逐 seed 指标](p4_7_safety_aware_geometry_artifacts/final/summary/p4_7_formal_final_seed_metrics.csv) 和 [科学解释 JSON](p4_7_safety_aware_geometry_artifacts/final/summary/p4_7_scientific_interpretation.json)。

## 旧长矩阵的处理

原方案要求 PointGoal、PointButton、Ant、Walker2d 共 `150M` 环境步，单张 RTX 3090 预计需要约 `68–104` 天，因此不再作为当前论文主线继续执行。

- 已停止旧队列及其自动恢复/同步服务，归档未完成的 [PR #10](https://github.com/lzqw/SAFEFLOW2027/pull/10)。
- 没有删除旧数据或 checkpoint。
- 主机已保留四个完整恢复 checkpoint：PointGoal `4.10M`、PointButton `4.10M`、Ant `4.15M`、Walker2d `4.15M`，另保留 `12` 个轻量 milestone/best-safe checkpoint。
- PointGoal 的超长训练在约 `4M` 时出现任务性能退化。它是后续稳定性诊断，不覆盖已经完成且预算严格相同的 P4.7 `200k` 正式比较。

## 当前服务器与训练队列

服务器已连接，RTX 3090 正常可用。当前 v3 控制器正在运行 PointGoal 正式基线的 seed 2027 配对训练；重启检查后没有发现 P4.7 重跑或旧长矩阵误恢复。

当前阶段是一次性的并发吞吐检查：

- 单 learner：SACLag seed 2027，`20k` steps，`95.14 FPS`；
- 双 learner：SACLag 与 SHAPO-PIDLag 各 `20k` steps，聚合 `180.57 FPS`；
- 若双 learner 聚合 FPS 至少达到单 learner 的 `1.20×`，正式队列保持两个槽，否则自动退回单 learner；
- 该检查只决定资源调度，不查看或选择科学指标。

实测双/单吞吐比为 `1.898×`，超过 `1.20×` 门槛，因此正式固定使用两个 learner。按实测线性外推，三轮 PointGoal 基线训练约需 `1.9` 小时；考虑长任务波动，运行窗口按 `2–4` 小时估计，最终固定种子评测和报告在训练后另行执行。

吞吐决定完成后，控制器会自动运行 PointGoal 的正式补充基线：

| 方法 | 实现 | Seeds | 预算 |
|---|---|---:|---:|
| SACLag | OmniSafe SACLag | 2027–2029 | `200k` / seed |
| SHAPO-PIDLag | 官方 SHAPO，commit `1bb7704` | 2027–2029 | `200k` / seed |

第一次双进程工程探针暴露了官方 SHAPO 的日志兼容问题：三个 CUDA 诊断标量被直接传入 NumPy，导致首个 epoch 写日志时退出。正式科学训练当时尚未开始。修复 commit `061787a` 只把这三个诊断副本转到 CPU 进行日志序列化，不改变优化器、环境、超参数或模型更新；失败目录完整保留，v2 从干净源码重新执行并完成工程探针。随后发现官方仓库会生成未跟踪的 Python bytecode，使严格源码洁净检查拒绝正式启动；未完成的短尝试已经停止并保留，v3 使用同一固定 commit、禁用 bytecode 写入后开始正式队列，没有复用任何部分训练结果。

训练使用冻结协议和修复后的源 commit `061787a`，`vector_env_nums=1`；SACLag 的 `UTD=1`。协议 SHA-256 仍为 `0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8`。20k 探针产物不会进入正式主表。

## 后续固定顺序与预计时间

1. PointGoal prior baselines：补齐 SACLag、SHAPO-PIDLag；训练实测预计 `2–4` 小时，含最终评测与汇总按 `1` 天内规划。
2. PointCircle 筛选与三 seed 正式 Ours/FLAC：预计 `2–3` 天。
3. PointCircle 两个 prior baselines：预计 `2–3` 天。
4. 单 seed 机制控制、一个参数点与 checkpoint-only 鲁棒性评测：约 `1–2` 天。

核心 Point 论文实验包总计仍按 `6–9` 天规划。Ant 是核心 Point 结果完成后的次级扩展；Walker2d 仅保留 seed 2027 作为附录案例。

PointCircle seed 2027 的筛选规则已经预注册：Ours 的 return 相对 FLAC 下降不得超过 `10%`，并且 mean cost/CVaR 至少改善 `10%`，或高风险 `V_N/V_T <= 0.9` 且危险外向速度更低。筛选通过后才继续三 seed `500k` 正式结果。

## 可复现身份

- 单 GPU 冻结协议：[single_gpu_paper_core_preregistration.json](single_gpu_paper_core_preregistration.json)
- 固定实验矩阵：[single_gpu_paper_core.yaml](../../configs/star/single_gpu_paper_core.yaml)
- 初始冻结实现 commit：`35474dec7c084be1e2891d78e7b157a5ac71c470`
- 日志兼容修复后的训练源 commit：`061787a2b393041cd54b8ccd38beab9316af46f8`
- 当前开发分支：`codex/star-single-gpu-paper-core`
- 当前汇总与执行 PR：[PR #12](https://github.com/lzqw/SAFEFLOW2027/pull/12)
