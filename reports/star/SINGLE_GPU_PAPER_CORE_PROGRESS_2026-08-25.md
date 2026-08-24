# SAFEFlow 单 GPU 论文核心实验：正式结果与进度

更新时间：2026-08-25 06:53（Asia/Shanghai）

## 当前结论

PointGoal 四方法正式比较已经完成。所有方法使用训练 seeds `2027–2029`、每 seed `200k` 环境步，并对每个最终 checkpoint 使用 `5` 个固定基础种子、每个 `20` 个连续 episodes，共 `100` episodes。独立统计单位是训练 seed（`n=3`），不是合并后的 episodes。

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

## 正式证据

- [四方法正式报告](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/POINTGOAL_FOUR_METHOD_REPORT.md)
- [四方法聚合主表](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/pointgoal_four_method_summary.csv)
- [逐训练 seed 指标](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/pointgoal_four_method_seed_metrics.csv)
- [完整 episode 记录](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/pointgoal_four_method_episode_records.csv)
- [终态 manifest 与证据哈希](single_gpu_paper_core_artifacts/pointgoal/postprocess/final/pointgoal_four_method_manifest.json)

主机同步目录只包含 JSON、CSV 和 Markdown。checkpoint、训练目录、工作目录和日志均未纳入 GitHub 证据包。

## 当前 PointCircle 阶段

PointGoal 后处理在 `2026-08-25 01:15` 达到 `complete` 后，条件队列已自动启动 PointCircle seed 2027 的 `300k` 配对筛选。截至 `06:52`，两条 learner 均到达 `155k/300k`，训练进程正常，GPU 利用率约 `80%–93%`：

| 配置 | Seed | 当前步数 | 最近 FPS | 当前状态 |
|---|---:|---:|---:|---|
| C0 / isotropic FLAC | 2027 | 155k / 300k | 7.65 | 训练中 |
| P4.7 / Ours | 2027 | 155k / 300k | 7.53 | 训练中 |

`150k` 预注册中间评测使用相同的 `5 × 4 = 20` episodes；它是学习曲线点，不是正式终点：

| 配置 | Return | Mean Cost | CVaR-20 | 训练累计 Cost | 高风险 V_N/V_T | 危险外向速度 |
|---|---:|---:|---:|---:|---:|---:|
| C0 / isotropic FLAC | 3.4453 | 17.0 | 85.0 | 8913 | 0.99908 | 0.09655 |
| P4.7 / Ours | **28.7077** | **0.0** | **0.0** | **6068** | 1.00000 | **0.01738** |

在这个单 seed 中间点上，P4.7 的 return 高约 `733%`，mean cost 与 CVaR-20 均降低 `100%`，训练累计 cost 降低 `31.9%`，危险外向速度降低约 `82.0%`。如果把冻结规则临时应用在 `150k`，它会通过 return 与 safety 分支；但正式决策只使用 `300k`，因此没有提前停止、没有修改配置，也不把该中间点表述为最终结论。高风险 `V_N/V_T` 仍接近 `1`，当前明显收益主要表现为成本、回报和外向速度，而不是强各向异性激活。

预注册筛选规则保持不变：P4.7 的 return 相对 C0 下降不得超过 `10%`，并且 mean cost/CVaR 至少改善 `10%`，或高风险 `V_N/V_T <= 0.9` 且危险外向速度更低。只有筛选通过后才继续三 seed `500k` 正式结果及其后续 prior、机制、参数和 checkpoint-only 鲁棒性实验。

按当前双 learner 的稳定吞吐，剩余 `145k` 步加最终评测与 checkpoint 预计还需约 `5.5` 小时，`300k` 正式筛选结果预计在 `2026-08-25 12:10–12:40`（Asia/Shanghai）生成。如果通过，队列会自动继续 PointCircle C0/P4.7 的三 seed `500k` 正式阶段；该两方法正式阶段预计还需约两天。

## PointCircle 元数据修复

`150k` 数值产物完整，但旧训练源生成的 evaluation JSON 漏写了 `environment` 标签，会使 `300k` 冻结门槛的任务身份校验失败。已部署最小、可审计的元数据修复：只添加 `"environment": "SafetyPointCircle1-v0"`，并分别保留修改前后文件 SHA-256 以及排除该标签后的完整内容 SHA-256。审计状态明确记录 `scientific_metrics_changed: false`。守护程序正在等待并以同样方式处理 `300k` 两个文件，不停止或修改 learner。

- [C0 150k 原始数值证据](single_gpu_paper_core_artifacts/pointcircle/screen_development_evidence/C0/evaluation_step_150000.json)
- [P4.7 150k 原始数值证据](single_gpu_paper_core_artifacts/pointcircle/screen_development_evidence/P47/evaluation_step_150000.json)
- [元数据修复审计状态](single_gpu_paper_core_artifacts/pointcircle/screen_metadata_repair_state.json)
- 修复逻辑初始 commit：`3a48c26dbd5d933efa4ccd5ee6d9fdb867bdde89`
- 轻量守护与来源历史 commit：`7b708b8dbb5e741eb5882ba1e2ce6d0e02ad77ca`

## 可复现身份

- 协议 SHA-256：`0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8`
- PointGoal 训练源 commit：`061787a2b393041cd54b8ccd38beab9316af46f8`
- PointGoal 评测与汇总源 commit：`7c63ac6dff81ae0135456a474d967c8d7dbbccff`
- PointCircle 训练源 commit：`ac334bfbcbedf6641aa3b454ac59497c38f8ce0b`
- 官方 SHAPO commit：`1bb770437bbbe40a89938661343744243e164bf9`
- 当前开发分支：`codex/star-single-gpu-paper-core`
- 当前执行 PR：[PR #12](https://github.com/lzqw/SAFEFLOW2027/pull/12)
