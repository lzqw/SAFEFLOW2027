# SAFEFlow 单 GPU 论文核心实验：正式结果与进度

更新时间：2026-08-25 01:16（Asia/Shanghai）

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

PointGoal 后处理在 `2026-08-25 01:15` 达到 `complete` 后，条件队列已自动启动 PointCircle seed 2027 的 `300k` 配对筛选：

| 配置 | Seed | 预算 | 当前状态 |
|---|---:|---:|---|
| C0 / isotropic FLAC | 2027 | 300k | 训练中 |
| P4.7 / Ours | 2027 | 300k | 训练中 |

预注册筛选规则保持不变：P4.7 的 return 相对 C0 下降不得超过 `10%`，并且 mean cost/CVaR 至少改善 `10%`，或高风险 `V_N/V_T <= 0.9` 且危险外向速度更低。只有筛选通过后才继续三 seed `500k` 正式结果及其后续 prior、机制、参数和 checkpoint-only 鲁棒性实验。

## 可复现身份

- 协议 SHA-256：`0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8`
- PointGoal 训练源 commit：`061787a2b393041cd54b8ccd38beab9316af46f8`
- PointGoal 评测与汇总源 commit：`7c63ac6dff81ae0135456a474d967c8d7dbbccff`
- 官方 SHAPO commit：`1bb770437bbbe40a89938661343744243e164bf9`
- 当前开发分支：`codex/star-single-gpu-paper-core`
- 当前执行 PR：[PR #12](https://github.com/lzqw/SAFEFLOW2027/pull/12)
