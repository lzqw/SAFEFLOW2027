# PointCircle 唯一几何增强：300k 冻结筛选报告

状态：`screen_not_passed`  
完成时间：2026-08-25 23:14（Asia/Shanghai）  
环境：`SafetyPointCircle1-v0`  
训练 seed：`2027`  
评测：5 个固定基础种子 × 4 episodes，共 20 episodes

## 冻结配置

原始 P4.7 在 `300k` 的高风险 `V_N/V_T > 0.9`，因此触发总协议唯一允许的一次增强。增强在训练前预注册，且只修改：

```yaml
safety_gradient_scale: 0.005 -> 0.001
safety_ensemble_scale: 0.05 -> 0.10
```

预注册 SHA-256 为 `fcde54b52e02c00f4c8997f676f5f963fd307dbae8d053559e55e1568c9ffc02`，配置 SHA-256 为 `d0252cb0bc22c783153bae45874322e99364d9a1458048591e3301c56111a6c3`。训练没有科学提前停止，也没有第二个候选或后续权重搜索。

## 结果

| 里程碑 | 配置 | Return | Mean Cost | CVaR-20 | 训练累计 Cost | 高风险 V_N/V_T | 危险外向速度 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 150k | C0 / isotropic FLAC | 3.4453 | 17.0 | 85.0 | 8913 | 0.99908 | **0.09655** |
| 150k | 原始 P4.7 | **28.7077** | **0.0** | **0.0** | **6068** | 1.00000 | 0.01738 |
| 150k | 几何增强 P4.7 | 16.2763 | 8.45 | 27.25 | 7208 | **0.75976** | 0.14266 |
| 300k | C0 / isotropic FLAC | 11.3807 | 0.0 | 0.0 | **9788** | 1.00000 | 0.10258 |
| 300k | 原始 P4.7 | 13.9828 | 0.0 | 0.0 | 11076 | 1.00000 | **0.03673** |
| 300k | 几何增强 P4.7 | **16.5585** | 0.0 | 0.0 | 10796 | 1.00000 | 0.09538 |

增强版在 `150k` 出现明确的高风险各向异性，`V_N/V_T=0.7598`；但到冻结终点 `300k` 时该指标回到 `0.999997`。因此中间节点支持“增强能够暂时激活方向机制”，但不能支持“机制在终点稳定激活”。

在 `300k`，增强版相对冻结 C0 的 return 提高约 `45.5%`，危险外向速度降低约 `7.0%`；训练累计 cost 高约 `10.3%`。两者最终 mean cost 和 CVaR-20 都是 `0`，所以按预注册规则相对安全改善记为 `0`。

## 冻结门槛

| 条件 | 结果 | 依据 |
|---|---|---|
| Return 下降不超过 10% | 通过 | 增强版 return 高于 C0 |
| Mean cost 或 CVaR-20 改善至少 10% | 未通过 | 两者均为 0，相对改善记为 0 |
| `V_N/V_T <= 0.9` 且 outward 更低 | 未通过 | outward 更低，但 `V_N/V_T=0.999997` |
| `return_ok AND (safety_ok OR geometry_ok)` | **未通过** | safety 与 geometry 分支均未满足 |

机器生成的冻结决定为：

```text
return_ok  = true
safety_ok  = false
geometry_ok = false
screen_pass = false
```

因此不启动 PointCircle 三 seed `500k` 正式矩阵、SACLag/SHAPO prior、机制控制或 PointCircle 鲁棒性评测。协议字段 `no_further_pointcircle_method_changes_allowed=true` 已锁定；PointCircle 方法修改和权重搜索到此停止。

这个单 seed 开发筛选不能被表述为 PointCircle 正式方法成功或失败，也不改变 PointGoal 已完成的三 seed 聚合结果。它提供的是一条明确的稳定性诊断：增强在 `150k` 激活了预期几何，但未将该激活保持到 `300k`。

## 证据

- [冻结决定](enhancement/screen_decision.json)
- [增强训练 receipt](enhancement/P47-geometry-enhanced-2027/receipt.json)
- [增强版 150k 评测](enhancement/P47-geometry-enhanced-2027/training/STAR-{SafetyPointCircle1-v0}/seed-2027-2026-08-25-12-34-18/artifacts/evaluation_step_150000.json)
- [增强版 300k 评测](enhancement/P47-geometry-enhanced-2027/training/STAR-{SafetyPointCircle1-v0}/seed-2027-2026-08-25-12-34-18/artifacts/evaluation_step_300000.json)
- [完整终态搬运 receipt](enhancement/stage_receipt.json)
- [正式矩阵终态](formal/queue_state.json)
- [prior 终态](priors/queue_state.json)
- [机制控制终态](controls/queue_state.json)
- [鲁棒性终态](robustness/queue_state.json)

冻结决定 SHA-256：`cd6a780d58f8e90711760e834346c9fb3fb60ac6cd9d45ba9f2e9246037e90e5`  
增强版 300k 评测 SHA-256：`38776b7d0a685ea830066360b847de48b7301f14a8a281993314fba55d50973b`
