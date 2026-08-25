# PointCircle 300k 冻结筛选报告

生成时间：2026-08-25 12:25（Asia/Shanghai）

## 结论

原始 P4.7 在 PointCircle seed `2027` 的 `300k` 筛选中 **未通过** 预注册门槛。它相对 C0 获得更高 return 和更低危险外向速度，但两种方法在终点评测上的 mean cost/CVaR-20 都为 `0`，且 P4.7 的高风险 `V_N/V_T` 仍接近 `1`，所以 safety 与 geometry 两个可选分支都未满足。

这是单训练 seed 的开发筛选结果，不是三 seed 正式比较。原始配置不会继续进入 PointCircle `3 × 500k` 正式矩阵。

## 结果

环境为 `SafetyPointCircle1-v0`。C0 与 P4.7 都从头训练 `300k` 环境步，训练 seed 为 `2027`；每个评测点使用 5 个固定基础种子、每个 4 episodes，共 20 episodes。

| 里程碑 | 配置 | Return | Mean Cost | CVaR-20 | 训练累计 Cost | 高风险 V_N/V_T | 危险外向速度 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 150k | C0 / isotropic FLAC | 3.4453 | 17.0 | 85.0 | 8913 | 0.99908 | 0.09655 |
| 150k | P4.7 / Ours | **28.7077** | **0.0** | **0.0** | **6068** | 1.00000 | **0.01738** |
| 300k | C0 / isotropic FLAC | 11.3807 | 0.0 | 0.0 | **9788** | 1.00000 | 0.10258 |
| 300k | P4.7 / Ours | **13.9828** | 0.0 | 0.0 | 11076 | 1.00000 | **0.03673** |

正式 `300k` 终点上，P4.7 相对 C0：

- return：`+22.9%`
- mean cost：相同，均为 `0`
- CVaR-20：相同，均为 `0`
- 训练累计 cost：`+13.2%`
- 危险外向速度：`-64.2%`
- 高风险 `V_N/V_T`：无实质下降，`0.999999` 对 `1.000000`

`150k` 点显示 P4.7 曾有明显的 cost、CVaR、训练 cost 和 outward 优势，但冻结协议只允许 `300k` 决定是否进入正式矩阵，因此该中间点不能替代终点判定。

## 冻结门槛

决策逻辑为：

```text
return_ok AND (safety_ok OR geometry_ok)
```

| 条件 | 结果 |
|---|---|
| P4.7 return 相对下降不超过 10% | `true` |
| Mean cost 或 CVaR-20 相对改善至少 10% | `false` |
| 高风险 `V_N/V_T <= 0.9` 且 outward 严格更低 | `false` |
| 最终 screen pass | **`false`** |

两种方法的终点 cost/CVaR 同为零时，相对改善按冻结决策实现记为零；几何分支则因 P4.7 的 `V_N/V_T=0.9999989271 > 0.9` 未通过，即使 outward 已经更低。

## 后续边界

`V_N/V_T > 0.9` 触发了总协议预先规定的唯一一次几何增强：

```yaml
safety_gradient_scale: 0.005 -> 0.001
safety_ensemble_scale: 0.05 -> 0.10
```

增强候选在训练前锁定为一次、一个 seed、`300k`，复用原始 C0 终点评测并沿用同一门槛。若增强仍未通过，不再进行 PointCircle 方法修改或权重搜索。

## 证据与身份

- [唯一一次增强预注册](../../single_gpu_pointcircle_geometry_enhancement_preregistration.json)
- [机器可读冻结决策](screen_decision.json)
- [队列终态](queue_state.json)
- [C0 训练 receipt](screen/C0-2027/receipt.json)
- [P4.7 训练 receipt](screen/P47-2027/receipt.json)
- [C0 150k 评测](screen_development_evidence/C0/evaluation_step_150000.json)
- [C0 300k 评测](screen_development_evidence/C0/evaluation_step_300000.json)
- [P4.7 150k 评测](screen_development_evidence/P47/evaluation_step_150000.json)
- [P4.7 300k 评测](screen_development_evidence/P47/evaluation_step_300000.json)
- [元数据修复审计终态](screen_metadata_repair_state.json)
- 协议 SHA-256：`0a21c9165116e15eb13c4f1c84956de515cb7631e3a1a93611239cdd54b1cdf8`
- 训练源 commit：`ac334bfbcbedf6641aa3b454ac59497c38f8ce0b`
- 冻结决策 SHA-256：`62873e89e8584fbb6663a183f71dcca84d33ef7b63296308bb47e88af1221d86`
- C0 300k 评测 SHA-256：`ac6e54f6086c6718209d415d552b0b97c5ee4e0958a858cc1aefd993091fbbcc`
- P4.7 300k 评测 SHA-256：`4d141267e74c579c14b4f4881e364fa5d875cdca0d481bf4ae894f4256b712e0`
