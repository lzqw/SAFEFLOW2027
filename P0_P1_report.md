# STAR P0/P1 工程门禁报告

## 结论

P0 与 P1 全部通过，可以进入后续代码评审；本次没有实现或启动 P2 及之后的 STAR 长训练。

- P0：四环境原生 API、Point 原生 cost、5k SACLag、真实反向更新吞吐矩阵、依赖锁和服务器清单全部通过。
- P1：Ant/Walker failure-cost、`terminated`/`truncated`、time-limit bootstrap、auto-reset final observation、seed/eval 隔离全部通过。
- 四环境分别完成 20k SACLag，共 80k 训练步；每个环境均完成 19,743 次真实更新、20 次 epoch 评测、5 个检查点以及最终检查点恢复评测。
- Point 与 MuJoCo 的实测最大吞吐配置均为 `4 个独立进程 × 每进程 1 个环境`。

20k smoke 只验证环境和训练管线，不是算法收敛实验。Point 的短程 cost 高于预算、Ant 的单回合恢复评测为零 failure，都不能被表述为最终安全性能或严格保证。

## 固定范围与版本

| 项目 | 固定值 |
|---|---|
| 开发分支 | `codex/star-p0-p1` |
| OmniSafe 基线 | `15603dd7a654a991d0a4648216b69d60b81a6366` |
| P0 提交 | `7a43ea1001886df37f9cb85b1c46f77eb30a954a` |
| FLAC 只读参考 | `61a25eef42935fc395b1e04071e2779311fcb705` |
| SHAPO 只读参考 | `1bb770437bbbe40a89938661343744243e164bf9` |
| Python | 3.11.15 |
| PyTorch / CUDA build | 2.8.0+cu128 / 12.8 |
| Gymnasium / Safety-Gymnasium | 0.28.1 / 1.0.0 |
| MuJoCo | 2.3.3；未安装 `mujoco_py` |

完整依赖见 [environment.lock.txt](environment.lock.txt)，安装约束见
[scripts/star/constraints-py311.txt](scripts/star/constraints-py311.txt)。Safety-Gymnasium 1.0.0
的 Python 3.11 兼容补丁是版本检查且幂等的；`pip check` 结果为
`No broken requirements found.`。

## 服务器与存储

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090，24,576 MiB，compute capability 8.6 |
| 驱动 | 570.124.04 |
| CPU cgroup 配额 | 14 cores |
| 内存 cgroup 上限 | 90 GiB |
| 数据盘 | `/root/autodl-tmp`，XFS，50 GiB |
| P0 清单时可用空间 | 37.884 GiB |
| P1 完成后可用空间 | 37.772 GiB |
| 计费口径 | 1.48 CNY/hour |

环境、缓存、运行日志、replay 和检查点均位于 `/root/autodl-tmp/SAFE`。主机只回传源码和小型报告；没有回传环境、缓存、replay、视频或完整检查点目录。完整硬件、磁盘、包版本和只读引用审计见 [system_manifest.json](system_manifest.json)。

清单中的 SAFEFLOW2027 状态记录为基线提交且工作树非干净，是因为性能门禁必须先运行，再把相同源码和结果封装为 P0 提交；对应的可复现 P0 快照是上表中的 `7a43ea1`。

## P0 结果

### 四环境原生 API

每个环境均执行了恰好 1,000 次原生 transition。

| 环境 | observation shape | termination | truncation | 原生 cost sum | 结果 |
|---|---:|---:|---:|---:|---|
| SafetyPointGoal1-v0 | 60 | 0 | 1 | 421 | 通过 |
| SafetyPointButton1-v0 | 76 | 0 | 1 | 7 | 通过 |
| Ant-v4 | 27 | 9 | 0 | N/A | 通过 |
| Walker2d-v4 | 17 | 45 | 0 | N/A | 通过 |

PointGoal 和 PointButton 都观察到非零原生 cost。原始结果见
[reports/star/four_env_smoke.json](reports/star/four_env_smoke.json)。

### SACLag 5k

SafetyPointGoal1-v0 完成 5,000 步、4,743/4,743 次真实更新和 6 个检查点。端到端耗时 112.107 秒，吞吐 44.60 FPS，峰值 CUDA allocated memory 35.4 MiB。结构化结果见
[reports/star/saclag_pointgoal_5k.json](reports/star/saclag_pointgoal_5k.json)。

### 吞吐矩阵

每组先预热 60 秒，再测量至少 600 秒。测量包含 rollout、真实 backward/update 和周期性 eval；全局 batch 固定为 256，实测 UTD 均严格为 1，且 actor 参数在测量窗口内发生变化。

| 环境 | P×E | FPS | update/s | GPU mean % | VRAM peak MiB | CPU quota % | RAM peak GiB | Temp peak °C | Power mean W | CNY/Mstep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SafetyPointGoal1-v0 | 1×1 | 53.61 | 53.61 | 9.1 | 403 | 7.1 | 14.59 | 40 | 110.2 | 7.668 |
| SafetyPointGoal1-v0 | 1×4 | 51.83 | 51.83 | 8.3 | 405 | 6.7 | 16.42 | 40 | 109.9 | 7.932 |
| SafetyPointGoal1-v0 | 2×2 | 90.93 | 90.93 | 69.7 | 806 | 12.8 | 17.98 | 49 | 161.9 | 4.521 |
| SafetyPointGoal1-v0 | 3×1 | 149.28 | 149.28 | 91.5 | 1,201 | 21.3 | 17.96 | 50 | 171.5 | 2.754 |
| SafetyPointGoal1-v0 | 4×1 | 178.23 | 178.23 | 95.4 | 1,601 | 28.4 | 19.77 | 50 | 174.1 | 2.307 |
| Ant-v4 | 1×1 | 58.90 | 58.90 | 9.1 | 379 | 7.1 | 13.73 | 40 | 110.3 | 6.980 |
| Ant-v4 | 2×1 | 115.86 | 115.86 | 86.2 | 754 | 14.2 | 14.87 | 50 | 168.8 | 3.548 |
| Ant-v4 | 3×1 | 166.43 | 166.43 | 98.0 | 1,129 | 21.3 | 16.00 | 50 | 174.9 | 2.470 |
| Ant-v4 | 4×1 | 199.89 | 199.89 | 98.3 | 1,505 | 28.3 | 17.13 | 50 | 175.8 | 2.057 |

完整精度和有效性字段见 [throughput_benchmark.csv](throughput_benchmark.csv)，机器可读选择见 [recommended_concurrency.json](recommended_concurrency.json)。

推荐配置：

- Point：`4×1`，178.23 FPS；需要额外运行余量时使用 `3×1`。
- MuJoCo：`4×1`，199.89 FPS；需要额外运行余量时使用 `3×1`。
- Ant/Walker 的 P1 wrapper 当前只支持单环境实例，因此 MuJoCo 推荐值表示多个独立进程，不表示单进程 vector env。

## P1 语义

### 环境与 replay 契约

- PointGoal/PointButton 直接使用 Safety-Gymnasium 原生 cost，预算 `d=10`。
- Ant/Walker 注册 `FailureCostMujocoEnv`，只有 `terminated and not is_healthy` 时 `cost=1`；time-limit truncation 的 cost 为 0，预算 `d=0`。
- `info` 记录 `failure_reason`、`is_failure`、健康高度、torso angle/orientation/upright、episode step/time 和 time-limit 标记。
- replay 分开保存 `terminated` 与 `truncated`；Bellman `done` mask 等于 `terminated`，所以 time limit 仍 bootstrap。终止和截断同时出现时保留 failure 标签。
- auto-reset transition 保存 `final_observation`，不把下一 episode 的 reset observation 写成 terminal next observation。
- train env 只在初始化设置 seed；后续 reset 推进 RNG。eval 使用 `seed + 10,000` 的确定性独立 seed stream，且 eval reset/close 不修改 train observation。

### 测试

主服务器检出执行结果：

| 测试 | 结果 |
|---|---:|
| `pytest -q tests/star/test_p1_semantics.py` | 12 passed |
| `pytest -q tests/test_env.py -k 'mujoco or safety_gymnasium'` | 9 passed |
| `pytest -q tests/test_buffer.py -k 'offpolicy_buffer'` | 2 passed，8 deselected |

12 项语义测试覆盖 Ant/Walker 人工跌倒、Walker 角度失败、纯 time limit、termination 与 truncation 同时发生、三种 replay done 组合、auto-reset final observation、连续 reset 布局、train/eval 隔离和固定且 disjoint 的 eval seed stream。

### 四环境 20k smoke

| 环境 | seed | steps | updates | FPS | 最终 epoch return | 最终 epoch cost | 恢复 return | 恢复 cost | checkpoints | CUDA peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SafetyPointGoal1-v0 | 2027 | 20,000 | 19,743 | 44.97 | 3.744 | 102.900 | 0.164 | 97.000 | 5 | 51.2 |
| SafetyPointButton1-v0 | 2028 | 20,000 | 19,743 | 41.17 | 4.410 | 206.500 | -0.257 | 191.000 | 5 | 88.2 |
| Ant-v4 | 2029 | 20,000 | 19,743 | 56.90 | 19.893 | 0.700 | 831.451 | 0.000 | 5 | 69.7 |
| Walker2d-v4 | 2030 | 20,000 | 19,743 | 60.45 | 332.087 | 1.000 | 355.610 | 1.000 | 5 | 65.7 |

每个环境都有 20 行有限 progress、`epoch-0/5/10/15/20.pt`，并从 `epoch-20.pt` 重新构造 evaluator 后完成一回合评测。机器可读结果见
[reports/star/p1_smoke_20k.json](reports/star/p1_smoke_20k.json)。

## 复现

在服务器检出中：

```bash
cd /root/autodl-tmp/SAFE/code/SAFEFLOW2027
SAFE_ROOT=/root/autodl-tmp/SAFE scripts/star/setup_server_env.sh
scripts/star/launch_p0_p1.sh p0-capture
scripts/star/launch_p0_p1.sh p0-env-smoke
scripts/star/launch_p0_p1.sh p0-saclag-5k
scripts/star/launch_p0_p1.sh p0-benchmark-point
scripts/star/launch_p0_p1.sh p0-benchmark-mujoco
scripts/star/launch_p0_p1.sh p1-tests
scripts/star/launch_p0_p1.sh p1-smoke-20k
```

统一入口、目录约束和结果回传规则见 [scripts/star/README.md](scripts/star/README.md)。

## 门禁边界

- P0/P1 状态：通过。
- P2+ 状态：未开始，也未运行 500k/5M 长实验。
- 这些 smoke 结果不能用于论文主表、收敛结论或零 failure 保证。
- 进入 P2 前仍应通过代码评审，并保持本报告中的 cost/done/reset 契约不变。
