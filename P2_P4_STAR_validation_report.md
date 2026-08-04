# STAR P2–P4 方法自身验证报告

> 状态：P2、P3 已完成；正式 `SafetyPointGoal1-v0` 300k × 3 已完成。严格 P4
> 判定为 **FAIL**，因此按预注册规则停止，不进入 P5。

## 1. 范围、判定原则与证据版本

本轮只验证 STAR 路线 A 本身，不开展正式大型 baseline 比较，不运行 1M/5M/10M
训练，也不把内部反事实诊断解释为最终消融。P4 必须三个 seed 的全部预注册条件同时
成立才记为 PASS；任一 seed 或任一条件失败即如实记为 FAIL，并停止在 P4。

- GitHub 仓库：`lzqw/SAFEFLOW2027`
- 分支：`codex/star-p2-p4`
- 正式训练源码：`1a4533cd4a72c1488af346016ddc8f29f09745e4`
- 正式 GPU 复评、审计与绘图源码：`fd03cc046d489e2157f6885ccfb674277f9b68e2`
- W2 空激活步兼容及严格汇总源码：`ff62e5d9d663acc627b7ddc79fa77030ad0d55a3`
- 正式训练 seeds：2027、2028、2029
- 固定评测 seeds：12027、12028、12029、12030、12031
- 固定评测里程碑：0、50k、100k、200k、300k
- 设备：单台 RTX 3090；三个训练进程共享同一 GPU，均为 `1 process × 1 env`
- 服务器单价：1.48 CNY/h

训练进程使用的早期实现中，在线 Spearman 采用双 `argsort`，相同值会被任意打散。
该问题只影响诊断统计，不影响模型、优化器、动作或训练更新。正式证据统一使用修复后的
average-rank tie-correct 实现；全相同标签时 Spearman 记为未定义（JSON `null`、CSV
空值），严格判据按失败处理。所有 15 个里程碑检查点都将在同一 `cuda:0` 设备上离线
复评，避免 CPU/CUDA 长闭环轨迹的浮点分歧。

## 2. P2：STAR 模块与测试

### 2.1 实现范围

| 组件 | 主要落点 | 本轮状态 |
|---|---|---|
| action-space flow actor、ODE 中间点 | `omnisafe/models/actor_critic/star_actor_q_critic.py` | 完成 |
| 双 reward critics、FLAC 风格 target | 同上及 `omnisafe/algorithms/off_policy/star.py` | 完成 |
| distributional cost critic、quantile target | 同上 | 完成 |
| Point remaining budget；Ant/Walker future failure | `omnisafe/adapter/star_adapter.py`、`omnisafe/envs/failure_cost_mujoco_env.py` | 完成 |
| boundary gate、normalized JVP、切/法向几何 | `star_actor_q_critic.py`、`star.py` | 完成 |
| safety reference drift、residual path energy | `star.py` | 完成 |
| energy dual、真实 episode global safety dual | `omnisafe/common/star_dual.py`、`star.py` | 完成 |
| CPU replay、安全边界分层采样 | `omnisafe/common/buffer/star_replay_buffer.py` | 完成 |
| checkpoint：模型/优化器/dual/normalizer/RNG/计数器 | `omnisafe/common/star_checkpoint.py` | 完成 |
| flow 中间点、方向风险和机制日志 | `star.py` 与 `scripts/star/*` | 完成 |

按计划未实现路线 B、二阶项、latent flow、AMP 或 `torch.compile`；run manifest 会明确记录
这些开关为关闭，避免与路线 A 结果混淆。

### 2.2 测试覆盖

`tests/star/test_p2_modules.py` 覆盖：flow 边界与中间轨迹、distributional Bellman target、
FLAC reward target、quantile Huber loss、normalized JVP 的正尺度不变性、
`0 < V_N <= V_T`、gate=0 的各向同性恢复、gate 增大时 `V_N` 单调下降、critic 参数冻结下
actor 的一阶动作梯度、不构造 Hessian、CPU replay、并发不改变 batch/UTD、Point 与
locomotion 风险语义、原生 PointGoal 布局读取、完整 checkpoint 恢复，以及 W3 的非时间
证据门控。P3 微循环另行验证固定更新顺序、实际参数变化、原子证据文件和断点恢复。

服务器最新冻结工作树的最终 STAR 测试结果为 `42 passed`（23.93 s）；GitHub CI 结果见
第 8 节。

## 3. P3：完整训练闭环

### 3.1 更新顺序与 warmup

每次 UTD 更新的记录顺序严格为：

1. distributional cost critic；
2. twin reward critics；
3. 重新生成 differentiable flow；
4. terminal reward-cost utility；
5. 后半段 flow 的 safety geometry；
6. flow actor；
7. energy dual；
8. 基于真实 episode cost 的 global safety dual；
9. target networks 软更新。

W0–W3 不是单纯计步器。W3 同时要求最小步数、cost TD、calibration、有效风险梯度、
ensemble disagreement 和足量 boundary samples。未满足时停在 W2 是预期保护行为，不是
兼容性崩溃。

### 3.2 四环境 20k smoke

四个环境均完成 20,000 env steps、18,000 updates，无 NaN 或训练崩溃。该矩阵只验证接口
与闭环兼容性，不比较最终性能。

| 环境 | launcher 时间 (s) | FPS | 费用 (CNY) | 最终阶段 | boundary 样本 | 说明 |
|---|---:|---:|---:|---|---:|---|
| SafetyPointGoal1-v0 | 908.75 | 22.01 | 0.3736 | W2 | 2 | calibration 与 boundary 证据不足 |
| SafetyPointButton1-v0 | 1008.11 | 19.84 | 0.4144 | W3 | 364 | 12k 自然进入 W3 |
| Ant-v4 | 872.28 | 22.93 | 0.3586 | W2 | 0 | calibration 与 boundary 证据不足 |
| Walker2d-v4 | 858.32 | 23.30 | 0.3529 | W2 | 0 | calibration 与 boundary 证据不足 |

矩阵墙钟 3665.01 s（61.08 min），费用 1.5067 CNY。完整小型证据位于
`results/star/p3/`。

### 3.3 并发吞吐

吞吐测试固定每 worker global batch=256、UTD=1，并在真实随机预填充后强制 W3，仅用于
工作负载测量，不作为自然通过 W3 的证据。

| 配置 | aggregate env FPS | GPU 利用率均值/峰值 | 峰值 VRAM (MiB) | CNY / 1M steps |
|---|---:|---:|---:|---:|
| 1×1 | 14.3483 | 8.18% / 9% | 383 | 28.6523 |
| 2×1 | 28.4356 | 75.75% / 88% | 762 | 14.4576 |
| 3×1 | 41.5205 | 91.47% / 98% | 1141 | 9.9014 |
| 4×1 | 53.8350 | 95.07% / 99% | 1521 | 7.6365 |

四档总墙钟 192.12 s，对应约 0.0790 CNY。4×1 吞吐最高；正式 P4 恰有三个 seeds，故以
3×1 并发运行三条独立 `1×1` 训练，不改变单 seed 的 batch 或 UTD。

## 4. P4：50k 开发运行

seed 2027 在 29,568 steps 自然进入 W3，50k 运行完成 48,000 updates，无 NaN/崩溃；
launcher 墙钟 3023.12 s，峰值 CUDA allocation 50.07 MiB，费用 1.2428 CNY。

| 指标 | 0 | 50k | 变化 |
|---|---:|---:|---:|
| evaluation return | -1.7259 | 0.1731 | +1.8990 |
| evaluation mean cost | 48.4 | 13.4 | -35.0 |
| cost CVaR20 | 200.0 | 67.0 | -133.0 |
| violation ratio | 4.84% | 1.34% | -3.50 pp |

最终 full replay 共 50,000 个已标注样本：ECE 0.1759、MAE 0.1908、tie-correct
predicted-cost/empirical-cost Spearman 0.1996；boundary/failure/normal 分别为
564/1229/48207。boundary 样本的随机动作来源比例和立即违规比例均为 0，说明边界信息
并非主要依靠主动执行明显不安全动作。

同状态、同 base action 的内部反事实显示：boundary gate 0.4365，interior gate 0.003065
（约 142.4×）；directional `V_N=0.5636` 对比 isotropic `V_N=1`，下降 43.6%；
`V_T=1`，保留 100%；有效梯度率 0.6904。50k action-risk map 的全平面变化仅约
`3.6e-5`，轨迹也较保守且未清晰到达目标，因此 50k 只能作为工程/机制开发证据，不能
单独支持 P4 PASS。在线 evaluation Spearman 受旧 tie bug 影响，不作正式证据。

开发证据位于 `results/star/p4/development_50k/`，不包含 checkpoint、replay 或视频。

## 5. P4：300k × 3 正式结果

三条训练均完成 300,000 env steps、298,000 updates，无崩溃或 NaN。它们在同一秒启动并
共享单张 RTX 3090；最慢 seed 2027 的 launcher 墙钟为 21,911.10 s（6.086 h），所以这是
三条并发训练的实际共享墙钟和计费基准，而不是三个 receipt 的简单相加。

每个检查点都用固定 seeds 12027–12031 在 `cuda:0` 重评。下表先对每个训练 seed 的五个
episode 求均值，再对三个训练 seed 求均值：

| checkpoint | return | mean cost | cost CVaR20 | violation ratio |
|---:|---:|---:|---:|---:|
| 0 | -16.3994 | 21.2000 | 83.0000 | 2.1200% |
| 50k | 0.1056 | 7.6000 | 36.6667 | 0.7600% |
| 100k | -0.0953 | 0.4667 | 2.3333 | 0.0467% |
| 200k | 0.0723 | 0.5333 | 2.6667 | 0.0533% |
| 300k | -0.0186 | 0.0000 | 0.0000 | 0.0000% |

最终结果按训练 seed 展开如下。三条最终评测的 empirical cost/risk 都完全相同为 0，故
tie-correct Spearman 在 300k 处是未定义值，而不是 0 或负相关；风险对应关系改由完整
resident replay 审计判断。

| seed | return 0 → 300k | cost 0 → 300k | CVaR20 0 → 300k | final eval violation |
|---:|---:|---:|---:|---:|
| 2027 | -1.7259 → -0.1404 | 48.4 → 0 | 200 → 0 | 0% |
| 2028 | -27.8912 → -0.0243 | 3.4 → 0 | 13 → 0 | 0% |
| 2029 | -19.5809 → 0.1088 | 11.8 → 0 | 36 → 0 | 0% |

完整 replay 的 300,000 个样本均有标签。风险排序在三个 seed 上都为正，但 calibration
并不理想：empirical risk/predicted risk 分别为 0.1128/0.0855、0.1100/0.0676、
0.1406/0.0739，后两条尤其有低估倾向。

| seed | 最终阶段（W3 激活步） | boundary 数 | ECE / MAE | replay cost Spearman | gate boundary / interior（倍数） | `V_N` / `V_T` | 危险法向速度：末次 W2 → final（变化） |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2027 | W3（29,568） | 1,377 | 0.1152 / 0.1342 | +0.2820 | 0.1957 / 0.003311（59.12×） | 0.7909 / 1.0000 | 0.0250 → 0.1491（+0.1241，FAIL） |
| 2028 | W2（未激活） | 4 | 0.1161 / 0.1270 | +0.2734 | 0.4164 / 0.037016（11.25×） | 0.6966 / 1.0000 | 0.0586 → 0.0586（0，FAIL） |
| 2029 | W3（212,251） | 880 | 0.1320 / 0.1481 | +0.3145 | 0.4495 / 0.002738（164.17×） | 0.6233 / 1.0000 | 0.1813 → 0.1286（-0.0527，PASS） |

seed 2028 的 gate 与 directional variance 是同状态反事实候选诊断；由于它始终停在 W2，
这些局部几何并未成为实际训练路径。三条 boundary 样本的 random-source fraction 和
immediate-violation fraction 都为 0，但 seed 2028 只有 4 个 boundary 样本，覆盖不足。

严格汇总共检查 11 项，结果如下。`PASS` 只表示该条预注册判据在该 seed 上成立：

| 严格判据 | 2027 | 2028 | 2029 |
|---|:---:|:---:|:---:|
| 完成且无崩溃 | PASS | PASS | PASS |
| actor / critics / duals 非退化 | PASS | PASS | PASS |
| predicted–empirical risk 正相关 | PASS | PASS | PASS |
| boundary gate 显著高于 interior | PASS | PASS | PASS |
| 法向方差下降 | PASS | PASS | PASS |
| 切向方差保持 | PASS | PASS | PASS |
| 危险区域法向速度下降 | **FAIL** | **FAIL** | PASS |
| boundary 非主要来自随机/立即违规动作 | PASS | PASS | PASS |
| 最终 warmup 证据通过 | PASS | **FAIL** | PASS |
| return 改善且未按判据坍塌 | PASS | PASS | PASS |
| evaluation cost 或 tail risk 改善 | PASS | PASS | PASS |

因此 seed-level 结果为 2027 FAIL、2028 FAIL、2029 PASS；跨 seed 的 P4 结果是 FAIL。

## 6. 十个问题的回答

1. **Critics 是否稳定？** 数值上稳定：三条都完成、参数与损失非退化、无 NaN，最终
   cost-TD EMA 为 0.0365、0.0666、0.0360。但训练图中的 reward-TD 后期仍上升，且
   calibration EMA 为 0.1471、0.3903、0.1780，所以只能说“可训练且有限”，不能声称
   已一致收敛。
2. **预测风险是否对应经验风险？** 排序方向是对的：full-replay tie-correct Spearman
   为 +0.2820、+0.2734、+0.3145。绝对校准仍弱，ECE 为 0.115–0.132，平均预测风险
   低于平均经验风险，不能把它解释为精确概率。
3. **Gate 是否主要在安全边界开启？** 是。boundary/interior 均值比为 59.12×、
   11.25×、164.17×；但 2028 的结果是未激活 W3 时的候选反事实。
4. **法向方差是否下降？** 是。相对 isotropic `V_N=1`，最终为 0.7909、0.6966、
   0.6233。
5. **切向方差是否保留？** 是。三个 seed 的 `V_T` 都为 1.0。
6. **Flow 是否在危险区降低法向速度？** 跨 seed 结论为否。只有 2029 从 0.1813
   降至 0.1286；2027 从 0.0250 增至 0.1491，2028 不变。这是本轮最核心的失败项。
7. **全局与局部机制是否都实际参与？** 全局 actor/critics/duals 在三条均非退化；局部
   W3 几何只在 2027 和 2029 激活。2028 的候选 gate/VN 不能等同于实际参与，因此答案
   是否定的。
8. **边界数据是否避免主动执行明显不安全动作？** 已观察样本中是：三条的随机来源和
   立即违规比例均为 0。不过 2028 只有 4 个 boundary 样本，只支持“未观察到”，不能
   支持充分覆盖。
9. **Return 是否改善？** 三条都相对极差的初始检查点明显改善，跨 seed 均值从
   -16.3994 到 -0.0186；但 50k–300k 在零附近波动，固定种子轨迹显示保守停滞，尚无
   证据证明可靠完成导航任务。
10. **Cost、tail risk 和违规是否改善？** 固定评测的 final mean cost、CVaR20 和
    violation ratio 在三条都为 0。训练 resident replay 的最终 failure/violation 比例仍为
    2.3207%、2.0957%、2.5513%，所以只能确认评测安全性改善，不能宣称训练分布风险消失。

## 7. 机制图与局限

所有公开 PNG 均在复制回主机后逐张人工检查：

- [聚合训练诊断](results/star/p4/final_300k/figures/star_training_diagnostics.png)显示评测
  cost/CVaR 与 violation 最终归零，return 在约 50k 后停留在零附近；同时训练 cost
  仍有尖峰、reward-TD 后期上升，危险法向速度没有跨 seed 一致下降。
- [固定种子轨迹汇总](results/star/p4/final_300k/figures/pointgoal_trajectories.png)及
  [2027](results/star/p4/final_300k/mechanisms/seed-2027/pointgoal_trajectories.png)、
  [2028](results/star/p4/final_300k/mechanisms/seed-2028/pointgoal_trajectories.png)、
  [2029](results/star/p4/final_300k/mechanisms/seed-2029/pointgoal_trajectories.png)的单 seed 图
  都显示后期轨迹聚集在起点附近，未清晰到达图中目标；初始远距离轨迹又把聚合图比例
  拉大。这使“零 cost”更像保守停滞，不能当作任务成功。
- replay reliability 图（[2027](results/star/p4/final_300k/audit/seed-2027/replay_risk_calibration.png)、
  [2028](results/star/p4/final_300k/audit/seed-2028/replay_risk_calibration.png)、
  [2029](results/star/p4/final_300k/audit/seed-2029/replay_risk_calibration.png)）显示 exceedance
  样本总体偏向较高 predicted risk，但 reliability bins 明显锯齿且存在低估，与
  ECE/MAE 数值一致。
- final action-risk map 在 2027、2028 上全网格严格为 0.5，2029 也仅在
  `[0.5, 0.5000001192]` 内变化；对应图见 [2027](results/star/p4/final_300k/mechanisms/seed-2027/action_risk_map.png)、
  [2028](results/star/p4/final_300k/mechanisms/seed-2028/action_risk_map.png)、
  [2029](results/star/p4/final_300k/mechanisms/seed-2029/action_risk_map.png)。这说明代表状态下
  动作风险面几乎饱和为常数，是当前实现的关键局限。
- flow storyboard（[2027](results/star/p4/final_300k/mechanisms/seed-2027/flow_storyboard.png)、
  [2028](results/star/p4/final_300k/mechanisms/seed-2028/flow_storyboard.png)、
  [2029](results/star/p4/final_300k/mechanisms/seed-2029/flow_storyboard.png)）能看到规则 action grid
  随 ODE 收缩到窄带，证明 flow 在变换动作；但在近常数风险面上，这本身不能证明安全
  几何学到了有意义的动作方向。

预注册输出包括训练/评测 return、训练/评测 cost、CVaR、违规样本比例、critic TD、
calibration、tie-correct Spearman、gate、有效梯度率、`V_N/V_T`、法向速度、两个 dual、
PointGoal 轨迹、action-risk map 与 flow storyboard。公开目录只保留小型 CSV/JSON/PNG；
不提交 replay、checkpoint、虚拟环境、缓存或大型视频。

## 8. 复现、测试、时间与费用

训练、恢复、离线评估、回放审计、绘图、严格汇总和无人值守流水线的命令见
`scripts/star/README.md`。后处理器先验证三个 seed、300k/298k 计数、五个里程碑和非空
检查点，再使用固定 CUDA 复评表生成 verdict。

无人值守后处理的 15 次 CUDA 复评、三个 full-replay 审计、机制导出和绘图均成功，最后
一步严格汇总最初因 seed 2028 合法的 `geometry_activation_step=null` 被错误执行
`int(None)` 而退出。失败 receipt 原样保留在
`results/star/p4/final_300k/postprocess_initial_failure.json`；修复提交 `ff62e5d` 只在 CPU
重新执行汇总，未重跑训练、GPU 评测、replay 审计或机制导出。恢复凭据见
`results/star/p4/final_300k/summary/summary_recovery.json`，科学结论读取
`summary/pointgoal_p4_verdict.json`。

验证结果：

- 服务器最新源码：`python -m pytest tests/star -q` → `42 passed in 23.93s`；
- GitHub 源码提交 `ff62e5d`：Python 3.8、3.9、3.10 三档测试和 lint 共 4 项全部通过；
- 最终目录共约 7.1 MiB：131 个服务器生成文件经 checksum dry-run 确认传输无差异，
  主机另加 1 个恢复凭据；未包含 `.pt`、`.pth`、`.ckpt`、`.npz`、replay、虚拟环境、
  缓存、视频或符号链接。

按 1.48 CNY/h、只统计有明确 receipt 或时间边界的服务器阶段：

| 阶段 | 实际共享墙钟 | 估算费用 (CNY) |
|---|---:|---:|
| P3 四环境 smoke | 61.08 min | 1.5067 |
| 并发吞吐测量 | 3.20 min | 0.0790 |
| P4 50k 开发运行 | 50.39 min | 1.2428 |
| P4 300k × 3 并发训练 | 365.18 min | 9.0079 |
| 训练完成后的离线后处理上界 | ≤11.38 min | ≤0.2806 |
| **可核算合计** | **≤8.19 h** | **≤12.1170** |

严格 verdict 中的 25.5478 CNY 是三个 seed receipt 的累计 seed-hours 成本；因为三个进程
共享同一按时计费实例，不能把它当作实际账单。上表也不包含阶段间空闲、环境准备和极短
CPU 恢复时间，因此是“已测作业阶段”的核算，不是云平台最终账单。

## 9. 最终判定

**P4：FAIL。** 2027、2028、2029 的 seed-level verdict 依次为 FAIL、FAIL、PASS。

- 2027 唯一失败项是危险区域法向速度未下降，且实际从 0.0250 上升到 0.1491；
- 2028 未通过最终 warmup 证据、始终停在 W2，并且危险区域法向速度没有下降；
- 2029 通过全部 11 项，但预注册规则要求三个 seed 全部通过。

结论不是“方法完全无效”：风险排序、boundary gate、法向方差抑制、切向保持和固定评测
安全性都有一致的正证据。但危险法向速度缺乏跨 seed 一致性、W3 激活不稳、代表状态的
action-risk surface 近常数、轨迹呈保守停滞，足以否决方法自身验证。依照停止规则，本轮
**没有启动 P5，也没有开展正式 baseline、大规模消融或更长训练**。下一轮若继续，应先
修复风险面饱和与 boundary 覆盖、加入明确的导航成功/进展指标，并重新从 P4 三 seed
门槛验证，而不是把本轮结果推进到 P5。
