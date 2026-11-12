**这一轮就按“CRAX PointGoal 高吞吐主实验＋Payload 操作扩展＋机制与泛化分析＋Piper 部署准备”四条线并行推进。**重点是尽快得到一个有竞争力的配置和部署候选，不再把“所有 seed 同时表现好”作为继续研究的前提。

完整执行文件已经整理好，可以直接交给 Codex：

**:chatgpt-content-reference{index="7"}[下载完整实验任务书：CRAX、Payload 与 Piper 总体方案](sandbox:/mnt/data/Codex_SAFEFlow_CRAX_Payload_Piper_MasterPlan_v1.md)**

这次方案有两个重要调整。

**第一，CRAX 必须按它自己的任务语义接入。**PointGoal 默认可以在到达目标后刷新目标，成本会累计；官方 SAC-Lag 也使用累计折扣成本 critic。因此不能直接复制机械臂的“首次违规终止＋sigmoid 概率 Q＋预算 0.1”。 

**第二，加速目标是完整训练，而不是单独提高仿真 FPS。**CRAX 已有设备内转换的 PyTorch wrapper，可以先保留现有 learner，不必重写全部算法；但该 wrapper 仍是四返回值接口，需要正确处理 reset、终止和截断。

下面是可直接复制的执行指令。附件进一步展开了接口、数据语义、成本归一化、调参分支和交付字段。

# Codex 执行任务：SAFEFlow 论文整体实验

## 一、总目标与执行原则

本轮围绕以下论文实验体系推进：

\[
\boxed{
\text{公开安全任务}
+
\text{机械臂操作扩展}
+
\text{方向熵与Pullback机制}
+
\text{扰动泛化}
+
\text{Piper实机准备}.
}
\]

优先调出一个“训练足够快、确实完成任务、成本受控”的强候选。允许在本任务规定范围内主动调参、修复实现和调整并行，不需要对每个学习率或 batch 修改重新请求确认。

不要求每个 seed 都获胜才继续。但所有运行结果、选择规则和实际预算必须保留。重点案例可以使用好的策略，基线也必须有相同的候选选择机会。

本轮不是恢复旧的失败训练，也不是再次要求完成一整套跨 seed 稳定性审计。旧结果不修改，新任务独立记录。

---

## 二、代码与资源

从以下提交建立独立分支：

```text
项目：lzqw/SAFEFLOW2027
起点：fc5716e6dc38bc2eefd1b61cc3cb52090b8f07ca
分支：codex/paper-crax-payload-v1
```

优先复用现有 EA-STAB v2 的 Actor、critic、margin、endpoint geometry、评估和恢复代码。

CRAX 使用官方 `TTomilin/CRAX`。本次核对版本为：

```text
63504b540705be98024abd675e646ff5e046c6c2
```

使用该版本或服务器上已经确认可运行的明确版本，记录实际依赖，不在运行中静默升级。

资源沿用已有授权：

| 资源 | 第一优先任务 | 后续补位 |
|---|---|---|
| A100 GPU0 | CRAX 原生 SAC-Lag、公共训练接口与吞吐配置 | PointGoal Iso／EA |
| A100 GPU1 | EA 批量采样、第二配置、成对实验 | 新 seed、Local 或 CVPO |
| RTX4080 | Payload 开发与训练 | 机制分析、Piper 仿真 |
| CPU 与空闲小槽 | 评估、汇总、视频、数值实验 | 不阻塞主要 learner |

启动前确认实际空闲资源，不停止其他项目进程。5080 继续不使用，除非另行获得授权。

大 checkpoint 和 replay 保留在 4080 的 4T 硬盘及必要的 A100 原件。GitHub 只上传代码、配置、紧凑结果和图表。

采用简单动态队列：一个槽完成后立即补任务，不等整批结束，不重新建设多层 watchdog 系统。

---

## 三、CRAX PointGoal：先做一个公开任务，不铺开全环境

### 3.1 固定任务

```text
env_id: safe_goal_point
difficulty: 1
observation: 原生状态观测
rendering: 训练时关闭
```

保持原生物理步长、frame skip、目标刷新、hazard、reward 与 cost。

本次核对的 SafeGoal 基础默认值为 1000 个控制步、物理步长 0.02、n_frames=4；实际创建 difficulty 后记录最终有效配置。

不要靠缩短 episode、减少障碍、放宽安全条件来制造加速或成功。

### 3.2 成本适配必须正确

为 learner 增加清楚的任务模式：

```text
episodic_binary：现有机械臂首次违规终止任务
cumulative：CRAX 原生累计成本任务
```

CRAX 模式：

- 不使用限制在 [0,1] 的概率 cost-Q；
- 不沿用机械臂预算 0.1；
- 首轮使用统一的 episodic cost budget 25，若固定官方配置另有覆盖，以记录值为准；
- 保留所有原生成本来源；
- 原始成本、折扣成本 Q、reward scaling 和 dual 更新单位分别说明；
- 每个并行环境独立累计 episode cost，不能把短采样块的成本直接当作完整 episode 成本；
- 自动 reset 后的新起点不能作为上一条 transition 的 terminal observation。

Actor、noise、投影和能量目标全部按实际动作维数参数化，移除写死的三维设置。

### 3.3 PointGoal 主要指标

报告：

\[
\text{原生Return},\quad
\text{Episode Cost},\quad
\text{Goals/Episode}.
\]

补充：

- 至少完成一次目标的 episode 比例；
- 满足成本预算的 episode 比例；
- 训练累计成本及每千 transitions 成本；
- 在成本预算内取得的任务回报。

不能只统计“碰到一次目标”，而忽略之后长时间停滞；也不能用低成本掩盖没有任务进展。

---

## 四、GPU 训练：重点调环境批量与学习更新比例

### 4.1 首选工程路线

```text
CRAX / MJX GPU 环境
        ↓
设备内 JAX–PyTorch 转换
        ↓
批量 stochastic flow Actor
        ↓
批量 replay 与 learner
```

第一版继续使用 PyTorch 算法，不立即整体重写为 JAX。

训练热路径要求：

- 环境 reset/step 批量化并编译；
- Actor 一次处理全部环境；
- Replay 批量写入和采样；
- 不逐环境执行 Python 网络调用；
- 不逐步在 CPU 与 GPU 之间搬运观测和动作；
- 不对每个 loss 高频调用 `.item()`；
- 几何计算保留输入梯度，不能被全局 inference mode 禁用；
- 训练、评估、环境与策略噪声使用分离的随机流。

JAX 与 PyTorch 混用时，先关闭 JAX 默认大块预分配：

```text
XLA_PYTHON_CLIENT_PREALLOCATE=false
```

再根据实际内存使用调整，不让两个框架各自按默认方式占满显存。

### 4.2 不再机械沿用“每条 transition 更新一次”

定义：

- \(E\)：并行环境数；
- \(F\)：每个采样块的 vector steps；
- \(G\)：每块采样后的 learner 更新轮数；
- \(B\)：训练 batch size。

则：

\[
N_{\mathrm{new}}=EF,
\qquad
u=\frac{G}{EF},
\qquad
r_{\mathrm{replay}}=\frac{GB}{EF}.
\]

其中 \(u\) 是每条新 transition 的 learner 更新轮数。

禁止把环境数增加到 256 后，仍在每次 vector step 后串行执行 256 轮更新。

先比较三个配置，\(F=1\)：

| 配置 | 环境数 \(E\) | Batch \(B\) | 每块更新 \(G\) | Replay exposure |
|---|---:|---:|---:|---:|
| Balanced，默认 | 256 | 1024 | 4 | 16 |
| Fast | 512 | 1024 | 4 | 8 |
| Learn-heavy | 128 | 512 | 8 | 32 |

只做一次短 profiling，然后固定最合适配置继续学习。显存不足时按比例缩小，不反复进行大量基准测试。

环境数更多不必然更好；相同 replay exposure 也不等于相同优化过程。最终按“任务表现＋安全代价＋完整训练吞吐”选择。

### 4.3 公共 learner 起点

建议初始配置：

```text
hidden_width = 256
actor_lr = 1e-4
critic_lr = 3e-4
reward_loss = Huber(beta=1)
reward_updates_per_round = 2
actor_update_interval = 2
reward_target_samples = 4
flow_steps = 4
reference_variance = 0.04
normal_variance_ratio = 0.25
replay_capacity = 1M
```

原始状态归一化、reward scaling、cost scaling 都记录并保持对照一致。

能量目标按动作维度计算：

\[
E_{\mathrm{target}}
=
d_a\times0.25/\nu_0.
\]

不能直接复制机械臂的 18.75。若 reward 使用缩放，初始化 \(\alpha_E\) 时同步检查量纲。

CRAX 的几何渐进启用按有效 margin 更新次数，而不是机械照搬旧 10k–30k transition：

```text
margin updates < 1000：各向同性
1000–5000：线性启用
>=5000：完整当前几何权重
```

Margin EMA 先用 0.005；采样与能量必须使用同一个实际参考矩阵。

### 4.4 速度目标

在可用 A100 上，完整 EA 训练争取：

```text
基本目标：约 2,000 transitions/s
进一步目标：约 5,000 transitions/s 或更高
```

这是工程努力目标，不是预计或保证。

同时记录：

```text
编译耗时
稳态完整训练SPS
optimizer updates/s
端到端总耗时
显存
评估与保存开销
```

速度不够时，先定位物理、Actor/tail、learner、同步四部分。不要为了达到 SPS 数字把更新量减到策略根本学不会。

---

## 五、PointGoal 调参和正式实验

### 5.1 开发顺序

第一步，原生 JAX SAC-Lag 跑一个短参考，与 Flow 接入并行。先 1M，学习趋势需要时延长到 3M。

第二步，让 Stable-Iso 在同一任务真正学会到达目标，再比较 EA。不能拿未学会的基线作为最终对照。

第三步，使用一个开发 seed 筛选配置：

```text
候选先500k
→ 保留非支配的两个配置到2M
→ 最佳共享主干与EA到3M–5M
→ 增加第二个开发seed
```

一个好候选可以继续进入独立评价和机制实验，不要求第二个 seed 必须同样好。

CRAX 开发总预算上限为 **24M transitions**，包括参考、候选、失败和工程采样，不是每个候选 24M。

### 5.2 允许主动调整的顺序

| 问题 | 优先动作 |
|---|---|
| 仿真快但整体慢 | 批量 VJP、减少同步、调整 \(E/B/G\) |
| 采样很多但 critic 没学会 | 增加更新比例或减少环境数 |
| SAC 能学、Flow 不学 | 检查动作尺度、reward/energy 量纲和 \(\alpha_E\) |
| 任务能学但成本过高 | 先检查 cost/dual 单位，再调共享安全压力 |
| 安全但停滞 | 查看 goal 数与 reward/cost/energy 梯度，不能只继续降噪声 |
| 几何几乎不激活 | 检查动作标签、距离尺度和 margin 覆盖 |
| 正确方向但效果弱 | 单独尝试 \(\rho_N:0.25\rightarrow0.1\) |
| EA 损失任务能力 | 单独尝试 \(\rho_N:0.25\rightarrow0.5\) |
| 值准但动作梯度错 | 小规模真实有限差分诊断，再决定是否加差值监督 |

最多先探索六个主要训练配置，不做学习率、网络、噪声、dual、batch 的全组合扫描。

SAC 也保留合理调优机会；不能只优化 EA，把无效 baseline 留在主表。

### 5.3 PointGoal 期望目标

优先找到满足以下目标的候选：

- 至少达到一个目标的 episode 比例 ≥80%；
- 平均目标数争取 ≥2，同时原生回报有实际提升；
- mean episode cost 不超过共同预算；
- 任务能力达到有效 SAC 参考的约 90% 以上；
- 相对 Stable-Iso，争取任务基本不下降时成本减少约 20%，或者同成本时任务表现提高约 10%。

回报为负时，使用绝对差或随机—有效基线之间的动态范围，不直接计算误导性的百分比。

以上是优化方向，不是要求程序必须输出的结论。

### 5.4 主比较

主方法：

```text
SAC-Lag
Stable-Flow-Iso
EA
```

每方法三个预先记录的训练 seeds。

根据开发结果冻结共同预算：

```text
优先5M / seed
仍明显学习时，统一10M / seed
```

不能只给 EA 延长训练。

CVPO 仅在 PointGoal 上作为第四个基线补充，不阻塞前三种核心结果。

预算：

| 部分 | 5M 主预算 | 10M 主预算 |
|---|---:|---:|
| 开发上限 | 24M | 24M |
| 三方法×三 seeds | 45M | 90M |
| CVPO×三 seeds | 15M | 30M |
| PointGoal 机制/参数上限 | 6M | 6M |
| 合计上限 | **90M** | **150M** |

这是 GPU 批量采样的整体预算上限，不要求用满，更不默认照搬官方 500M×全部算法设置。

---

## 六、Payload 与 CRAX 并行推进

新增任务：

```text
SafePayloadTransport-v1
```

机器人开始时已夹持轻质物体，固定姿态，使用三维末端增量绕障到目标。先不学习抓取。

安全距离：

\[
d_{\min}
=
\min(d_{\mathrm{arm,obs}},d_{\mathrm{payload,obs}}).
\]

Margin 使用真实控制区间最小距离。允许的抓持接触不算违规；机器人或载荷撞障都必须统计。

所有方法共享载荷尺寸、位姿等可获得观测。

### 训练

先做：

\[
3\text{ 方法}\times2\text{ 开发seeds}\times300k
=
1.8M.
\]

方法为 SAC-Lag、Stable-Iso、EA。最多追加约 1.2M 针对性修正，Payload 开发总上限 3M。

确认阶段统一使用 0.6M；需要时所有方法统一到 1M，每方法三个 seeds。Payload 总训练上限 12M。

若从旧 Reach 权重迁移，单列 transfer 结果，初始化条件公平，不把已有训练成本当作零。

### 期望效果

```text
Nominal safe success ≥90%，进一步争取95%
Nominal violation ≤5%
Stress safe success ≥80%
Stress violation ≤10%
```

重点争取相近风险下比 Iso 多约 10 个百分点的安全成功率，或任务基本保持时明显减少风险。

不再把“350次/1M”作为统一硬门槛；累计违规、episode 数与违规率一起报告。

---

## 七、好策略选择与主实验二

### 7.1 好 seed 可以成为重点候选

每种方法保存：

1. 固定终点的全部 seed 结果；
2. 共同验证规则选出的部署策略，在独立测试集上的结果。

候选数、允许的 checkpoint 数和选择规则对所有方法一致。

先满足共同风险要求，再按任务能力排序，最后比较完成时间。没有满足风险要求的候选时，输出明确标记的 Pareto 折中，不把停滞策略称为成功。

EA－2027 可以保留为历史正例和后续分析起点，但已经看过的测试不再称为新的独立测试。

### 7.2 固定策略泛化

保留三类测试：

| 测试族 | 内容 |
|---|---|
| 几何 | 障碍半径、位置、间隙、载荷尺寸 |
| 执行 | 只选择控制延迟或动作尺度误差的一族 |
| 替代路线 | 改变已有障碍位置，让常用路线受限但另一条仍可行 |

先每条件 30–50 episodes；最终选定策略每条件 100 episodes。

测试时冻结 Actor、margin、normalizer 和 dual。逐条件报告，不只给有利条件的平均值。

Reach 已有 7/8 cm、横向偏移结果继续保留；不重跑原来的全部长训练。

已知球形半径外推问题可以做一个独立的几何响应修正版本，但不能直接针对已看过的 8 cm 测试拟合常数补丁。若改变训练随机化，所有方法共享，并另留真正未见范围。

---

## 八、机制与参数实验

机制集中在一个机械臂任务，PointGoal 只补少量共同状态分析，不在每个任务复制全矩阵。

### M1：方向分配还是单纯减小噪声？

比较：

```text
Iso-High
Iso-Low
Normal-Shrink
Directional-Full
```

分别明确匹配总 trace、法向噪声或其他条件，不能声称所有指标同时匹配。

重点画：

\[
\text{真实风险}
\quad\text{vs.}\quad
\text{切向方差／任务进展}.
\]

改变全局 reference variance 时，同步检查 energy 与 \(\alpha_E\) 的量纲。

### M2：为什么需要 pullback？

同一冻结 Actor、共同状态、共同噪声谱：

```text
Local
Endpoint-Direct
Endpoint-Pullback
Random
```

测真实一步风险、最终 \(V_N,V_T\)、方向错位、clipping。Oracle 仅用于共同评价，不能进入 learned 执行选择。

建议共同状态池：

```text
128 states × 64 draws
```

包含安全内部和近边界，不只选好 seed 的安全状态。

### M3：随机性是否产生有用行为？

统计安全成功路线、任务相关安全覆盖、成功次数与超时。

增加原生采样器／共同采样器交叉评估，区分训练后的 Actor 改善与测试时几何调制。

### M4：小型理论数值实验

使用可控线性 flow 验证：

- pullback 的终点风险方差公式；
- 方向错位的 \(\sin^2\vartheta\) 项；
- 固定 trace 的 constrained log-det 解；
- 同 covariance Gaussian 的相对熵公式。

不要把椭球体积变化当作任意真实动力学泛化的证明。

### 参数和效率

只扫描：

\[
\rho_N\in\{0.1,0.25,0.5,1.0\},
\]

复用默认和 Iso，最多新增两个参数点。

K=2/4/8 先作为冻结模型采样器敏感性，明确不是三个完整重训结果。

报告单动作 p50/p95 延迟、完整训练 SPS、编译时间、更新数和显存。

Arm 机制、参数和小额修复合计新增训练上限 3M。优先复用已有 checkpoint。

---

## 九、Piper 实机线：准备工作立即并行，实体执行单独启用

### 9.1 Codex 本轮直接完成

- 接入匹配实际型号、夹爪的官方 Piper 模型；
- 建立仿真 Reach 和 Payload；
- 对齐状态、动作单位、坐标系、控制周期；
- 生成 Iso、Iso-Low、EA 部署候选；
- 完成 dry-run、记录回放、通信超时处理；
- 保存 Actor、margin、normalizer、动作缩放和版本。

不能把 Panda 权重直接改名为 Piper 策略。

Piper 仿真开发预算先设 3M；机器人信息不完整时生成待填配置，继续其他实验，不阻塞全部项目。

### 9.2 实体运动边界

本任务不授权 Codex 自动连接实体机器人启动运动或随机探索。

实际实验由现场操作者确认设备、急停、工作空间和速度/负载限制后启用。所有方法使用同一保护条件，不关闭保护来制造对照。

### 9.3 固定策略实机实验

```text
任务：Reach、Payload
方法：Iso、Iso-Low、EA
条件：Nominal、一个新几何、一个新执行或载荷条件
每组合：20 episodes
```

合计 360 回合，执行顺序交错。

指标：

```text
Safe success
无介入Safe success
真实违规与碰撞
保护介入率
动作修正幅度
最小安全距离
Capped完成时间
推理延迟与deadline miss
```

期望目标：Reach 安全成功 ≥90%、Payload ≥85%；任务基本保持时，保护介入或动作修正减少约20%。

真实硬碰撞目标为零；没有观察到碰撞不能写成理论零风险。

### 9.4 小预算真实微调

正文若保留“适应效率”主张，再做一个任务的微调：

```text
共同Piper仿真checkpoint
→ Iso-FT / Iso-Low-FT / EA-FT
每分支50个真实训练episodes
尽量2次独立适应重复
0/10/25/50节点独立评估
```

保留冻结 pretrained 参照。

训练与控制分离，Replay 记录原始动作和真实执行动作，不能把被保护层拒绝的动作当作已执行数据。

努力目标是：同样任务水平需要更少真实交互，或同预算下安全成功率提高；达不到时报告真实结果，不延长某一方法直到它获胜。

---

## 十、交付与完成标准

新增目录建议：

```text
configs/paper_crax_payload/
scripts/paper_crax_payload/
reports/paper_crax_payload/
deployment/piper/
```

至少交付：

```text
EXPERIMENT_PROTOCOL.md
CRAX_FAST_PROFILE.md
CRAX_POINTGOAL_RESULTS.csv
PAYLOAD_RESULTS.csv
ROBUSTNESS_RESULTS.csv
MECHANISM_RESULTS.csv
PARAMETER_EFFICIENCY.csv
MODEL_SELECTION.json
TRIAL_LEDGER.csv
PIPER_DEPLOYMENT_PLAN.md
PAPER_EXPERIMENT_REPORT.md
```

最终报告必须回答：

1. PointGoal 最快且有效的配置是什么，完整训练速度、return、goal、cost 分别是多少？
2. 哪些提升来自工程加速，哪些来自改变更新预算或方法？
3. 好的 EA 候选相对同样选择过的基线，是否有独立测试收益？
4. 全部 seed、固定 endpoint、失败结果是否完整保留？
5. Payload、泛化、机制与 Piper 各完成了什么？
6. 实际新增训练、模拟查询和真实交互成本分别是多少？

不要只回复“所有训练已完成”。

只做必要的任务语义、几何数值、设备转换和一次保存恢复检查。不重新审计旧18,000回合，不重跑全部历史测试，不进行无关工程重构。

**本轮优先级：**

\[
\boxed{
\text{PointGoal快速有效配置}
>
\text{Payload有效策略}
>
\text{独立泛化和机制证据}
>
\text{Piper部署候选与实机程序}
>
\text{额外环境和更多seeds}.
}
\]

有强候选就继续推进后续证据，不因个别 seed 波动停止整个研究；同时不通过隐去失败或修改测试来完成目标。

### 我建议你重点关注三个执行结果

**首先看 PointGoal 的“有效吞吐”。**最重要的不是环境能跑几十万 FPS，而是包含 EA 采样、critic 更新和 replay 的完整训练速度。上面给出的 256/512 环境配置是开发起点，不能直接当作已经验证的最优值。CRAX 官方共享配置中的 500M 步和 2048 环境也不应直接照搬到我们的 off-policy learner。

**其次看独立选择后的策略表现。**这允许充分利用好的 seed，但让 Iso、SAC 和 EA 都通过同样的验证选择过程。这样得到的好结果可以合理支撑部署价值，而不是只能作为偶然案例。

**最后看 Piper 是否真正形成可执行实验。**官方已有 Piper SDK 和机器人模型资源，Codex 应现在就准备仿真、动作接口、候选与记录程序，不必等所有公共基准跑完；但真实运动仍应由现场操作者明确启用。[GitHub](https://github.com/agilexrobotics/pyAgxArm)

JAX 与 PyTorch 共用显卡时，显存预分配也已写入任务书：JAX 默认可能预占较大比例显存，混合训练需要主动管理，而不是等 OOM 后再盲目减少所有 batch。[docs.jax.dev](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)

**这轮的核心方向就是：先调出快且有效的 PointGoal 配置，同时让 Payload 和 Piper 准备工作不断推进；把算力花在能够形成论文证据的候选和对照上，而不是再次陷入单任务长训练与反复审计。**