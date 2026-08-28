# SAFEFlow 单 GPU 论文核心实验：正式结果与进度

更新时间：2026-08-28 19:26（Asia/Shanghai）

## 当前结论

PointGoal 四方法正式比较已经完成。PointCircle 原始配置及协议唯一允许的几何增强都已完成 `300k` 单 seed 筛选，并按完全相同的冻结规则判定为未通过；PointCircle 方法修改与权重搜索已经停止。PointGoal 所有方法使用训练 seeds `2027–2029`、每 seed `200k` 环境步，并对每个最终 checkpoint 使用 `5` 个固定基础种子、每个 `20` 个连续 episodes，共 `100` episodes。独立统计单位是训练 seed（`n=3`），不是合并后的 episodes。

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

## PointGoal checkpoint-only 鲁棒性结果

冻结的 PointGoal 鲁棒性矩阵已经完成。它直接读取 C0/P4.7 的 `3 × 2` 个 `200k` checkpoint，没有重新训练策略。nominal 和 medium 条件各使用每 checkpoint `50` episodes；预先指定的 strong 条件最终各使用 `100` episodes。统计单位仍是训练 seed（`n=3`）。

| 条件 | Episodes/checkpoint | C0 / P4.7 Return | C0 / P4.7 Mean Cost | C0 / P4.7 CVaR-20 | C0 / P4.7 V_N/V_T | C0 / P4.7 Outward |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 50 | 0.2764 / **0.3404** | **11.300** / 18.880 | **56.233** / 92.667 | 0.9928 / **0.8426** | 0.23587 / **0.17045** |
| action noise medium | 50 | 0.3398 / **0.3640** | 17.947 / **11.853** | 86.200 / **57.067** | 0.9979 / **0.8427** | 0.22429 / **0.17399** |
| action noise strong | 100 | 0.3289 / **0.3979** | 16.683 / **12.143** | 79.750 / **60.050** | 0.9930 / **0.8625** | 0.24613 / **0.18534** |
| hazard radius medium | 50 | 0.1677 / **0.2933** | **20.020** / 20.307 | 95.667 / **94.033** | 0.9994 / **0.8948** | 0.21004 / **0.17648** |
| hazard radius strong | 100 | 0.1516 / **0.2160** | 25.237 / **24.877** | 110.933 / **108.617** | 0.9977 / **0.8522** | 0.21856 / **0.17016** |
| hazard position medium | 50 | **0.4501** / 0.3972 | **11.720** / 17.193 | **56.667** / 84.300 | 0.9979 / **0.7752** | 0.23126 / **0.18509** |
| hazard position strong | 100 | 0.4982 / **0.5019** | **13.287** / 19.140 | **62.017** / 93.350 | 0.9996 / **0.8076** | 0.23186 / **0.19458** |

结果支持三条有边界的结论：

1. 方向机制在扰动下非常稳定。P4.7 的高风险 `V_N/V_T` 和危险外向速度在全部 `7/7` 个聚合条件以及逐 seed 的 `21/21` 个配对中都低于 C0。
2. 动作噪声下形成最清楚的鲁棒安全收益。medium 条件的 return 提高 `7.1%`、mean cost 降低 `34.0%`、CVaR 降低 `33.8%`；strong 条件分别为 `+21.0%`、`-27.2%`、`-24.7%`，且 cost/CVaR 在两个条件中均为 `3/3` seeds 更低。扩大 hazard 半径时，P4.7 return 提高 `74.9%/42.5%`，强扰动的 mean cost 与 CVaR 分别小幅降低 `1.4%/2.1%`。
3. 安全收益不是任意分布偏移下都成立。整体平移 hazards 时，方向几何仍稳定激活，但 medium/strong 的 mean cost 分别高 `46.7%/44.1%`，CVaR 高 `48.8%/50.5%`。这说明当前 critic 学到的安全边界法向对动作噪声和半径变化较稳健，但对障碍物整体位置变化仍存在泛化缺口。

鲁棒性协议中的 nominal 使用独立冻结评测 seed 集和每 checkpoint `50` episodes，只是同一扰动协议的参考点，不替代前述 `100` episodes PointGoal 正式主结果。整个矩阵是趋势分析，不是新选择门槛，也不构成每 seed Pareto 声明。

- [完整鲁棒性报告](single_gpu_paper_core_artifacts/pointgoal/robustness/final/POINT_ROBUSTNESS_REPORT.md)
- [聚合指标](single_gpu_paper_core_artifacts/pointgoal/robustness/final/point_robustness_summary.csv)
- [逐训练 seed 指标](single_gpu_paper_core_artifacts/pointgoal/robustness/final/point_robustness_seed_metrics.csv)
- [最终 manifest](single_gpu_paper_core_artifacts/pointgoal/robustness/final/point_robustness_manifest.json)
- [A100 六路并行工程披露](single_gpu_pointgoal_robustness_a100_parallelism_amendment_2026-08-27.json)

## 当前运行状态

`2026-08-28 19:26` 的按需只读快照显示，RTX 3090 上 C0/P4.7 seed `2028` 分别到达 `1.785M/1.780M`，即 `89.25%/89.00%`；累计吞吐为 `6.76/6.82 FPS`，两条训练预计均还需约 `9` 小时。A100 上 C0/P4.7 seed `2029` 分别到达 `225k/220k`。训练、checkpoint 迁移协调器与终态证据搬运器均保持运行；没有执行新的方法修改、权重搜索或 checkpoint 评测。

seed `2028` 当前 progress row 中保留的中期评测为：

| 配置 | Steps | Return | Mean Cost | CVaR-20 | 训练累计 Cost |
|---|---:|---:|---:|---:|---:|
| C0 / isotropic FLAC | 1.785M | 4115.4390 | 0.270 | 1.000 | 886 |
| P4.7 / Ours | 1.780M | **5596.7417** | **0.060** | **0.300** | **570** |

在这一个训练 seed、近似配对步数的中期观察中，P4.7 相对 C0 的 return 高 `36.0%`，mean cost 低 `77.8%`，CVaR-20 低 `70.0%`，训练累计 cost 低 `35.7%`。这些数值与 PointGoal 的安全收益方向一致，但它们不是冻结终点，也没有用于选择配置；Ant 正式结论仍只依据 C0/P4.7 三 seeds 的 `2M` checkpoint 与固定 `100` episodes 评测。完整机器可读观测见 [Ant 运行时与中期结果快照](single_gpu_ant_runtime_observation_2026-08-28.json)。

`2026-08-29 01:31` 起，Seeta RTX 3090 的 SSH 入口持续返回 `Connection refused`。中断前最后一次可认证的轻量快照在 `01:00`：C0/P4.7 seed `2028` 分别到达 `1.930M/1.920M`。主机协调器已增加网络中断容错：Seeta 不可达时保持等待；恢复后要求原队列控制器仍匹配，并对每个 wave-one arm 接受“冻结身份校验通过的有效 receipt”或“匹配且存活的 learner”二者之一。后一个 receipt-aware 分支避免断网期间已正常完成并退出的 learner 被误判为丢失，从而不会因缺少已完成进程而空等；它没有改变或重启训练。两个严格沿用冻结 source/config/seed 的 replay-checkpoint 恢复 launcher 已在主机准备，但截至本记录尚未部署或启动。A100 两条 seed `2029` 不受影响，`05:20` 均达到 `330k/2M`。协调器在等待 Seeta 时每 `300` 秒执行一次 A100 工程存活保护：仅当有效 receipt、预期 tmux session 与匹配 learner 均不存在，并且冻结 source/config 身份重新认证通过时，才调用原 replay-aware launcher；当前两条 learner 均被识别为正常，未触发重启。`05:26` 容量检查显示两张 A100 分别为 `100%/88%` GPU 利用率且各占用约 `66/82 GB` 显存，CPU 一分钟负载约 `131`，因此没有追加第三个 learner，避免 OOM 或降低现有总吞吐。这是工程中断披露，不含新评测、方法修改、配置选择或科学提前停止。

PointCircle 唯一一次几何增强已于 `2026-08-25 23:14` 完成全部 `300k` 环境步和固定 `20` episodes 终点评测，端到端吞吐为 `7.808 FPS`。增强版 return 为 `16.5585`，高于冻结 C0 的 `11.3807`；两者 mean cost/CVaR-20 都是 `0/0`。增强版危险外向速度从 C0 的 `0.10258` 降至 `0.09538`，但高风险 `V_N/V_T=0.999997`，未达到 `<=0.9`。因此冻结决定为 `return_ok=true`、`safety_ok=false`、`geometry_ok=false`、`screen_pass=false`。

`150k` 固定评测曾确认增强配置在高风险样本上形成明显各向异性：`V_N/V_T=0.7598`。该节点的 return 为 `16.2763`、mean cost 为 `8.45`、CVaR-20 为 `27.25`、训练累计 cost 为 `7208`，但危险外向速度 `0.1427` 高于冻结 C0 在同一节点的 `0.0965`。因此增强的几何作用是暂时激活而非终点稳定激活；学习曲线证据保留，但不能覆盖 `300k` 冻结决定。

PointCircle 三 seed `500k` 正式矩阵、prior baseline、机制控制、控制后处理及原组合鲁棒性队列都已传播 `screen_not_passed` 终态，`scientific_training_launched=false`，没有消耗任何下游 PointCircle 训练预算。PointCircle 鲁棒性因此取消。现有三 seed PointGoal checkpoint 的 checkpoint-only 鲁棒性已在 A100 CPU 上完成，状态为 `complete`、`retraining=false`；六份 checkpoint 结果与最终 manifest 已经同步回主机及原 RTX 3090 run root。

Point 核心达到协议终态后，Ant 二级扩展于 `2026-08-25 23:16` 自动启动。`2026-08-27 20:05` 的按需快照显示首轮 C0 seed `2028` 到达 `1.180M/2M`、P4.7 seed `2028` 到达 `1.175M/2M`，累计吞吐分别为 `7.309` 与 `7.243 FPS`；两个 learner 均正常，首轮训练剩余约 `32` 小时。

用户随后增加了一个 `2 × NVIDIA A100 80GB PCIe` 节点。为缩短 wall-clock，`C0-2029` 与 `P47-2029` 已于 `2026-08-27 20:19–20:25` 分配到两张 A100；原 RTX 3090 当前配对保持原样，完成后只运行 `C0-2027`。原队列仅暂停控制器进程以避免重复启动，当前两个 learner 没有暂停；持久化主机协调器会在三条分布式运行完成后将 A100 结果经主机同步回原 run root，再恢复原严格 finalizer。

`2026-08-27 21:48`，两条 A100 任务同时形成首个预先约定的 `20k` 稳态吞吐节点：C0 为 `3.384 FPS`，P4.7 为 `3.542 FPS`，合计 `6.926 FPS`。在服务器既有高负载作业持续不变的保守假设下，较慢任务距离 `2M` 还需约 `162.5` 小时，即 A100 配对预计于 `2026-09-03 16:20` 左右完成。该节点仅是工程吞吐观测，不包含评测结果，也没有用于方法或超参数选择。

`2026-08-28 00:24` 的轻量快照显示，RTX 3090 上 C0/P4.7 seed `2028` 分别达到 `1.300M/1.290M`，累计吞吐约 `7.45/7.47 FPS`，当前配对还需约 `26.5` 小时。A100 两条 seed `2029` 均达到 `45k`；共享节点的瞬时吞吐受既有高负载任务及刚结束的 CPU 鲁棒性评测影响，不据此重新拟合科学计划。结合已部署的 work-conserving fallback，当前完整 Ant 终态仍保守按约 `6–8` 天估计。后续不做高频进度校验，只检查终态 receipt。

为避免共享 A100 持续拥堵成为单点关键路径，主机协调器在不停止任何 learner 的前提下增加了 work-conserving fallback：RTX 3090 完成 `C0-2027` 后，如果 A100 配对仍未全部形成有效 receipt，就立即恢复原冻结队列；已从 A100 验证并同步的 arm 会被原队列跳过，其余 arm 在空闲 RTX 3090 上与 A100 分支竞速，最终只接受冻结身份校验通过的 receipt。若 A100 任一任务发生工程失败，也沿用原顺序队列兜底。完整机器可读观测见 [A100 20k 运行时观测](single_gpu_ant_a100_20k_runtime_observation_2026-08-27.json)。

为利用 A100 节点的 `128` 个 CPU cores，同时避免两张已满负载 GPU 上的额外训练竞争，PointGoal checkpoint-only 鲁棒性矩阵在任何新鲁棒性 episode 产生前分配到该节点的 CPU。它只读取现有 C0/P4.7 三 seed、`200k` checkpoint，保持冻结扰动、评测 seeds、episode 数和 fixed-random-flow 语义，不训练或修改策略。原始控制器先配对运行 seed `2027`；在尚无任何完成结果时，为响应加速要求，控制器被暂停但其两个子进程继续，seeds `2028/2029` 的四个相同 evaluator 同时启动，使工程并发从 `2` 增至 `6`。所有 evaluator 使用同一 source commit、协议、CPU device、checkpoint 和输出格式；结束后原控制器恢复并逐份认证，再运行未修改的 finalizer。机器可读注册与事后并发披露分别见 [A100 运行时补充记录](single_gpu_pointgoal_robustness_a100_runtime_amendment_2026-08-27.json) 和 [六路并行工程披露](single_gpu_pointgoal_robustness_a100_parallelism_amendment_2026-08-27.json)。

A100 CPU 队列的前两次工程启动都在 evaluator 导入阶段退出，完成 robustness episodes 为 `0`，没有产生可选择的科学结果；原 RTX 3090 等待控制器均由 fallback 恢复。直接复现确认根因是队列对 venv `python-bin` 使用 `Path.resolve()`，将其符号链接解析为不含 venv site-packages 的基础 Anaconda Python。修复只将解释器路径绝对化而不解析符号链接，默认设备与既有调用保持不变，并加入针对该行为的回归测试。失败目录完整归档，不删除或伪装为科学运行。机器可读记录见 [A100 鲁棒性解释器工程修复](single_gpu_pointgoal_robustness_a100_interpreter_repair_2026-08-27.json)。

这次变更只涉及运行时资源分流：训练 seeds、每条 `2M` 步预算、两种配置、历史训练源 commit、`vector_env_nums=1`、`UTD=1`、checkpoint 与 `100` episodes 固定评测全部不变。A100 节点因驱动约束使用 PyTorch `2.7.1+cu118`，RTX 3090 使用 `2.8.0+cu128`；该软件/硬件差异作为运行时 nuisance variable 明确披露，不隐藏为同质单机结果。分流在 A100 尚未产生首个 progress row 或任何科学评测结果时锁定。完整机器可读注册见 [Ant 分布式运行时补充记录](single_gpu_ant_distributed_runtime_amendment_2026-08-27.json)。

PointGoal 鲁棒性等待控制器启动时曾因脚本路径形式导致一次 import 阶段工程退出，零 evaluator、零科学步；改为模块入口后已正常等待，该失败记录保留在服务器 controller log 中。等待控制器已进一步升级到来源 commit `23d09f8025ca98a5d94ba0e35f80f2f46e5a23ac`：未来任一 evaluator 或 finalizer 的工程异常都会原子写入 `failed` 终态并由主机搬运，避免永久停留在 `evaluating`。此次替换没有启动 evaluator，也没有触碰 Ant learner。

主机终态证据搬运器已将增强筛选、全部 PointCircle 条件队列以及 PointGoal 鲁棒性终态轻量证据同步到主机，并为每个文件生成 SHA-256 receipt。PointGoal 鲁棒性包包含 `30` 个 portable files、约 `1.435 MB`，不包含 checkpoint 或训练日志；六份原始 evaluation JSON、最终 CSV/Markdown、运行配置 manifest 与终态 identity 均保留。该服务继续监控 Ant 终态，主机重启后会自动恢复，也不会直接修改 Git 工作树。

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
- [PointGoal 鲁棒性完整报告](single_gpu_paper_core_artifacts/pointgoal/robustness/final/POINT_ROBUSTNESS_REPORT.md)
- [PointGoal 鲁棒性原始评测与 SHA receipt](single_gpu_paper_core_artifacts/pointgoal/robustness/stage_receipt.json)
- [全实验完成状态审计](single_gpu_paper_core_completion_audit.json)

主机同步目录只包含 JSON、CSV 和 Markdown。checkpoint、训练数据、日志和源代码工作树均未纳入 GitHub；仅保留轻量 evaluator 配置与 run manifest 以复核运行身份。

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

增强实验固定为 seed `2027`、`300k`，复用哈希锁定的原始 C0 `300k` 评测，没有重跑基线、增加第二个候选或科学提前停止，并使用完全相同的筛选门槛。最终 `screen_pass=false`，因此 PointCircle 方法修改和权重搜索已经按协议停止。

`150k` 开发里程碑结果如下。增强版相对冻结 C0 的 return 提高 `372.4%`，mean cost 降低 `50.3%`，CVaR-20 降低 `67.9%`，训练累计 cost 降低 `19.1%`，高风险 `V_N/V_T` 从 `0.9991` 降至 `0.7598`。由于增强版危险外向速度在该节点更高，几何分支不能单独据此判定通过；安全分支在这个中间节点满足门槛。正式决定仍只使用 `300k` 终点。

| 里程碑 | 配置 | Return | Mean Cost | CVaR-20 | 训练累计 Cost | 高风险 V_N/V_T | 危险外向速度 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 150k | C0 / isotropic FLAC | 3.4453 | 17.0 | 85.0 | 8913 | 0.99908 | **0.09655** |
| 150k | 原始 P4.7 | **28.7077** | **0.0** | **0.0** | **6068** | 1.00000 | 0.01738 |
| 150k | 几何增强 P4.7 | 16.2763 | 8.45 | 27.25 | 7208 | **0.75976** | 0.14266 |

冻结终点结果如下：

| 里程碑 | 配置 | Return | Mean Cost | CVaR-20 | 训练累计 Cost | 高风险 V_N/V_T | 危险外向速度 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 300k | C0 / isotropic FLAC | 11.3807 | 0.0 | 0.0 | **9788** | 1.00000 | 0.10258 |
| 300k | 原始 P4.7 | 13.9828 | 0.0 | 0.0 | 11076 | 1.00000 | **0.03673** |
| 300k | 几何增强 P4.7 | **16.5585** | 0.0 | 0.0 | 10796 | 1.00000 | 0.09538 |

增强版相对 C0 的 return 提高约 `45.5%`，危险外向速度降低约 `7.0%`；但安全相对改善为 `0`，高风险 `V_N/V_T` 也未达阈值。最终门槛逐项为：

| 条件 | 结果 |
|---|---|
| Return 下降不超过 10% | 通过 |
| Mean cost 或 CVaR-20 改善至少 10% | 未通过 |
| `V_N/V_T <= 0.9` 且 outward 更低 | 未通过 |
| 总门槛 | **未通过** |

- [增强预注册](single_gpu_pointcircle_geometry_enhancement_preregistration.json)
- [增强配置](../../configs/star/single_gpu_pointcircle_p47_geometry_enhanced.yaml)
- [增强版 150k 数值证据](single_gpu_paper_core_artifacts/pointcircle/geometry_enhancement_development_evidence/P47/evaluation_step_150000.json)
- [增强版 300k 冻结报告](single_gpu_paper_core_artifacts/pointcircle/geometry_enhancement_terminal/POINTCIRCLE_GEOMETRY_ENHANCEMENT_300K_REPORT.md)
- [增强版机器可读冻结决定](single_gpu_paper_core_artifacts/pointcircle/geometry_enhancement_terminal/enhancement/screen_decision.json)

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
