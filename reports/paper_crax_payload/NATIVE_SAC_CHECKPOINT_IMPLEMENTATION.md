# 原生 SAC-Lag：完整恢复与公平 checkpoint 选择接口

## 当前证据边界

新实现 `scripts/paper_crax_payload/native_sac_checkpoint.py` 已完成聚焦CPU契约测试、
AST编译与CLI帮助检查。主协调者随后批准的一次真实JAX/MJX恢复检查已完成，科学源
`dc197f333dab788c915075f05a4f6c2f81ce0ec5`：实际新增8448训练步、0评测控制、0真实交互。
**Learner/replay/RNG逐字节一致，所有canonical物理状态/观测/partial episode一致；
两个MJX派生浮点缓存在1.2e-7内不同，故不声称整个native Data逐字节一致。**
完整原始数值比较保留在 `native-sac-resume-dc197f3/comparison-bytes.json`。

已完成的旧 `native_sac_reference.py` 仍保持原样。其1,000,192步reference只保存policy/
normalizer，不具备训练恢复状态；本工具明确拒绝将其pickle当完整checkpoint恢复。
若重跑更长参考，真实重复采样必须重新计费，不继承“零成本1M前缀”。

## 科学实现不变、仅HOST epoch边界插桩

官方来源 `TTomilin/CRAX@63504b540705be98024abd675e646ff5e046c6c2`；
原生SAC-Lag train.py SHA256：
`e4d73d7d16e06ef6d50cd907b75ceba2622f108594a7b2cc483d845c1910baa4`。

源码不匹配时直接拒绝，不在运行中静默适配其他版本。工具读取官方源码，在独立
Python namespace中只编译host-instrumented train函数，不编辑官方文件，也不改变
另一个Python进程中正在运行的reference。

插桩前后对以下六个内部函数的完整AST逐一比较，必须完全相等：

```text
sgd_step / get_experience / training_step / prefill_replay_buffer
training_epoch / training_epoch_with_timing
```

因此原生loss、所有optimizer、target更新、cost critic、normalizer、dual与训练scan
保持原样。任务仍使用原生PointGoal level1和先前明确记录的CompleteReset及pre-reset
terminal observation修复，不改变episode、hazard、reward、cost或目标刷新。

`--normalize-observations` / `--no-normalize-observations` 是显式配置；默认仍为旧参考
的false，不自动启用。后续可作为一个合理的baseline调参因素，但更改后不能伪称
原配置精确续训。配置还记录JAX/Flax/Optax/MuJoCo等实际版本及关键原生模块hash，
恢复时严格匹配，不静默升级。

## 保存内容

每个host epoch完成官方评测之后，先`jax.effects_barrier()`等待异步callback，再保存：

- 完整replicated `TrainingState`：Actor、reward/cost critics与targets、三个网络optimizer、
  alpha及其optimizer、normalizer、dual/aux、env_steps和gradient_steps。
- 完整native MJX environment state，含当前观测、物理状态、reset keys和逐环境累计/
  partial episode字段，不用新reset observation伪装上条transition终点。
- replay全部数据、insert/sample位置、replay RNG和Python层size。
- host `local_key`，它决定后续epoch内的环境/Actor/SGD随机流；不是仅保存seed。
- evaluator RNG `_key`、评测次数及其累计时间；resume跳过重复initial eval。
- logger计数/buffer、Python与NumPy RNG。恢复后仅重置SPS wallclock起点以排除离线等待，
  不修改算法状态。

`full_latest.pkl` 采用临时文件、flush/fsync、原子rename。每次显式续训使用新输出目录，
旧最终full不会被续训覆盖。pickle仅允许加载本工具生成的可信本地存档，不加载未知文件。

## 真正续训与实际步数

Resume会恢复整个state并**跳过prefill和initial eval**。外层epoch的随机key分块是官方
算法的一部分，因此不能只修改target后重新计算epoch长度，否则未来随机流已改变。
本实现保留原记录的`actor_epoch_steps`，从已完成epoch继续。跨更大target必须显式
`--allow-target-extension`；缩短终点或静默改变配置会拒绝。

代价是终点可能跨过requested_steps一个旧host-epoch的量；
`actual_rounded_final_steps`、实际steps、initial_steps和真实新增步数全部明确记录。
不能把这个overshoot漏记或只给SAC/EA某一方法额外预算。正式主比较宜新建共同固定预算
的fresh run；如需共享舍入终点，要在运行前对所有方法一起冻结。旧policy-only参考不能
利用此机制补造可恢复前缀。

## 五个公平选择机会

默认声明target的20/40/60/80/100%对应最早host边界，最多五个不同eligible checkpoint。
所有checkpoint是按预算/时间表选取，完全不读取表现。中间额外host epoch只做恢复保存，
不能自动增加候选或调用更多选择评测。

每个eligible点保存`policy_normalizer_STEP.pkl`，带实际步骤与epoch。官方评测保留为
diagnostic；共同61027等layout validation应在这些固定五个候选上离线统一进行。
**保存了五个policy不等于已经完成共同验证或独立测试。**

扩展target时仍只允许新预算表中的五个候选。如果它要求一个早已过去但没有保留的
policy snapshot，程序直接拒绝，不能凭最终权重倒造中间checkpoint；应改成明确计费的
fresh run。所有旧候选仍保留历史记录，不因为选择范围变化而删除。

## 一次实际小型真实检查（已批准完成，不再重跑）

使用相同seed和工程设置，比较完整run与“host epoch1保存停止→新进程完整恢复”的最终
state。默认任务物理/horizon仍不改变；只缩小E/replay/batch和显式工程host粒度。

```text
--engineering-check --skip-native-evaluation --epoch-actor-steps 32
--steps 4096 --num-envs 8 --batch-size 16 --updates 1
--min-replay-size 128 --max-replay-size 8192 --seed 99027
```

分段第一段加`--stop-after-epochs 1`，第二段用新输出目录加
`--resume FIRST/full_latest.pkl`；整段与两分段的每个叶子应比较：所有网络/optimizer/
normalizer/dual、replay及RNG、nativeenv state、outer/eval keys和计数。
Wallclock/logger计时不同不作为数值失败。真实采样预期每条完整lineage4224步，
两条lineage合计8448步，不把恢复对照的重复采样计作零；以实际receipt为准。

实际三进程都使用同一dc197f3文件，依次完成：

| 分段 | 原生步骤 | 真实新增训练 | 梯度更新累计 | 进程耗时 |
|---|---:|---:|---:|---:|
| uninterrupted |0→4224|4224|512|182.888秒|
| prefix，显式host停止 |0→384|384|32|96.209秒|
| resumed，新进程恢复 |384→4224|3840|512|92.160秒|
| **实际总新增** |两条完整lineage|**8448**|实际总1024|约371.257秒|

没有自动重启或额外物理试验。三条receipt的评测均0；共同validation没有被工程检查
冒充完成。两条lineage训练原始成本均699.019287，真实合计1398.038574；分段续训只计
699.019287−47.266903的新增成本，不把prefix累加两遍。该4k工程试验不是策略效果基线。

### 全状态比较，而非只看policy

实际保存的130个TrainingState叶子（所有网络、optimizer、normalizer、dual、计数）、
4个replay叶子（含完整轨迹和key）、outer local_key与evaluator_key在完整/分段终点逐字节
相同。所有候选实际steps相同：896、1664、2688、3456、4224；各恰好五个机会。

环境141个叶子中139个逐字节相同，包括qpos/qvel、观测、partial成本、reset keys等。
仅`.pipeline_state.subtree_com`差1.1920929e-7、`._impl.subtree_angmom`差1.39698e-9，
均在预先比较容差rtol1e-6/atol1e-7内。当前MJX3.11源码`forward.py:70–74`
调用kinematics→com_pos；`smooth.py:140–161`从xipos/body_mass重新计算subtree_com，
`smooth.py:456+`重新计算subtree angular momentum。这些是派生缓存；低位差原因与
不同编译/浮点归约一致，但并未直接审计HLO来证明某个唯一原因。没有删去这两个叶子，
原始严格`exact_resume_contract_passed`仍为false，数值容差合同为true。

另外dc197f3最初没有固定未使用的Python/NumPy全局RNG，故独立fresh进程的这些host状态
不相同。恢复后的两个host RNG与**其自身prefix**存档逐项相同；官方算法使用显式JAX keys，
这些算法key在两条lineage完全一致。后续版本已前瞻性增加host seed，并记录Reset wrapper
文件hash；没有更改这三条正在运行时的源，也没有为美化字节比较而额外重复8448步。

### 大文件存档

三个完整pickle已从A100流式复制到4080 4T：
`/home/carla/4T/SAFEFLOW2027_PAPER/archives/native-sac-resume-dc197f3/`。
每个约10.79MB，源/目的SHA256逐文件一致；必要A100原件保留，未删除任何旧记录。
GitHub只放JSON、日志与代码，不放这些full pickle。详见同目录紧凑
`ARCHIVE_MANIFEST.json`。
