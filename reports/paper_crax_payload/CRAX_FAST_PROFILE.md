# CRAX完整训练吞吐：Fast批处理复核完成，尚非“最快有效策略”结论

四条完整工程profile的原始证据在 [native_runs](native_runs/)；不重新采样或追加profiling。
三个v2来源为 `f6731395bc5e2b06eed4641ab4242f6fd0b73a6e`，
一次Fast批处理v3来源为 `cec7e5dfae0a2053d54a66e4dcbe9825987b80ea`。
每条32768个真实transitions，均为EA全几何热路径；完整四条合计131072个新增工程训练
transitions，计入24M开发预算。此前失败v1、设备语义检查及其他训练另由总账保留，
**131072不是整个CRAX开发的总开销**。

## 实际结论

当前最快的**已测计算配置**是 Fast批处理v3：E512、F1、B1024、G4，
末段完整训练吞吐 **2481.83 transitions/s**，仅就稳态热路径达到约2000SPS；
runner全程837.75SPS仍低于2000，不能说整体耗时目标已完全达成。
没有达到5000SPS，也不能把这次短profile称为已证明长期有效的强策略。
它在同一Fast更新预算下观察到较v2末段快2.294倍；包含初始化/首次编译/保存的runner
计时吞吐只提高1.215倍。两者都必须报告，不能只选较大的倍率。

所有profile训练不足完整原生episode：Fast每环境64控制步、Balanced128、Heavy256，
原生horizon为1000，实际complete episodes均为0。因此本表没有可信的
Episode Return、Episode Cost、Goals/Episode或满足预算比例；这些不是0，而是未测。
不能将零完整episode的工程profile替代500k及后续真正学习与独立评估结果。

## E/B/G、末段与完整计时分别报告

| 实际run | E / B / G | 更新/transition u | replay exposure | 末段steady SPS | runner全程SPS | runner计时秒 |
|---|---:|---:|---:|---:|---:|---:|
| Balanced v2 |256 /1024 /4|0.015625|16|491.56|530.54|61.763|
| Fast v2 |512 /1024 /4|0.0078125|8|1081.97|689.58|47.519|
| Learn-heavy v2 |128 /512 /8|0.0625|32|118.59|277.10|118.251|
| Fast批处理 v3 |512 /1024 /4|0.0078125|8|2481.83|837.75|39.114|

F全部为1。E/B/G改变既影响速度也改变更新预算，三个v2之间不是纯工程对照；
Fast v2与v3的E/B/G、采样数、损失与更新计数相同，改动是两个头的batched VJP与
target Actor的9B联合采样，见 [批处理优化](CRAX_BATCHING_OPTIMIZATION.md)。
随机数消耗顺序不同，并非逐bit轨迹一致；单次共享GPU测量不提供固定倍速保证或统计显著性。

`steady_training_sps` 是progress最后一个真实记录窗口的采样+learner吞吐，
不是整个科学训练平均值；它排除该窗口之外的首次编译、评估/保存和日志开销。
`runner全程SPS` 使用receipt的
`full_training_including_compile_eval_save_sps=new_training_transitions/elapsed_seconds`。
此计时从main内部初始化开始，到最终full保存完成，包含初始化、reset、首次step编译、
训练和已发生的eval/save/周期日志；**不含Python进程启动/import、参数解析及最后receipt输出**，
不是外部秒表测得的完整进程生命周期。不能把两种scope混称为同一个end-to-end速度。

## 编译阶段、保存和完整计算组成

| run | 初始化/reset阶段秒 | 首次step阶段秒 | 两阶段合计秒 | eval秒 | save秒 | actor ms/vector | env+bridge ms/vector | replay+learner ms/vector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Balanced v2 |9.827|11.834|21.661|0|0.132|25.082|17.584|449.615|
| Fast v2 |9.842|11.834|21.676|0|0.175|24.970|19.133|420.889|
| Learn-heavy v2 |10.601|13.860|24.462|0|0.079|19.895|16.897|662.924|
| Fast批处理 v3 |8.945|11.834|20.780|0|0.139|17.782|19.830|166.690|

原始字段名为 `reset_compile_seconds` 和 `first_step_compile_seconds`。
第一阶段还含Python内部对象创建、learner/replay初始化与reset等，不是XLA纯编译器耗时；
这里明确报告测量阶段，不把两阶段合计包装为纯编译延迟。四次均使用记录中的同一路径
JAX compilation cache，不能假装每次都是独立清空缓存的冷启动。

三个计算分量来自CUDA events中位数，记录窗口分别有49、24、99、24次事件；
replay+learner包括该vector step后的G轮更新，不是单一optimizer step延迟。
Fast v3的该部分420.889→166.690ms，观察到2.525倍缩短；env/bridge没有提速。
这是包含Actor、实际几何、replay、critic和更新的训练测量，不是仿真FPS。
也不能把E512的actor批量延迟除以512，冒称单动作p50/p95。

## optimizer steps，不把learner rounds/s冒充更新速度

令U为learner rounds，A为实际Actor updates。当前每round有margin、cost、
两次reward optimizer step，共4U；每次Actor更新还包括Actor和alpha两个step，
所以实际optimizer总数为 **4U+2A**，target/EMA Polyak不是optimizer step。

| run | U / reward / A / margin | 实际optimizer steps | 末窗rounds/s | 末窗optimizer steps/s | runner全程optimizer steps/s |
|---|---:|---:|---:|---:|---:|
| Balanced v2 |200 /400 /100 /200|1000|7.681|38.403|16.191|
| Fast v2 |100 /200 /50 /100|500|8.453|42.265|10.522|
| Learn-heavy v2 |800 /1600 /400 /800|4000|7.412|37.058|33.826|
| Fast批处理 v3 |100 /200 /50 /100|500|19.389|96.947|12.783|

progress的 `updates_per_second` 实际是rounds/s。本表末窗optimizer率按该窗口
实际ΔU、ΔA计算 `rounds_per_second*(4+2*ΔA/ΔU)`，没有把Actor间隔简化成额外训练预算。
全程optimizer率使用receipt计数除以runner计时；预热、初始化和保存也计入分母。
Fast v3没有通过减少更新量获得这次加速。

## Torch显存与共享整卡占用不可混同

| run | Torch peak allocated MiB | 整卡GPU0采样used MiB区间 | 整卡GPU1采样used MiB区间 | 整卡GPU1采样util区间 |
|---|---:|---:|---:|---:|
| Balanced v2 |592.468|3276–3412|2528–3438|44–88%|
| Fast v2 |592.469|3276–3412|3304–3440|49–89%|
| Learn-heavy v2 |559.340|2107–2107|3252–3332|67–96%|
| Fast批处理 v3 |674.730|2632–2632|3304–3540|63–94%|

Torch peak是本进程Torch allocator的 `max_memory_allocated`，不是reserved，
不包含全部JAX、CUDA context或其他项目；整卡数值是各run四次nvidia-smi采样的
两张80GiB A100总used/util，不是本训练进程自己的GPU利用率，也不是连续采样得到的绝对峰值。
不能用整卡used减Torch peak推算其他项目显存。9B批处理的Torch peak增加约82.26MiB，
没有OOM；共享负载不同仍是速度比较的混杂因素。没有停止其他项目来制造独占结果。

## 任务语义与后续配置边界

实际固定CRAX `63504b540705be98024abd675e646ff5e046c6c2`，
`safe_goal_point` difficulty1，原生62维状态、2维动作、horizon1000，
物理步长约0.02、frame skip4、控制周期约0.08；16个原生hazards，原生目标刷新、
reward/cost全部保留，没有首次违规终止。Cost-Q是未限幅的折扣累计成本，预算25，
几何标签仅为endpoint-clearance proxy，不冒充真实控制区间最小距离或cost概率。
2维动作的参考energy target为12.5，不是Panda的18.75。

这些profile使用观测identity/none，profile-only显式关闭critic预热并强制beta1，
仅为测昂贵的EA完整几何路径；不伪造margin已训练5000更新。真正候选恢复注册的
1000critic-only与1000→5000有效margin-update ramp。新候选若显式改变normalizer或
dual，必须另行记录、对基线公平并保留旧结果，不能将学习差异归于本次执行批处理。

四条均记录 `float32_matmul_precision=high`，允许支持硬件上的TF32内部矩阵运算，
保留float32输出；不是highest的逐bit等价，也不是独立测得的TF32因果加速。
[PyTorch2.7 API](https://docs.pytorch.org/docs/2.7/generated/torch.set_float32_matmul_precision.html)。
JAX precision单独记录；既有Arm/Payload任务未因此修改。
训练使用已实际验证的JAX–Torch同设备转换和新环境cuDNN loader，
`XLA_PYTHON_CLIENT_PREALLOCATE=false`，不改旧环境。

机器表见 [PARAMETER_EFFICIENCY.csv](PARAMETER_EFFICIENCY.csv)：保留全部56条旧的
冻结模型/单动作记录，新增这4条engineering-profile行及结构化scope字段。
CRAX单动作p50/p95没有在本次profile测量，相关单元格保持空白，不填推算值。
后续按已注册候选方案真正训练、完成原生episode，再回答“快且有效”而非仅“快”。
