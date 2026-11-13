# CRAX：一次针对 learner 的批处理优化

状态补记：下文保留实现当时的计划与验证记录。一次同配置 Fast v3 复核现已完成，
实测末段2481.83SPS、runner全程837.75SPS；完整范围和限制见
[最终profile记录](CRAX_FAST_PROFILE.md)。这里的“等待复核”不再是当前待办，
不得据此重复启动profiling。

## 实际瓶颈，不是环境FPS

优化前来源是 `f6731395bc5e2b06eed4641ab4242f6fd0b73a6e`，三个
`native_runs/crax-profile-*-v2` 的真实 receipt/progress 均保留，每个32768个真实
工程采样transitions。它们只是短profiling，completed episodes均为0，不是有效策略结果。

| 原profile | 末段steady SPS | receipt端到端SPS | actor ms/vector step | env/bridge ms/vector step | replay+learner ms/vector step |
|---|---:|---:|---:|---:|---:|
| Balanced |491.56|530.54|25.08|17.58|449.61|
| Fast |1081.97|689.58|24.97|19.13|420.89|
| Learn-heavy |118.59|277.10|19.90|16.90|662.92|

末段steady与receipt全程计时不能混用；短profile还有预热、编译及共享GPU负载影响。
Fast/ Balanced/ Heavy 本身改变E/B/G与replay exposure，因此不是纯工程加速对照。
本次固定同一Fast配置，只改下面的执行批处理；GPU提速幅度须等一次同配置复核，
不能提前声称达到2000SPS或5000SPS。

## 两处同一语义的批处理

1. `UpdateGeometry.query` 一次 batched VJP 同时求两个margin头的梯度：
   `autograd.grad(head_sums, variable, grad_outputs=eye(heads), is_grads_batched=True)`。
   每个头仍独立保留，不能用平均头梯度替代集成一致性。实际同一个冻结generator的
   deterministic tail、真实clamp导数、输入梯度、strict-normal阈值、margin sigmoid、
   head agreement、一次beta ramp、停止梯度均不变，不计算Hessian。
   仅在请求时构建诊断张量；请求诊断时字段及值与原实现一致。

2. 每个learner round将同一冻结target Actor和EMA geometry的三次采样合为一次9B：
   第一个B用于cost，后两个4B分别用于两次reward更新。仍然是1+4+4个独立动作样本，
   不是复用一组样本给两次reward TD。target Actor只在原有Actor更新后Polyak，
   geometry只在round末更新EMA，所以整轮内提前采样这9组不改变其条件分布。

特别保留时序：**第二次reward target重新调用已经过第一次Polyak更新的Qt**。
只预采样动作/energy，不预计算两轮Qt值。cost target仍在cost critic更新前计算。
margin标签、损失、原始reward/cost单位、batch size、G、每round两次reward更新、
四target样本、Actor间隔、dual、ramp和采样/energy共同参考矩阵不变。

合并之后随机数消耗顺序不同，所以不声称与旧checkpoint逐bit轨迹一致。
新 `learner.settings['batching_version']='head_vjp_target9_v1'` 随checkpoint保存；
旧settings缺少该字段会被原有exact-config guard拒绝，须使用新配置/新输出路径，
不能把旧profile或旧状态静默续成新版本。

`FlowActor.forward(noise_increments=...)` 仅增加显式共同噪声入口供数值比较，
验证K/batch/action形状、设备和dtype。正常训练仍使用原来的随机流，不搬运到CPU。
历史Arm learner/geometry与旧结果均未修改；warmup缺少loss时的metrics修复保留。

## 必要CPU证据，0新增物理采样

`python -m pytest -q -p no:cacheprovider tests/test_crax_batching.py`：
**6通过，0失败，0.60秒**，无GPU任务、环境构造或模拟控制：

- 三种geometry模式在部分ramp、真实越界clamp下，batched VJP的normal、weight及
  全部诊断与原 `EndpointGeometry` 在浮点容差内一致；参数requires-grad状态恢复，
  geometry输出停止梯度。
- 动作维度2和6、同一冻结Actor和共同噪声下，9B与原1B/4B/4B分块的动作、energy、
  clipping、weight一致；包含触发clipping的大噪声，错误噪声形状被拒绝。
- 一次合成CPU learner round仅调用一次9B target Actor，仍有两次reward optimizer
  更新和一次margin更新；第二Qt输入前的参数等于第一轮Polyak后的值而非旧快照。
  warmup没有伪造loss，旧未标版本settings不能假装精确恢复。

这些是有限数值/时序契约，不是学习性能或GPU速度结果。后续仅由主流程做一次
Fast32768真实工程复核，再进入500k候选；真实采样由各自receipt完整计入24M开发预算。
