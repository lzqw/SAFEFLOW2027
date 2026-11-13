# 低优先级：两个真正训练的Arm rho参数点（仅准备，尚未admit）

本登记不表示已经训练。优先级严格低于PointGoal有效候选、主要Payload与Piper候选；
主协调者释放容量并显式admit后才能执行。没有新服务器进程、训练、模拟查询或git提交。

## 范围与实际训练预算

仅新增EA rho=.1和.5，各从头训练seed2027到固定1M，总新增训练上限2M。
这属于master的**Arm机制/参数/小额修复3M**，不计入CRAX六个配置名额；剩余1M不是
自动重试或增加第三个参数点的授权。失败/工程丢弃尾部必须额外如实记账。

历史EA rho=.25和Iso有效rho=1复用原固定1M checkpoint，不重训、改名或恢复它们。
历史seed2027是已知强例，选定在新参数结果产生前；这是单seed条件性参数证据，不能
伪装为多个新种子稳定性结论，也不得在已经冻结的默认/正式实验选择中事后改选rho。
先前rho/K结果仍标记**frozen sampler、未重训**，不会因本计划而变成trained结果。

## 已检查的真实接口：必须300k→1M，不存在fresh-1M开关

现有 `scripts.star.arm_safe_entropy.train_stability` 明确拒绝不带resume的1M训练。
合法且与旧recipe相同的路径是：

| 顺序 | 点 | 原生CLI终点 | 初始化 | 实际新增训练 |
|---|---|---:|---|---:|
| 1 |rho=.1|300000|fresh|300000|
| 2 |rho=.1|1000000|自身full_300000.pt完整恢复|700000|
| 3 |rho=.5|300000|fresh|300000|
| 4 |rho=.5|1000000|自身full_300000.pt完整恢复|700000|

CLI中的steps是绝对终点，不能把300k+1M误计成每点1.3M，也不能把两点的700k续训称为
旧Actor迁移。每个点的第一段都不加载历史权重或replay；第二段恢复同点learner、全部
optimizer/replay/RNG、partial episode和真实observed physical state。

核对过的实际flags只有：`--config --variant s_ea --seed 2027 --steps --output
--source-commit --device cuda:0 --num-envs 2 --evaluation-workers 10`，第二段加
`--resume OWN_PREFIX/full_300000.pt`。未发明`--fresh-1m`、`--rho`或`--evaluate-only`。

配置：

- `configs/paper_crax_payload/arm_rho_010_v1.json`
- `configs/paper_crax_payload/arm_rho_050_v1.json`
- 四段有限队列：`configs/paper_crax_payload/arm_rho_queue_v1.json`

## 必须先补的控制器工程gate

该队列复用已有 `scripts.paper_crax_payload.run_queue`，slots=1，按上表顺序。
两个700k continuation均登记：

```text
requires_successful_job: 自己对应的300k前缀job名
required_checkpoint: 该前缀/full_300000.pt
required_parent_receipt: completed /300000 /s_ea /seed2027
```

**现有generic controller尚不理解这些新dependency metadata。** 主协调者必须先加一个
小型可选exit0+checkpoint/receipt gate，或逐段人工admit；不能直接执行当前准备计划
后声称已经有依赖保护。`launch_authorized=false`也是登记状态，不是现有runner的强制
拦截器。source/config SHA已登记，实际admit时须使用包含这些文件的真实新commit，
不得把新配置伪标为旧3c308e4。每个点的配置bytes在prefix开始前冻结，两个阶段之间
不得修改它；admission/running/completed状态放在queue/receipt，不回写进点配置。

失败只保留并停止对应依赖，不自动restart或用别的rho的full补齐。无success-rate门槛、
科学早停或最好checkpoint替换；四段固定预算均保留。无需改Arm学习core或建多层watchdog。

## 旧训练recipe完整复用，训练科学变量仅rho改变

训练源anchor为3c308e49d4d93d49c15ee8112071d04087e622d8；旧工程driver anchor
fc5716e6dc38bc2eefd1b61cc3cb52090b8f07ca。当前Arm learner/endpoint geometry/runtime/
evaluation核心相对3c308e4的聚焦diff为0；queue记录实际文件SHA，供admission校对。

实际有效设置：Panda RelativeLayout-v3、6cm训练半径、25维观测/3维动作、Actor宽128、
margin宽64、两训练环境、batch256、replay300k、UTD1、随机2000步、critic-only1000更新、
每round两reward更新、Actor interval2、target4samples、Huber1、Actor LR1e-4、critic LR3e-4、
Polyak .001、K4、速度界2、A0=0、margin EMA .005、实际transition10k→30k渐进启用。
原生TaskConfig的reward、H100/substeps20、2.5cm末端增量、安全/成功阈值全部不变。
Arm仍是首次违规终止、probability cost-Q和预算.1；**不能复制CRAX累计成本/25预算/新dual**。

新配置的shared_learner与原development_v2.json相比只改rho；geometry完全相等。
新cohort/预算metadata和前瞻评测配置单列，不改变训练reward/cost、更新量或模型容量。

实际参考矩阵规则是：

```text
vN = .04 * [1 - q*(1-rho)]
vT = [3*.04 - vN]/2
```

因此rho是q=1时法向方差相对base variance的比，不是终点VN/VT，也不直接等于vN/vT。
q=1时rho=.1/.25/.5分别给(.004,.058)/(.010,.055)/(.020,.050)，每个切向维度用vT，
注入trace均.12。原StableFlowActor同一D同时用于采样和energy，不能只改采样噪声。
global reference variance .04未变，3D energy target18.75和初始alpha .0008保持原单位。
没有新的critic、二阶模型、动作候选过滤或oracle执行选择。

## 前瞻、适度的新验证，不针对已看过的8cm拟合

原学习曲线节点100k/200k/300k/600k/1M均保留，但新验证只用：

| 条件 | radius | lateral |
|---|---:|---:|
| nominal |.060m|0|
| radius_065 |.065m|0|
| lateral_015 |.060m|±.015m，平衡两侧|

中间布局360027系列，final参数验证370027系列，动作采样seed3970027。
没有原已看过的8cm测试；不会根据它拟合常数、改训练randomization或选择rho。
中间每节点nominal20/stress各10，固定1M每条件30：每点250episode，两个新点共500。
按原H100，上限50000新增非训练控制区间；实际以receipt为准。评测仍使用原隔离
Python/NumPy/Torch/CUDA RNG机制，不把新验证噪声/重置混入训练。

历史.25与Iso anchor应另做**相同新final validation**，各90episode，共至多18000控制区间；
不重放旧18000历史回合，不把旧8cm结果直接与新布局的rho点放进同一公平表。
现有对应模块没有evaluate-only CLI。可复用已经存在的Python API：从各自原生
run_config建立原learner、load其policy_1000000.pt，再把fresh evaluation字典传给
`evaluate_stability(learner,evaluation_config,2027,1000000,workers=10)`。这只变评测plan，
不改已加载Actor/margin/normalizer/dual；单独输出/计费。当前没有执行或伪造这些验证。

四个参数点只在固定1M做新验证参数曲线；完整训练曲线、成本、episodes、所有失败均保留。
Iso anchor的stored actor rho=.25，但geometry禁用后的有效协方差是rho=1各向同性；
必须这样标记，不能称为新训练的EA rho1。历史.25的工程丢弃前缀/跨主机交接也保留，
不把它的旧end-to-end FPS当作与两个新单主机点的严格算法速度对照。

## 默认资源路径

默认4080一个旧Arm环境slot，输出直接在4T的`SAFEFLOW2027_PAPER/runs/paper_arm_rho_v1/`，
避免旧`runs/ea_stability_v2`目录或历史报告被覆盖/自动计入旧cohort。Python指定原已工作
的`SAFEFLOW2027_ARM/env/bin/python`；admit时须确认实际版本并确保不继承新CRAX CUDA12库
preload，准备queue本身不会自动清理父进程环境。
不停止其他项目，不使用5080。若主协调者改为A100，只改资源路径/GPU绑定并重新登记，
训练recipe和2M预算不变；大文件后续归档4T，GitHub仅紧凑代码/配置/图表/账目。
