# 原生Piper完整轨迹部署模式实验：六个冻结模型，Metric同样处理

已有终止噪声诊断只在同一个终止状态试了一个无噪声动作，不能据此推断完整无噪声策略的成功率。Reach部分随机违规可避免，Payload部分无噪声动作仍危险；既不能预设统一降噪解决所有问题，也不能据一步动作否定整段轨迹的改变。本轮直接检验完整部署策略，零训练，不追加权重/温度扫描。

固定六个既有完整100K端点及SHA（见配置）：Reach原固定P5、固定+2mm、Metric；Payload原固定P5、固定+4mm、Metric。全部来自4080已保存端点、原父seed41027，在A100同GPU0上分别跑原sampled与zero_noise两种完整策略。没有纳入已失败的Reach taper，没有选择最好中间点/硬件模型；此次Reach+2mm仍用原88%端点，而非刚达到91%的A100对照，避免后见挑模型。所有原stochastic结果继续单列保留。

zero_noise直接使用原Actor的`deterministic=True`，四步flow每次Brownian增量置零；不是对多个采样动作平均，也没有action候选筛选、回退或环境oracle。它是**不同的部署策略**，不能把其结果改写成原stochastic SAFEFlow结果。方向训练/critic/参数/optimizer/replay/dual不改；训练风险也不会因此降低，当前训练期未达标事项仍未解决。Metric也同样跑无噪声，不能只对SAFEFlow使用更有利部署。

使用已有target Actor，沿用原任务环境/实际动作scale/15mm安全距离/horizon/首次违规终止。Reach新DEV布局11970000–11970099、动作11971000+条件索引×1000+episode；Payload11980000–11980099、动作11981000+条件索引×1000+episode。所有模型/两模式共享布局，轨迹可不同，不假造相同访问状态。Reach nominal/57.5/60共三条件、Payload nominal/55/57.5/execution085共四条件，各100回合，固定42个条件节点/4200回合。不是旧独立测试集，也不是新冻结独立验收；结果允许后续开发但不能宣称独立winner。

记录所有回合安全成功/普通终端成功/违规/timeout/stall/return/horizon-capped完成时间/真实推理p50与p95，逐布局配对零噪声相对采样的增益与退化。不得丢弃更早违规/超时/停滞或只有nominal成功的模型，不进行科学早停，不自动重试或按前面结果更换后面的模型/模式。推理耗时是A100共享环境测量，不是实机或旧0.812秒episode时间。

输入只复制六个白名单checkpoint及原生receipt到A100数据盘，约0.7GiB，源4080全部保留不删除。新source/cache/TMP/结果经storage wrapper，整个新任务保守预留2GiB、保留40GiB及增长余量，避免占home；GPU0现空闲，单共享槽allocator cap .025，若其他项目占用导致显存不足则等待，不信号它们。GPU1两条原Payload训练及4080候选继续，不改运行源码。

这是固定模拟实验，预计约20–60分钟，依实际物理与推理速度调整；每30分钟监测。没有5080、实机、CAN或SDK任务。部署模式即使有改善，也仍缺新训练配置跨seed、Panda回归、完整训练风险共同改进和最终独立集；不能以此替换原全面目标。
