# PointGoal正式阶段：共同10M已冻结，完整九组合待启动

计划文件是 `configs/paper_crax_payload/pointgoal_formal_plan_v1.json`。
协调者在2026-11-12 07:38:24UTC根据完整Iso/EA4M对照，已将其显式冻结为
三方法共同10M、三个新训练seed、每seed五个checkpoint；这不是部署模型选择结果。
此前准入功能准备阶段没有启动训练或消耗测试集，一个开发seed好即可继续研究，
不新增“所有seed同时获胜”的研究前提。

SAC使用归一化、LR.01和原生E256/B1024/G4，Flow使用共享v3/Balanced。
Iso从2M的3.82目标升至4M的12.42，EA从0.42升至18.22；两者成本仍高于25。
据此共同预算选择10M以覆盖仍在进行的学习；没有单独延长某种方法。
十二个算法文件hash已固定，九组合CPU准入检查通过，0训练/模拟查询。
正式启动由有限队列执行，实际PID/角色和向量/host epoch取整记录在运行凭证中。

## 固定范围

- 三方法：`sac_lag`、`stable_iso`、`ea`；native CLI的别名`sac_lag_native`规范为`sac_lag`。
- 三个已注册正式训练seed：51027、51028、51029，完整九组合必须在计划内。
- 只允许共同每seed预算5M或10M；不允许方法单独变更目标。
- 精确名义checkpoint fractions：0.2、0.4、0.6、0.8、1.0，共五个选择机会。
- 固定原生PointGoal difficulty1、累计cost语义及共同episode预算25。

Flow和native SAC两个CLI新增 `--phase {development,formal}`，默认仍为development；
仅formal需要 `--formal-plan`。未冻结计划、未注册seed/方法、不相符的共同预算、
配置或候选机会都会在GPU库导入/设备初始化之前拒绝。即使冻结参数与CLI共同写错，
Flow的env/difficulty/task-mode/budget仍须与原生TASK一致，且environment overrides必须为空。
Formal禁止profile-only、
工程提前停止/跳过评估、个别目标扩展等旗标；development原有选项保持可用。

示例只说明参数，历史prepared计划会被拒绝；当前frozen计划仍要求参数完全一致：

```text
python -m scripts.paper_crax_payload.train_crax ... --phase formal --formal-plan configs/paper_crax_payload/pointgoal_formal_plan_v1.json
python -m scripts.paper_crax_payload.native_sac_checkpoint ... --phase formal --formal-plan configs/paper_crax_payload/pointgoal_formal_plan_v1.json
```

## 后续显式冻结，不由guard自动决定

后续依据完整开发证据选择共同5M或10M，填写全部三方法的runtime参数快照及freeze记录。
Flow当前候选来自`pointgoal_native_dual_v3.json`，但还未宣布它是最终配置；Iso和EA的
共享参数必须相同，方法差异仍由显式method决定。Native SAC的normalization、LR和E/B/G
也必须选定并记录，不能只冻结EA或把SAC默认值当作已选结果。

`flow_runtime_parameters(master,profile,eval_envs)` 与
`native_runtime_parameters(args)`生成运行时比较用的普通JSON字典，不创建GPU对象。
`algorithm_file_hashes()`只读计算12个直接算法/入口依赖文件的SHA256，不写计划。
只有后续明确填写这些快照、代码hash、共同预算和冻结决策记录，才可把status改为frozen。

绑定的是**实际代码文件hash**，不是要求“包含计划的commit等于计划内自身commit”，
因此没有自引用Git hash陷阱。CLI的source commit仍记录本次调用来源；仅文档/计划文件
的新commit不会改变已固定算法hash。计划文件本身的完整字节SHA另行保存，冻结后不能
静默调整内容。准备或修改计划都不会自动launch；资源调度仍由原有简单有限队列处理。

## 阶段记录与恢复边界

Formal configuration、policy、full state、validation及receipt写入`phase=formal`和
`formal_admission`，记录计划SHA、方法/seed、共同预算、精确名义checkpoints和实际代码hash。
Formal中的validation仍不冒充independent test。恢复必须来自同一formal计划；
旧development权重不能仅添加`--phase formal`就改名成正式训练。Formal checkpoint也不能
通过默认development调用被隐藏。代码commit标签或文件路径迁移本身不会绕过/破坏相同
计划及相同算法hash检查。

元数据在learner/native `settings` **之外**；未给formal标志时，既有development
configuration/state字段和exact settings比较保持不变，缺少新phase字段的旧development
full checkpoint仍能按原严格设置恢复。独立调用`train_crax.evaluate`也有自己的lazy
环境导入，不依赖先执行main。

五fractions是精确的**名义训练预算比例**。Flow仍按向量批次向上取整，native仍按已保存的
原生host epoch边界执行；没有为了计划改写native SGD或模拟器。具体actual checkpoint
steps、native candidate epochs和rounded endpoint另行记录，所有取整新增transitions照实计账。
本版formal continuation只支持同一冻结共同目标；不会悄悄把某一方法5M续成10M。

## Native SAC累计训练成本的前瞻历史

`FullCheckpointManager.save_epoch`新增小文件`checkpoint_history.jsonl`，每个**已经实际保存**
的host epoch追加一条：steps、epoch、初始步数及本次调用新增前缀、训练累计cost/goals、
cost/1000 transitions、evaluation calls、候选资格和调用来源。它不新增策略候选、不保存
所有full/replay，也不增加SGD/环境步；原有`full_latest.pkl`行为不变。

后续collector可按候选的真实steps匹配这一历史获取当时累计成本。旧运行已经缺失的中间
成本保持unknown，**不从最终累计成本回填，不重跑历史**。Development默认唯一新增的
持久产物是这份前瞻JSONL；既有settings和恢复格式没有强行加新字段。

聚焦验证：formal准入/提前拒绝和原生checkpoint状态契约共11项CPU测试通过，0GPU任务、
0模拟控制、0真实交互；其中原生六个scientific inner函数AST不变也继续通过。实际正式
训练、冻结决策以及正式测试均不在这份接口准备凭证中冒充完成。
