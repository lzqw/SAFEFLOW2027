# 2026-11-18运行交接（11:36 UTC核对）

目标未完成，持续优化；所有运行均仿真，不使用5080或机器人。用户取消旧gate/总预算上限，不改变15mm安全距离、timeout、horizon、任务成功标准或违规终止。

已完成Reach前两轮5×新增100K和真正SafeReach6×fresh100K；全节点/负结果已保留，紧凑数据、图和代码在结果分支。大模型与原始轨迹仍在服务器数据盘，原始DEV本地完整归档；不把大模型放GitHub。

当前：

|主机|有限任务|状态（非最终结果）|
|---|---|---|
|A100 GPU0|E3固定P5、4:1边界采样|12K/100K，原生PID2789743|
|A100 GPU1|E3匹配Metric-only、相同采样|12K/100K，原生PID2789744|
|4080|Reach 2mm激活+quarter75K|新增33K/100K，原生PID2080615|
|4080|同EA400K父显式Metric-only|同队列原生PID2080617|
|4080接续|Payload固定P5/2mm激活/Metric-only各100K|已启动有限等待队列，Reach两完成收据后启动，不额外抢占槽|

A100 E3训练source cf010fa9，4080 Reach source cba886b1，Payload source70b414cb。每个项目独立snapshot，不在运行时热改。Payload新增的action_scale实际执行修复只放独立source-payload；source-r3保留原版本，原Reach所有条件scale1不变。

半小时只读资源/进度监测在两台服务器后台运行，有限队列完成后退出，不杀外部进程、不自动重试。A100新队列通过run-with-storage.sh，数据盘尚约107GiB，预留≥40GiB及预估增长；4080的4T盘约1.55TiB可用。外部共享负载会影响速度，不用瞬时GPU空闲作为长期算力保证。

第一阶段早期学习曲线的逐点比较已另存early_pairwise：P5对Iso/Metric均在4K首次出现累计违规更多，不能说从起点始终领先；对两基线各12个DEV条件×时间节点中只有7个在成功、违规、超时、停滞、return、capped time六项上描述性同时不差。单训练seed不做统计显著性结论，相同随机前缀允许平局，不虚构严格优势。

仍待完成：当前三系列终点；按训练违规+任务+timeout/stall共同筛选；有证据后固定配置并fresh多seed、250K后期保持、Panda Payload≤2pp回归、新独立集；逐任务/逐几何明确胜平负。没有通过这些要求之前不宣布全任务占优。

定向检查：BoundaryReplay3项、Metric native identity1项、execution085执行1项均通过；另10步环境smoke为engineering_only，不计入科学结果。4080首次pytest仅因第三方自动加载插件冲突失败，通过禁用自动插件运行同一测试，无安装或修改共享环境。单独补充缺少goal的测试桩后execution测试通过，不是运行环境故障。

关键本地镜像：`/home/lzqw/SAFEFLOW2027_delivery/关键结果汇总_20261114/新仿真优化_v12/`。
