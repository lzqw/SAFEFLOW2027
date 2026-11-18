# 新seed250K匹配机制实验：方向项与lateLR独立贡献

原完整任务P2明确要求matched ablation；v10最终报告也明确说其方向机制+lateLR整体包与恒定LR控制不能隔离贡献。v12现有短100K筛选还未触发150–200K LR退火，因此不能回答后期优势来自哪里。当前局部温度/预热/重采样没有合格winner，不继续扫描它们，改做这项尚未完成的机制归因。

本实验不是把未达标候选晋升确认，也不是重复旧独立集；不把几何改善或某个基线胜率当做预设结论。四条新训练seed2034，从完全相同原生25D初始化训练到250K，固定预算合计1M（用户已取消旧总预算限制）；共享网络、optimizer初值、环境/reset种子、更新次数、真安全距离15mm、35mm任务成功距离、100步horizon和首次违规终止。

|arm|Normal-shrink|方向项|150–200K Actor LR|
|---|---|---:|---|
|A-full-P5|是|1|1e-4余弦降至3e-5|
|A-no-direction|是|0|同full|
|A-no-lateLR|是|1|恒定1e-4|
|A-Metric-matchedLR|否；各向同性采样/原方向能量|0|同full|

full−no-direction仅隔离方向损失；full−no-lateLR仅隔离lateLR；full−Metric仍是方向损失+协方差整体，不伪装单因素。“去投影”不存在：现原生算法无独立投影动作，不虚构一个已实现部件去消融。原两批SafeReach Iso仍仅历史/不同seed，不把它替代本轮同期Iso，不宣称本轮已覆盖所有baseline。

初始2000随机步、原1000 critic-only updates、actor每2次learner更新一次、batch256、UTD1、两次reward更新/四target样本、EMA .005、10–30K几何ramp、rho .25/方差.04均不改；不复用2033前缀、不给warm-up候选隐藏早期违规。新DEV11910000起、动作11920000+条件索引×10000+episode，30回合/条件、nominal60mm/radius65mm/execution085，25/50/75/100/150/200/250K全部记录。原target Polyak.001部署不变，评测不更新learner或RNG。没有独立测试，也不选择最好中间点。

必须分别报告0–100K、100–250K及全250K训练违规/成功/超时/stall/return，逐DEV节点比较。对照匹配并不保证从训练起点始终领先；随机前缀可平局。只有一个新训练seed，不能给训练seed标准差或统计显著性；DEV的30布局不是30训练seed。此批仅用于理解机制，真正最终新配置仍须多seed、新独立集和Panda≤2pp回归。

A100两共享槽，前两条为full/no-direction，随后no-lateLR/Metric；不停止其他GPU任务。每条250K可完整保存原始轨迹和milestone，数据全写storage0，经run-with-storage.sh，预计总增长24GiB，启动预留≥80GiB并保留≥40GiB底线。4080当前Panda继续，不因此抢占额外槽，不使用5080或实体机器人。每半小时只读监测，不科学早停，不自动重试。

本批答案可能仍是否定，报告全部负结果；若方向项在同LR下无任务收益，则后续应修正方法解释/针对独立贡献改造，而不是继续只展示安全代价。它不代表Reach60mm或Payload安全目标已经解决，当前全任务目标保持未完成。
