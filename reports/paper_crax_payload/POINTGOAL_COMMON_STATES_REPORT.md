# PointGoal共同状态：条件性方向响应，但本池未见单步native风险下降

本轮是对两个**已关闭的开发4M Actor**做的小型共同状态分析，不是独立测试，
也不是当前正式训练策略的评估。结论限于这个状态池及已经固定的动作：
PB对部分近边界/零cost状态产生了法向缩小、切向增加的响应，但没有降低本池
native cost-positive比例；EA Actor全池切向方差反而下降较多，不能只展示法向数字。

原始证据：[pool](pointgoal-common4m-161637c/pool.json)、
[plan](pointgoal-common4m-161637c/plan.json)、
[results](pointgoal-common4m-161637c/results.json)、
[receipt](pointgoal-common4m-161637c/receipt.json)。
分析代码SHA256为
`161637c588264ba2c6c6dd59ab1dfc2f7db8bcf532402bbecc7f9706524ee7ac`；
两个Actor训练来源均为 `6f9e3bf3488164cabf1fe4e3dcc51b695d85b5af`，
seed41027、固定4M，normalizer均冻结在count4M；full文件SHA见plan，非仅按文件名认定。
这不是两个独立训练seed。

## 设计与可比范围

- 32个保存状态指纹互异，Iso来源16、EA来源16；不是只选好Actor的安全状态。
  合计4个unsafe-proxy、12个near-boundary、16个interior，每个来源分别2/6/8。
  近边界定义为当前可观测静态proxy margin在[-0.15,0]m，危险侧为>0。
  这是刻意分层的训练分布状态池，不是策略真实访问频率样本。
- 每个状态16次共同Gaussian draws；同一个冻结Actor比较iso-reference与
  endpoint-pullback。两Actor各自使用其冻结normalizer和EMA margin；没有更新参数或dual。
- 两动作维度、K4、reference variance0.04、rho0.25。参考D的trace均0.08，
  每个Brownian注入的dtD trace为0.02；两sampler**不是同谱**，
  也没有强行匹配最终输出动作的trace。不能把这个干预称为完整的同谱M2矩阵。
- 公共轴来自同一状态**零动作**处、单个原生控制区间之后的endpoint-clearance
  proxy对动作的中心有限差分，epsilon0.05。它不是native cost-Q法向、
  真实控制区间最小距离法向或安全证书。32个轴都超过1e-7有效阈值；
  单epsilon近似没有被包装为解析精确梯度或收敛误差界。
- **所有四组learned动作先固定，之后才执行FD和native标签查询**。
  FD轴只用于评价投影，未进入learned执行、候选动作筛选或替换；oracle选动作数为0。
  保留difficulty1、horizon1000和原生0.02物理步长/frame skip4；没有缩短训练任务。
  没有新reset来替换原状态。

这里的“endpoint VN/VT”指**Flow终端归一化动作**的协方差在公共动作轴上的投影，
不是机器人/Point物理位置的方差。每状态先用16个draws计算ddof1协方差，再平均；
2维动作中VT只有一个切向方向。以下VN/VT比值是“均值之比”，不是逐状态比值的均值。

## 全池原生cost与任务相关诊断

| 冻结Actor | sampler | 平均native单步cost | cost>0 draw比例 | endpoint proxy危险比例 | 距源目标中心单步接近量m |
|---|---|---:|---:|---:|---:|
| stable_iso | iso_reference | 0.08164465 | 12.500% | 12.500% | 0.045207 |
| stable_iso | endpoint_pullback | 0.08164369 | 12.500% | 12.500% | 0.045208 |
| ea | iso_reference | 0.08150941 | 12.500% | 12.500% | 0.044680 |
| ea | endpoint_pullback | 0.08154131 | 12.500% | 12.500% | 0.044680 |

四组均为相同4个状态的全部16个draws出现正cost：状态索引**0、14、20、24**，
每组64/512=12.5%；另外28个状态的全部已测draws均cost=0。
因此没有出现native正cost事件比例或proxy危险事件比例的下降。
平均cost的PB减iso-reference差分别约**−9.69e−7（Iso Actor）**、
**+3.19e−5（EA Actor）**，不是稳定或实质native安全收益，EA方向还略为反向。

这些是**单步raw cost**，不是Episode Cost，不能直接对照episode预算25，
也不能乘1000当作策略episode成本，因为本池经过风险分层。
所有样本native goal计数均为0，native done也均为0。距离接近量只是诊断：
原生distance-reward系数为0，不等同于native return增益；PB对这一接近量的变化约微米量级。
本次没有路线覆盖、目标完成或长期任务收益证据。

## 全池终端动作几何、clamp与真实连续权重

| Actor | sampler | VN | VT | VN/VT | 终端动作trace | 平均坐标clamp率 | 几何平均权重 |
|---|---|---:|---:|---:|---:|---:|---:|
| stable_iso | iso_reference | 0.033782 | 0.046243 | 0.7305 | 0.080025 | 2.417% | 0.00000 |
| stable_iso | endpoint_pullback | 0.032815 | 0.045877 | 0.7153 | 0.078692 | 2.368% | 0.09764 |
| ea | iso_reference | 0.023330 | 0.051466 | 0.4533 | 0.074796 | 7.935% | 0.00000 |
| ea | endpoint_pullback | 0.023173 | 0.048669 | 0.4761 | 0.071842 | 7.813% | 0.08243 |

PB使全池VN下降 **2.86% /0.67%（Iso/EA Actor）**。但EA的VT同时下降**5.43%**，
VN/VT从0.4533上升到0.4761；不能称为全池切向探索保留得更好。
终端trace分别下降1.67%和3.95%，再次说明注入点trace相同不保证输出trace相同。
非线性tail/clamp之后的总体缩小不能全部归因为纯方向分配。

即使两Actor都使用共同iso-reference，EA Actor与Iso Actor的终端协方差已经不同
（VN0.023330 vs0.033782）。因此原生EA与原生Iso之间的差异不能全部归于测试时PB调制；
本交叉干预只能区分当前固定Actor/normalizer与sampler的条件作用，不证明Actor训练改善的因果性。

clamp字段是跨K步、动作坐标的平均越界并被真实clamp的比例，
不是“任意一次clipping的trajectory比例”，也不是碰撞率。
几何字段为连续权重的平均（含实际ramp/可靠度），**不是9.76%或8.24%的状态被二元激活**。
没有记录条件层的clamp/activation、learned-normal cosine或margin误差，
因此本报告不按全池均值补造这些条件统计；FD轴有效也不证明learned法向正确。

## 条件诊断：内部、近边界、危险侧与零cost高风险

以下不增加任何native查询，只复用同一per-state原始记录。正负号表示PB相对iso-reference
方差变化；这些子集有重叠，不能相加为额外样本数或独立训练重复。

| Actor | 条件 | native cost>0比例 iso→PB | VN iso→PB | VT iso→PB | VN变化 /VT变化 |
|---|---|---:|---:|---:|---:|
| stable_iso | 内部，n=16 | 0.000→0.000% | 0.041242→0.041206 | 0.038179→0.038161 | -0.09% / -0.05% |
| stable_iso | 近边界，n=12 | 0.000→0.000% | 0.028228→0.026921 | 0.053156→0.054784 | -4.63% / 3.06% |
| stable_iso | 已在proxy危险侧，n=4 | 100.000→100.000% | 0.020605→0.016936 | 0.057759→0.050024 | -17.81% / -13.39% |
| stable_iso | 零cost标签中危险最高20%（6/28），n=6 | 0.000→0.000% | 0.024470→0.022218 | 0.054151→0.057352 | -9.20% / 5.91% |
| ea | 内部，n=16 | 0.000→0.000% | 0.034482→0.034453 | 0.047224→0.047283 | -0.08% / 0.13% |
| ea | 近边界，n=12 | 0.000→0.000% | 0.014569→0.014473 | 0.051414→0.051629 | -0.66% / 0.42% |
| ea | 已在proxy危险侧，n=4 | 100.000→100.000% | 0.005004→0.004155 | 0.068591→0.045332 | -16.96% / -33.91% |
| ea | 零cost标签中危险最高20%（6/28），n=6 | 0.000→0.000% | 0.017739→0.017594 | 0.042843→0.043242 | -0.82% / 0.93% |

**零cost高风险子集的事后定义**：取共同零动作FD标签native cost=0的28个状态，
按该零动作之后的endpoint proxy margin由大到小排序，取ceil(0.2×28)=6个，
即索引**19、7、16、27、13、25**；6/28≈21.4%，不伪称恰好20.0%。
它们都属于原near-boundary，两个Actor、两sampler使用同一个子集；
不是重新选择动作、独立测试或后续候选门槛。

这个子集中，Iso Actor的PB是**VN−9.20%、VT+5.91%**，
EA Actor则是**VN−0.82%、VT+0.93%**。可见部分方向性协方差重分配，
不能只用全池均值说完全没有响应；但本子集已测native cost全部为0，**不存在可展示的cost下降**，
也不能从有限draws的零cost推成零风险。
已在proxy危险侧的4状态中，两Actor正cost比例仍100%；EA虽然VN−16.96%，
VT却−33.91%、trace−32.76%，平均cost还约+0.000255。
这是需要保留的反向/无收益结果，不能只取零cost子集的有利几何数字。

## 成本、CSV映射与结论边界

实际非训练native控制区间为 **2208**：
32×(1+2×2)=160个FD标签，4组×32×16=2048个动作标签。
原生receipt耗时81.232s；不是单动作推理延迟或完整训练SPS。
新增训练/optimizer更新/真实交互/reset均为0，报告汇总本身也没有产生新查询。

`MECHANISM_RESULTS.csv`保留原108行和原schema，追加4条PointGoal全池行：
`source_commit`在这4行明确指Actor训练来源；analysis SHA、两checkpoint SHA、
native cost/activation和各条件聚合置于现有`numerical_evidence_json`。
`true_interval_risk`、interval“true axis”相关字段及同谱字段留空，不把proxy结果塞成
机械臂的真实区间安全指标。`mean_reference_trace=0.08`指D，不是dtD或终端协方差。
条件层是复用同一标签的诊断，不重复计2208成本。

这份补充支持的最强表述是：
**在固定开发Actor、共同状态和共同噪声下，PB存在条件性的方向分配响应；
但本小池没有native事件风险下降或可用任务收益证据。**
它不证明全局策略优于Iso、不替代正式三seed结果、独立泛化或预算内任务评估，
也不把真实控制区间/任意动力学安全性当作已被oracle验证。
