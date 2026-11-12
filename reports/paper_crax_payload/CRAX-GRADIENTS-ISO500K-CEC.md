# CRAX Iso500k：一次冻结replay梯度诊断

输入：`crax-iso-norm-balanced-41027-500k-v1/full_latest.pt`，真实500224步，
训练源`cec7e5dfae0a2053d54a66e4dcbe9825987b80ea`。本次分析仅执行一次，耗时4.17秒；
0新增训练、0optimizer step、0环境构造/step、0模拟查询、0真实交互。

分析脚本不是伪称已提交的科学版本：它在运行前后以明确工作树SHA256固定，
`d7af777eecea685954dfceda777f0dd4878f41886e4cff01b807e179d321680c`。
实际导入的learner SHA `19cfc387...` 与CEC训练源一致；normalizer SHA `7c3889bf...`。
完整64位hash、全部原始指标、256个replay索引及其hash见同名JSON。

## 定义与单位

一次固定256-row raw replay采样，使用独立诊断RNG，未消费保存的replay RNG。
加载真实保存的running_acme normalizer（count500224、保存的mean/std），不重新估计。
Actor只生成**一批共同动作/同一随机路径**，三项损失共用它：

```text
L_R = mean[-min(Q_R)]
L_C = lambda * mean[min(Q_C)]
L_E = alpha_E * mean[path_energy(D)]
```

以下都是对同一个Actor参数向量的**loss梯度**，不是把正reward梯度和负loss梯度混用。
权重、normalizer与输入文件未改变。geometry为Iso的0权重；没有更改噪声或安全规则。

## 实际结果

| 项 | 已乘实际系数的目标均值 | Actor参数梯度L2范数 | 相对reward梯度 |
|---|---:|---:|---:|
| Reward loss |+0.278851|0.0238343|1.00|
| Cost penalty |−1.854261|0.4394443|18.44|
| Energy penalty |+0.030240|0.0569882|2.39|
| 合计 |−1.545170|0.4386958|—|

实际lambda=1.395762，episode-cost EMA=164.576，dual已记录256个完整episode。
alpha_E=.000887548，平均path energy34.071（目标12.5）。采样动作clipping比例20.459%。
总梯度范数小于10，因此这批若更新不会触发全局Actor梯度裁剪。

| 两项loss梯度的余弦 | 实际值 |
|---|---:|
| Reward /Cost |+0.55848|
| Reward /Energy |−0.13368|
| Cost /Energy |−0.31563|
| Total /Reward |+0.59640|
| Total /Cost |+0.99105|
| Total /Energy |−0.19353|

**Cost项确实主导当前局部更新，但这不是“Cost梯度处处反对任务梯度”的证据。**
本批cost与reward loss梯度反而正相关；total仍与reward改善方向正相关，且几乎沿cost梯度。
Energy也不是可忽略项，其梯度约为reward的2.39倍。不能仅凭范数就断言去掉/减弱cost
必然解决停滞，也不能将这次off-policy局部条件诊断当成on-policy总体因果估计。

## 另一个重要异常：负的policy cost-Q

原始成本非负，但当前生成动作上的min cost-Q均值**−1.32849**，范围
**[−3.10331, +0.97683]**。因此lambda*cost-Q目标均值为负，不是一个正的任务惩罚常数。
真实折扣累计成本应非负；这是critic低估/动作外推的警示，不意味着需要恢复机械臂的
sigmoid概率Q，也不是一个已证明的唯一停滞原因。线性unbounded cost-Q的任务语义仍正确，
需要区分它的表达范围与学到的实际校准。

Reward-Q均值−.27885不单独等价于reward critic错误：该实现的reward target包含未来
energy正则项。此次原始replay batch的reward全部为0，符合稀疏目标任务的一个有限样本；
不能据此声称整个replay没有任何成功。原始cost均值.10002、28/256非零，最大3；
真正terminated为0，truncated为1/256。原始reward/cost没有偷偷缩放。

## 下一步解释边界

把Flow dual的统计/更新单位与原生SAC-Lag明确匹配，是有依据的**前瞻性单因素候选**：
仍保留共同episodic budget25，明确它与25/H的per-step界和.01/H的dual步长关系。
当前raw-episode batch规则在首批完成时产生约1.396的lambda，而任务还在稀疏学习阶段；
它与原生规则不是简单增加环境并行数就可互换的同一优化过程。

但本诊断没有执行该候选，也没有证明该候选会获胜。负cost-Q、energy量纲、稀疏reward
和有限更新量都仍是解释范围内的因素；不能用一次诊断替换真实候选训练/共同验证。
所有更改应公平应用于Iso/EA共享主干，并保留当前500k失败/折中结果。
