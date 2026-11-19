# Panda/SafeReach完整3400回合：Panda部署护栏通过本次DEV，SafeReach仍落后

五个冻结原生模型、两种完整部署方式、34个固定条件节点共3400个新DEV回合正常完成，62829个真实仿真控制区间，训练/optimizer更新0。来源`f54300e1`；原模型SHA、训练来源、25D/45D原生维度及恢复规则均保留。此次未选择中间checkpoint或更好硬件模型，所有失败和完整轨迹已收回本地，紧凑指标重新从原JSON计算发布。

## Panda Payload45D：固定P5的部署变化没有退化

|模型/模式|nominal|radius070|payload_wide|execution085|
|---|---:|---:|---:|---:|
|P5 sampled|100%|100%|97%|100%|
|P5 zero_noise|100%|100%|100%|100%|
|Metric sampled|99%|98%|95%|100%|
|Metric zero_noise|100%|98%|100%|100%|

每格100回合，全部无timeout/stall。P5 zero_noise相对本模型sampled逐条件未下降，满足本次开发护栏≤2pp及每条件≥95%；其中payload_wide由97%升100%。公平同模式比较，P5在radius070比Metric高2pp，其余成功率平。100%处没有可提高的成功率空间，不改写成所有条件严格更高。

效率仍有权衡：zero_noise nominal/payload_wide/execution085的P5 horizon-capped时间为0.7508/0.7500/0.8696秒，Metric0.7276/0.7268/0.8424秒，P5稍慢；radius070则P5更好（0.7476对0.7916秒）。执行缩放条件return P5约11.4836对Metric11.4866，不能称所有指标都严格更高。推理p50/p95为共享GPU逐回合分位数的均值，不能据此给跨负载速度因果结论或实机保证。

这只证明**父seed41028的这两个固定45D模型、这批新DEV布局、部署策略变化**的结果。它不是新训练算法的Panda多seed回归通过，不是独立测试，也不能拿Panda41028当成Piper独立父seed41028已经训练完成。

## SafeReach25D：无噪声部署不是通用修复

|模型/模式|nominal|radius065|execution085|
|---|---:|---:|---:|
|P5 sampled|96%|96%|96%|
|P5 zero_noise|96%|98%|96%|
|Iso sampled|98%|95%|98%|
|Iso zero_noise|100%|99%|100%|
|Metric sampled|98%|94%|98%|
|Metric zero_noise|98%|99%|99%|

每格100回合，全部无timeout/stall。zero_noise对P5仅改善radius065 2pp；Metric该条件由94%升99%，Iso由95%升99%，其余条件也有提升。因此P5原本在随机部署radius065的局部优势，在同样无噪声的公平对照下消失；三个条件P5都低于Metric/Iso。不能用“P5 zero_noise 98%高于Metric sampled94%”声称方法优越。

同模式P5相对Metric的capped用时分别慢0.0940/0.0584/0.1192秒、mean return分别低约0.465/0.239/0.680。SafeReach原seed2034训练风险/任务代价保持原记录：固定P5全250K违规682，前100K295；Iso742/345、Metric721/384。部署没有删除已发生的违规/超时/stall，也不能证明训练曲线始终领先。

## 对当前方法选择的影响

Piper零噪声与这次Panda结果是实际正进展，支持保留训练期方向机制并区别探索与部署；但SafeReach反例说明不能直接冻结为“所有任务所有几何全面最优”。Panda的开发护栏可标为此次固定部署比较满足，其多seed/最终配置/独立验收仍缺；SafeReach的训练性能—安全矛盾和最终同模式任务差距依然未解决。

已完成的loss-only taper在Piper两任务和SafeReach均未共同达标，停止该分支追加，不改阈值继续扫描。后续SafeReach修改需针对训练本身，不能靠选择有利部署模式、挑中间节点或重复独立测试凑优势。Piper两个新独立父模型41028/41029继续按原300K完整预算训练，之后的扩域/同预算方法确认仍需单独登记实际父SHA。当前没有新部署模型被选作最终winner。

两项部署实验合计7600个DEV回合、194705个仿真控制区间、零新训练；任务各自只有一个训练seed，不能将7600布局当7600训练seed或统计显著性。所有实机/5080任务继续排除，数据盘原模型与轨迹保留，不新增清理。

数据：[34节点](panda_safereach_modes_complete_20261119/summary.csv)、[3400回合指标](panda_safereach_modes_complete_20261119/episodes.csv)、[同模型部署变化](panda_safereach_modes_complete_20261119/paired_mode_changes.csv)、[相同模式下Metric/Iso配对](panda_safereach_modes_complete_20261119/paired_against_baselines_same_mode.csv)、[Panda开发护栏](panda_safereach_modes_complete_20261119/panda_development_regression.csv)、[原始文件SHA及来源](panda_safereach_modes_complete_20261119/receipt.json)。
