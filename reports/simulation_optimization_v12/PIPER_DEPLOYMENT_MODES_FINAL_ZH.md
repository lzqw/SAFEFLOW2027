# 完整4200回合：无噪声部署改善两项Piper困难几何，但不是训练期全面最优

六个预登记原生100K终点×两种完整部署模式均正常完成，42个条件节点、4200个DEV回合、131876个真实仿真控制区间，新增训练/optimizer更新0。原生收据来源`c7cb61a9`，六个checkpoint SHA全部与预登记一致；每个条件保留原100布局和完整轨迹，无更换失败、无最好checkpoint/硬件模型选择。两模式均使用原target Actor；zero_noise是把四步flow噪声置零的**另一个部署策略**，不修改原随机策略结果。

## 同一新DEV布局下的完整结果

|冻结模型|模式|nominal|Reach57.5mm / Payload55mm|Reach60mm / Payload57.5mm|Payload execution0.85|
|---|---|---:|---:|---:|---:|
|Reach原固定P5|sampled|100%|94%|80%|—|
|Reach原固定P5|zero_noise|100%|97%|95%|—|
|Reach固定+2mm|sampled|100%|97%|81%|—|
|Reach固定+2mm|zero_noise|100%|97%|95%|—|
|Reach Metric|sampled|100%|82%|69%|—|
|Reach Metric|zero_noise|100%|95%|91%|—|
|Payload原固定P5|sampled|99%|81%|60%|100%|
|Payload原固定P5|zero_noise|100%|90%|80%|100%|
|Payload固定+4mm|sampled|100%|85%|69%|100%|
|Payload固定+4mm|zero_noise|100%|96%|85%|100%|
|Payload Metric|sampled|100%|83%|63%|100%|
|Payload Metric|zero_noise|100%|90%|78%|100%|

表格所有42条件均**timeout=0、stall=0**，失败全部按原15mm安全距离判违规、首次失败终止，horizon未延长。新布局与旧DEV不同，不能把旧Reach88%与这里95%直接相减当同一评测改进；真正同布局模式变化是原固定P5的80→95%、+2mm的81→95%。这同时显示不同布局/动作seed下原随机策略仍不稳定。

公平比较必须同模式：Reach原固定P5 zero_noise相对Metric zero_noise在57.5/60mm多2/4pp，nominal平；Payload+4mm zero_noise在55/57.5mm多6/7pp，nominal/execution平。Metric自身也大幅改善（Reach60mm69→91%、Payload57.5mm63→78%），不能只把SAFEFlow改为无噪声后与原随机Metric作比较。

原固定P5 zero_noise的Reach已在这一开发seed/布局上满足57.5≥93%和60≥90%；+2mm没有额外成功率提升，不再据这一结果追加offset扫描。Payload+4mm较强，但原固定P5也达到90%/80%，其55mm与Metric平、57.5mm高2pp。选择哪个配置还要看训练停滞、安全代价和多seed，不单看这张表。

## 完整轨迹改善不等于每个布局都改善

Reach原固定P5在60mm救回15个sampled失败布局，无新失去的成功布局；但Reach+2mm在57.5mm两模式均97%，其中救回2个、也失去2个。Payload原固定P5在55/57.5mm分别救回10/21、失去1/1；Payload Metric57.5mm也失去2个旧成功布局。因此不能宣称无噪声在每个状态/每个布局严格更安全，逐布局胜平负全部保留。

此前一步终止动作probe中Payload的无噪声动作仍危险，不与这里整段轨迹改善矛盾：另一部署策略从较早状态已经改变访问路径。它们测的不是同一个干预，不把小probe重写成完整策略结论，也不据本轮证明唯一噪声因果。

## 任务收益、效率与仍未达标的训练条件

Reach原固定P5 zero_noise在57.5/60mm return9.530→10.152、6.650→9.737，horizon-capped时间1.400→1.286秒、1.793→1.347秒；相同模式Metric分别1.300/1.418秒。nominal P5仍比Metric慢（1.166对1.118秒），成功率都100%，不能称所有效率也严格更优。

Payload+4mm zero_noise在55/57.5mm capped时间1.470/1.761秒，Metric1.614/1.931秒；nominal/execution P5稍慢（1.355/1.593对1.345/1.564秒）。推理数据是A100共享运行时逐回合p50/p95均值，不是合并全部调用后的全局分位数，更不是实机。前后模型约16–20ms随共享CPU/GPU负载变化，不能以顺序测试的时间差证明算法速度因果或稳定10%上界。

zero_noise不改变任何已经发生的训练违规、timeout或stall。输入模型训练历史仍为：Reach原固定P5违规58、+2mm46，均零stall；Payload原固定136/stall1，+4mm108/stall2，Metric197/stall0。所以+4mm虽任务DEV较好，仍有训练stall回归，不能作为全部指标winner。原固定P5对Payload Metric训练风险降约31%，但也没有消除所有训练stall差距。SafeReach early100K/250K问题不由这个零训练评测解决。

## 结论与下一步边界

这是一个实际任务进展，支持“保留原训练期方向机制，部署时不继续注入探索噪声”作为待确认方案；当前不能叫最终SAFEFlow已冻结或全面优于所有baseline。训练只有原父seed41027，DEV可用于后续开发而非独立测试；已有两条真正独立父seed41028/41029正在训练，未来跨seed确认还要原生扩域和同预算方法比较。Panda原生回归、SafeReach全曲线、训练期风险/任务共同改善、新独立冻结集均缺。

不追加密集噪声温度扫描、不降低安全距离或延长timeout、不只报最好模型/seed。下一项应以相同模式下的Metric为准，分别核实Panda与SafeReach的完整部署表现；该准备不授权跳过独立父训练或隐藏训练风险。5080/实机继续不使用，全部原始轨迹/计数/失败保留。

数据：[42节点总表](deployment_modes_complete_20261119/summary.csv)、[4200回合完整指标](deployment_modes_complete_20261119/episodes.csv)、[同模型两模式配对](deployment_modes_complete_20261119/paired_mode_changes.csv)、[同模式Metric配对](deployment_modes_complete_20261119/paired_against_metric_same_mode.csv)、[逐项目标检查](deployment_modes_complete_20261119/development_criteria.csv)、[来源收据与每个原始文件SHA](deployment_modes_complete_20261119/receipt.json)。所有轨迹JSON已收回本地，紧凑指标上传GitHub，模型大文件只留数据盘。
