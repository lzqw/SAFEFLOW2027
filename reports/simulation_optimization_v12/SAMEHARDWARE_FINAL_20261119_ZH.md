# 同4080四臂完整250K：早期风险达标，lateLR没有带来共同改进

新增完整P5和同期Iso均正常完成250K，来源`2c06ca2a`；与此前4080恒定LR/Metric（`2f09231b`）构成同硬件、同seed2034、相同初始化的完整四臂对照。新增训练500K，四臂共同分析覆盖1M实际transition、1000个1K训练记录、全部七DEV节点/三个条件/各30回合，共2520回合。不把原A100完整P5混入这张同硬件表，不当作四个独立训练seed。

|方案|前100K违规|100–250K违规|全程违规|安全成功数|timeout|stall回合|累计return|
|---|---:|---:|---:|---:|---:|---:|---:|
|完整P5+lateLR|295|387|682|10844|361|511|116858.41|
|Iso+同lateLR|345|397|742|11353|383|505|122250.52|
|完整P5，恒定LR|295|402|697|10884|359|508|117177.30|
|Metric+同lateLR|384|337|721|11843|287|437|128317.95|

完整P5前100K风险295达到历史≤331.5，并低于同期Iso345/Metric384；但前100K成功2606对2909/3356、timeout356对381/286，仍有任务代价。后100–250K P5违规387相对Iso397只少2.5%，相对Metric337反而多14.8%；后期成功8238对8444/8487，timeout5对2/1，stall11对5/1。不能把早期安全优势当作后期全面领先。

全程P5对Iso风险只降8.1%、return降4.4%；对Metric风险只降5.4%、return降8.9%，均不满足风险至少降低20%和return降幅≤1%。训练累计成功数也分别少4.5%/8.4%。timeout比Iso少22，但stall多6；对Metric timeout多74、stall多74。零DEV停滞不等于训练过程没有任务放弃。

## 250K固定终点DEV

|方案|nominal成功/30|radius065成功/30|execution085成功/30|三条件违规总数|timeout/stall|
|---|---:|---:|---:|---:|---|
|完整P5+lateLR|28|26|28|8|0/0|
|Iso+同lateLR|29|29|30|2|0/0|
|完整P5，恒定LR|29|28|29|4|0/0|
|Metric+同lateLR|29|29|29|3|0/0|

三个条件的完整P5均未超过对应Iso/Metric；不选择200K或最好milestone代替终点。每条件只有30布局，一个训练seed；表格不提供跨seed显著性、独立成功率或全任务结论。

## lateLR单因素的证据

完整P5与恒定LR在前150K的150个保存节点，训练累计数以及actor/reward/margin/cost loss逐项精确一致，未发现早期分叉。后期同GPU、同源码核心学习函数、同初始化/随机前缀，因此排除了此前A100/4080硬件拆分造成的前期混杂；结论仍限本seed/本任务。

退火相对恒定LR全程仅少15次违规（682对697，−2.2%），成功少40、timeout多2、stall多3、累计return低0.27%；终点成功28/26/28对29/28/29。当前不支持把lateLR写成兼顾任务与安全的关键独立收益，也不支持继续细扫退火时间/终值。原A100方向项消融的后期安全作用保持独立结论，不能拿它覆盖这里的负结果。

## 后续运行与证据

严格分析已在本地和4080各生成一次，均验证完整原生收据、所有节点/布局/同初始化；没有新增模拟。为避免多等半小时的只读观察周期，在4080原输出`samehardware-analysis/`直接运行相同严格分析，然后核对完整收据后仅退休原只读等待进程3750007（精确模块/cwd检查）。没有重启、缩短或改变任何训练；已登记候选队列3915728收到依赖后自行接续，原方向taper固定实验预算250K不变。候选结果未有，不晋升。

A100 Reach loss-only taper已完整失败并停止追加，该负结果不冒充SafeReach或Payload；两条Payload依旧按已登记100K运行。整体仍缺合格共同配置、跨独立父seed稳定性、Panda≤2pp回归和新冻结独立评测，目标未完成。

数据：[完整三个训练区间](samehardware_complete_20261119/training_windows.csv)、[七DEV节点](samehardware_complete_20261119/development_nodes.csv)、[所有DEV回合](samehardware_complete_20261119/development_episodes.csv)、[所有1K训练节点](samehardware_complete_20261119/training.csv)、[逐节点差值](samehardware_complete_20261119/paired_nodes.csv)、[来源与前150K一致性](samehardware_complete_20261119/receipt.json)。服务器原始checkpoint/replay/轨迹及已退休观察日志保留，不额外删除证据。

已生成[完整学习曲线PNG](samehardware_figures_20261119/samehardware_learning_curves.png)及[可用于论文排版的PDF](samehardware_figures_20261119/samehardware_learning_curves.pdf)：上排全部训练违规/累计return/timeout，下排全部七DEV节点的三个原条件。阴影只标原150–200K LR区间，不代表获益区间；连接保存点不证明连续每一步领先。没有跨训练seed误差条，全部4条曲线/负节点保留；stall仍见上表/CSV，不把timeout替代stall。
