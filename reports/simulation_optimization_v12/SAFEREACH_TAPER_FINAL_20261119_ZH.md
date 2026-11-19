# SafeReach方向损失局部化完整250K：任务部分恢复，训练风险失守，拒绝候选

固定候选`D-direction-only-taper`正常完成250K，seed2034、原生来源`61bd30f0`、同4080相同初始化。全部250个1K训练节点、七个DEV milestone、三个条件各30原布局均保留；与同GPU的P5/Iso/Metric/恒定LR一起严格汇总，五臂1.25M真实训练transition、3150个DEV回合。没有拿本地分析当新增训练，也没有用原A100较好端点替代对照。

|方案|前100K违规|100–250K违规|全程违规|安全成功数|timeout|stall回合|累计return|
|---|---:|---:|---:|---:|---:|---:|---:|
|方向损失taper候选|357|443|800|11541|355|445|123838.33|
|固定P5+lateLR|295|387|682|10844|361|511|116858.41|
|Iso+同lateLR|345|397|742|11353|383|505|122250.52|
|Metric+同lateLR|384|337|721|11843|287|437|128317.95|
|固定P5恒定LR|295|402|697|10884|359|508|117177.30|

相对固定P5，候选成功数增加6.4%、累计return增加6.0%、timeout少6、stall少66，但全程违规增加17.3%、前100K增加21.0%、后150K增加14.5%。因此不是靠更多timeout降低风险，而是放松方向项后任务量恢复、风险也升高，未形成Pareto改善。前100K357超过历史331.5及同期Iso345；低于Metric384不足以抵消其他要求。全程违规也比Iso高7.8%、比Metric高11.0%，没有20%风险降幅；return仍比Metric低3.5%。

## 固定250K终点DEV

|方案|nominal成功/30|radius065成功/30|execution085成功/30|timeout/stall|
|---|---:|---:|---:|---|
|taper候选|29|28|28|0/0|
|固定P5+lateLR|28|26|28|0/0|
|Iso+同lateLR|29|29|30|0/0|
|Metric+同lateLR|29|29|29|0/0|
|固定P5恒定LR|29|28|29|0/0|

候选对固定P5部分DEV改善，但radius065/execution085仍低于Metric，不能称全部条件占优。最早25K候选nominal timeout13对固定P5的7，也不能从零终点timeout推断从训练开始始终领先。完整逐节点、逐1K差值及每项验收布尔值均保存，不挑中间最佳checkpoint。

**拒绝此候选，不做taper阈值/起始权重追加扫描，不进入多seed。** Piper Reach与Payload同一方向的完整100K也都失败，这三项原生结果合起来不支持推广loss-only局部化作为最终方法。此前离线梯度减扰量是真实测量，但没有包含实际访问分布变化/安全塑形，不能把它当作优化任务的充分条件。本轮不凭负结果再随意叠加新机制。

目前训练期收益主线仍应保留原方向约束和宽几何参考；已完成Piper4200回合表明部署时不注入探索噪声有实际任务收益。Panda/SafeReach同模式评测已登记并运行中，将公平处理所有baseline、单独报告部署策略，不能改写这里800次已发生训练违规。两条真正独立Piper父模型继续训练，共用基础不是最终配置确认；Panda回归、多seed、新独立集及全面目标仍未完成。

全部原生模型/replay/逐步轨迹留4080 4T盘，紧凑JSON/progress已收回本地。后台只读收尾观察器仍按半小时等待原队列完成并生成服务器侧副本，不会重试或加训练；本地已直接用相同严格分析器完成汇总，无需为报告等待下一轮。未停止其他项目，无5080/实机任务，未额外删除存档。

数据：[三训练区间](direction_taper_complete_20261119/training_windows.csv)、[七DEV节点](direction_taper_complete_20261119/development_nodes.csv)、[3150回合指标](direction_taper_complete_20261119/development_episodes.csv)、[所有训练曲线](direction_taper_complete_20261119/training.csv)、[每1K与四基线差值](direction_taper_complete_20261119/training_curve_comparisons.csv)、[逐项目标检查](direction_taper_complete_20261119/development_criteria.csv)、[来源与收据哈希](direction_taper_complete_20261119/receipt.json)。
