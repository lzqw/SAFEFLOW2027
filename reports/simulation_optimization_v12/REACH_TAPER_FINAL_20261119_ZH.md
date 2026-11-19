# Reach方向损失局部化完整100K：拒绝候选，保留同期固定对照的正结果及不稳定性

原生两条训练正常完成100K新增transition、五个固定DEV节点和所有原100布局，共200K新增训练与3000个DEV回合。来源`d904b9c8`、A100 GPU1、同EA400K完整父/父seed41027。完整收据和原始JSON已收回本地；模型/replay继续在A100数据盘，未追加评测、训练或独立test。

|方案|训练违规|安全成功|timeout/stall|累计return|DEV nominal|57.5mm|60mm|
|---|---:|---:|---|---:|---:|---:|---:|
|loss-only taper+2mm候选|118|3297|0/0|34376.62|100%|91%|83%|
|本轮A100固定P5+2mm control|48|3254|0/0|34566.87|100%|94%|91%|
|历史4080原固定P5|58|3259|0/0|34529.13|100%|94%|88%|
|历史4080 Metric-only|169|3335|0/0|34324.95|100%|86%|72%|

本候选相对同硬件control训练违规增加145.8%，两个困难条件下降3pp/8pp；也未满足57.5mm≥93%和60mm≥90%，训练违规远高于原固定P5的110%上限63.8。全部训练及DEV均零timeout/stall，不是用任务放弃换低违规，但任务—风险联合表现仍退化。相比历史Metric仍有部分优势不能抵消对原P5和同期control的退化。**拒绝Reach loss-only taper，不继续扫其支撑阈值，不进入多seed确认。**

最终DEV capped完成时间候选57.5/60mm为1.4572/1.6868秒，同期control1.3956/1.4848秒；nominal候选1.1732秒，control1.2032秒。安全失败按原horizon capped口径计，不能把失败更早终止当作效率提升。累计return只降0.55%不是共同达标理由。

此前保存replay诊断观察到的梯度冲突确实存在，但移除额外方向损失并未改善真实外推任务，反而破坏安全。梯度余弦是局部critic目标，不等于真实策略改进；本结果不支持用“任务梯度冲突下降”来选择最终算法。也不改写先前诊断为全错，它测量的量与终点成功不同，原数据与哈希完整保留。

本轮**固定P5+2mm对照**单seed DEV91%/94%、训练违规48，确实同时跨过当前Reach任务阈值和原固定P5风险上限，无timeout/stall。这是正结果，应保留，不能因它被称为control而忽略。但历史4080同设置DEV88%/95%、违规46，说明同一父seed/配置跨硬件数值轨迹能改变是否达标。两条不是两个独立父训练seed，当前不足以确认稳定winner。应先保留固定方向+宽几何参考，在本批其他任务完成后设计独立父seed的有限确认，不伪造新seed或挑最好硬件结果。

已有4080 SafeReach loss-only候选保持既定单因素筛选；Piper Reach负结果不冒充SafeReach或Payload结果。A100两条Payload已自动接续，PID3890430/3891505，源同`d904b9c8`、同GPU1，全部原100K预算和四条件DEV不改。尚无Payload终点，不晋升不提前终止。Panda新配置回归/多seed/冻结独立集仍缺。

完整对比数据：[训练终点](reach_taper_complete_20261119/training_endpoints.csv)、[全部五DEV节点](reach_taper_complete_20261119/all_dev_nodes.csv)、[每个布局的配对胜负汇总及时间/return差值](reach_taper_complete_20261119/paired_nodes.csv)、[逐项目标检查](reach_taper_complete_20261119/development_criteria.csv)、[来源收据哈希](reach_taper_complete_20261119/receipt.json)。历史参考明确跨硬件，不包装成同期因果量。
