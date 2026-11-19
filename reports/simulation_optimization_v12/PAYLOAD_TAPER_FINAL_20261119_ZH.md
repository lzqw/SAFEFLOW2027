# Payload loss-only taper完整100K：同样未通过，停止该Piper分支追加

两条A100原生训练正常完成各100K新增transition、全部五DEV节点及四条件各100布局，新增训练200K、DEV4000回合。来源`d904b9c8`、同GPU1、同原生EA120K完整父seed41027，全部原始JSON/progress已收回本地。与已完成Reach合计四条400K/7000个DEV回合；不是多个独立训练seed。

|方案|训练违规|安全成功|timeout/stall|累计return|DEV55mm|DEV57.5mm|
|---|---:|---:|---|---:|---:|---:|
|loss-only taper+4mm候选|178|2819|0/1|28475.82|79%|63%|
|本轮同A100固定P5+4mm control|118|2711|0/2|27870.01|83%|64%|
|历史4080原固定P5|136|2738|0/1|27992.95|74%|62%|
|历史4080 Metric-only|197|2832|0/0|28436.36|78%|64%|

所有方案nominal/execution085终点DEV均100%，最终DEV均零timeout/stall。候选对同期control训练违规增加50.8%，两个困难条件各下降4pp/1pp；对原固定P5违规增加30.9%，超过原110%上限149.6。对Metric的57.5mm成功仍少1pp，训练违规降幅只有9.6%，也没有达到全任务共同改善要求。候选虽比control累计return高2.2%、训练成功多108、训练stall少1，不能据此忽略安全和困难几何退化。

候选55/57.5mm horizon-capped完成时间为1.9528/2.3560秒，对照1.8364/2.3172秒；失败按原horizon计，未增加timeout或修改终止来改善结果。训练stall是可与成功/违规重叠的诊断，不加入终止类别总数。

**拒绝本Payload候选，与Reach一起停止Piper方向损失支撑阈值的追加搜索，不进入多seed。** 此前保存replay上梯度冲突减弱不等于真实任务获益，两个原生任务的完整结果均不能支持推广。现有固定+4mm仍作为参考，但本轮训练stall2对原固定1/Metric0，不能因违规较低选作全面winner；旧同配置84%/68%与本轮83%/64%也提示数值路径/硬件稳定性限制。历史参考跨硬件单列，不能当同期因果量。

当前仅保留已经登记并运行的SafeReach25D候选250K，不追加其阈值扫描；Piper负结果不提前充作SafeReach结果。A100 GPU1这四条训练已退出，GPU0正在做另一项零训练的冻结模型部署模式实验；它比较整段轨迹，不能改写原训练违规或把无噪声结果当作原随机策略。

数据：[训练终点](payload_taper_complete_20261119/training_endpoints.csv)、[所有五DEV节点](payload_taper_complete_20261119/all_dev_nodes.csv)、[逐布局配对汇总](payload_taper_complete_20261119/paired_nodes.csv)、[逐项目标检查](payload_taper_complete_20261119/development_criteria.csv)、[来源及收据哈希](payload_taper_complete_20261119/receipt.json)。所有模型/replay/原始记录保留，不删除证据、不使用5080/实机。Panda回归、独立多seed、冻结独立测试和全面最优仍未完成。
