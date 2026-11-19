# 有限差分诊断：不支持历史末期P5法向普遍错误，当前候选早期风险仍未解决

原登记`db41080a`已在A100 CPU完成。128个固定历史真实状态均在预设容差内重建：原动作interval-min margin最大重放误差`1.3090e-7 m`，约0.000131mm，低于事前2e-6m门槛；没有放宽容差、剔除状态或重选样本。全部38个真实危险动作保留。随后对三个坐标各±0.002做有限差分，并重放原动作核对快照恢复；总1024个控制区间、20,480物理substeps，另一次初始化20 settling substeps，无训练/optimizer更新或策略动作选择。

|冻结geometry EMA模型|全体非flat112状态signed cosine均值|38个真实危险动作cosine均值|危险动作反向数|全体非flat反向数|
|---|---:|---:|---:|---:|
|P5|0.908|0.976|0 / 38|4 / 112|
|Iso|0.901|0.970|0 / 38|5 / 112|
|Metric|0.901|0.967|0 / 38|2 / 112|

P5在38个危险动作的axis cosine平方均值0.953、margin MAE0.425mm；既有数值校准与这组动作梯度方向均没有显示“比Metric明显更差”。其中16/128状态的真实有限差分梯度为flat，均保留但不计算方向cosine；实际边界±5mm的67状态中flat2，安全内部60状态中flat14。不能把真实flat零向量与模型法向相乘后称作方向正确或错误。

head agreement≥0.8这一预登记分析层中，P5共有103状态，4个true-flat，99个有效比较平均cosine0.960、反向0；Metric100状态、4个true-flat，96个有效比较平均cosine0.960、反向0。这个分层只作诊断，不新增行为门控，不把agreement当可靠概率。全体P5仍有4个非flat反向状态，不能声称所有法向准确。

**结论边界**：这三个模型是既有seed2034固定250K终点，不是当前终点残差候选的100K模型；输入是各自最后10K训练动作的混合池，38危险+90固定安全分层样本，不代表自然访问分布比例，可能in-sample。测量的是实际终点动作处的直接margin梯度，不是flow各注入点deterministic-tail pullback法向，也不是新DEV访问状态。由此只能反驳“历史末期危险动作的法向普遍学反”这一特定解释，不能证明方向机制总体最优、当前候选的早期梯度正确或某项改动的训练因果。

因此当前不更换法向估计器、不增加critic或温度/阈值扫描，也不因这个好的局部诊断晋升前100K违规394的已不合格候选。继续完成既定250K的完整任务—风险曲线和两个300K独立父模型；下一方法修改须由完整证据支持，不能凭另一个有利局部指标掩盖训练失败。

来源与全量数据：[预登记](safereach_normal_truth_20261119/registration.json)、[全部128原始状态/动作](safereach_normal_truth_20261119/selected_actual_actions.json)、[每状态原标签重放误差](safereach_normal_truth_20261119/reconstruction.csv)、[真实有限差分梯度](safereach_normal_truth_20261119/true_gradients.json)、[全部384模型—状态比较](safereach_normal_truth_20261119/model_normal_comparisons.csv)、[固定分层统计](safereach_normal_truth_20261119/summary.csv)、[检查点SHA和完成收据](safereach_normal_truth_20261119/receipt.json)。大型检查点未复制到本地或GitHub，正在训练的源码/参数未修改，无实机或5080任务。
