# 首个固定20K节点：有训练安全收益，仍有任务条件短板

2026-11-17 14:29 UTC。只解读首个登记的20K DEV，不打开独立测试、不选中间模型、不科学早停。
四条已启动训练均会继续收集到新增100K；Piper Payload两条等待A100有限槽释放。

## 同新增20K的全部训练代价

| 任务 | control违规／成功／超时 | P5违规／成功／超时 | 违规减少 |
|---|---:|---:|---:|
|Piper Reach|45／657／0|18／645／0|60.0%|
|Panda Payload|110／1029／0|69／1012／0|37.3%|

两项都没有通过增加超时减少违规，但累计成功各少约1.8%/1.7%，需要在完整预算继续观察。
这只是单个原生父状态的配对，不是跨seed确认或完整100K结论。

## 全部20K固定DEV条件，每条件30回合

表格为安全成功／违规／超时计数。

|任务／条件|control|P5|
|---|---:|---:|
|Piper Reach nominal|30／0／0|30／0／0|
|Piper Reach 57.5mm|23／7／0|24／6／0|
|Piper Reach 60mm|17／13／0|21／9／0|
|Panda Payload nominal|29／1／0|30／0／0|
|Panda Payload 70mm|29／1／0|30／0／0|
|Panda Payload wide|29／1／0|27／3／0|
|Panda Payload execution085|30／0／0|30／0／0|

Reach P5此节点nominal100%、57.5mm80%、60mm70%，刚达到预设的第一阶段任务目标，
但两种未见半径的违规仍达20%/30%，不是最终逐条件≤10%的安全目标。
不将当前开发30回合当作独立成功证明，不以较好20K替代预定100K。

Panda P5在加宽载荷仅90%，低于95%目标，而且低于同轮control的96.7%；
其余条件好不能覆盖这个负结果。完整训练结束前不追加课程、裕量或其他候选。
原来的历史修复457次只作背景，本次风险比较始终用当轮control。

[完整片段训练曲线](first20k_analysis/learning_curves.png)、
[全部训练进度CSV](first20k_analysis/all_training_progress.csv)、
[全部14个DEV条件CSV](first20k_analysis/development_curve.csv)、
[相同新增步数的风险／成功／超时CSV](first20k_analysis/matched_training_costs.csv)。
快照包含采集时已经完成的所有训练行，不裁掉20K之后的数据；DEV只包含首个固定节点。
统计仅读既有原生输出：新增训练0、额外仿真0。下一次资源观察仍由30分钟只读进程执行。
