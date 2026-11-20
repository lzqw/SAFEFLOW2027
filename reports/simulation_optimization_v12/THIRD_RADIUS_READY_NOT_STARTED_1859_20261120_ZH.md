# 18:59继续运行与第三Reach扩域准备：未抢跑父终点

2026-11-20 18:57（UTC+8）读取/proc与原生status：四条训练和半小时监测仍在运行。18:59归档原子status进一步确认A100最后Metric新增145K/150K、4080第三Reach父266K/300K，均尚非完成收据，不重启、不提前合并完整900K。

|任务|18:57实际进度|训练安全成功 / 违规 / timeout|
|---|---|---|
|A100 Reach Metric41028|新增144K/150K|4764 / 300 / 1|
|A100 Payload父42028|109K/300K|0 / 102 / 1050|
|4080 Reach第三父41030|265K/300K|5862 / 689 / 846|
|4080 Payload父42029|215K/300K|0 / 193 / 2099|

18:59下载的三份status是稍后的实际状态，并非上表同一时刻：Metric145K、Reach父266K、4080 Payload217K。训练仍按原固定预算进行，半小时监测继续，不因近期指标变差科学早停。

## Payload出现新的价值估计与动作幅度问题，保留后再诊断

4080 Payload从203K的94违规、0成功，到215K193违规、0成功，12K内多99违规。215K日志policy Q均值约5.80、最大19.00、replay Q超过原上界的样本比例25%、action clipping约39.6%；18:59归档217K日志均值约8.64、最大18.44、越界比例34.375%、clipping约43.5%，训练成功仍为0。

这支持“价值估计升高并未对应任务改善，且动作幅度/安全代价同时增大”的诊断；单凭这些日志不能确认唯一原因，也不能把Q上涨写成学会任务。全部仍是有限数值，未观察到NaN/OOM；不触发科学提前停止、不改当前奖励、timeout或安全阈值。相反，A100同任务109K的policy Q仍约−0.33、也无成功，因此不能将两个seed描述成相同数值路径。

来源：[4080 Payload217K实际状态](third_radius_preparation_20261120/payload_parent_observed.json)。此前[全部已观察训练曲线与DEV失败](MATCHED100K_AND_PARENT_REGRESSION_1848_20261120_ZH.md)保留，下一次完整父快照会继续纳入这段真实前缀，不丢弃高违规区间。

## 第三Reach父后续100K原生扩域入口已准备，尚未启动

新增[原生第三seed扩域入口](../../scripts/paper_crax_payload/train_reach_third_radius_20261120.py)，目的为第三训练seed后续P5/Metric对照准备真实400K父，沿用前两个seed同一分布fork：

- 只接受41030原生EA Reach固定300K终点，原父源码`58bf769dceb1eabd2eec07c627decf68d810c2aa`、完整收据、原配置、全部继承成功/违规/timeout必须相符；240K等较好/较早模型或另一seed均拒绝。
- 启动命令必须显式提供真实checkpoint、完成收据、run_config三个SHA256；逐一比对原文件，并确认原父进程已退出、五个DEV文件仍齐全。没有真实300K文件和收据时不能自动占位、替代或先启动。
- 只增加`training_radius_values=[0.045,0.05,0.055]`，从300K到400K新增100K；原网络、replay、优化器、alpha、dual和计数由已有原生fork载入。按原训练seed重新seed随机流，**不是精确RNG resume**，父的全部失败仍继承。
- 继续仅4080 GPU0共享，不暂停Payload或外部项目。新输出/配置/存档留4T盘，每次保存保留40GiB底线及本次完整2GiB、Payload完整2GiB增长预算。
- 不启动新的P5/Metric确认，不增加offset、温度、噪声扫描，不改变原训练horizon/奖励/安全距离；这次仅部署代码准备，零新增训练transition和零仿真控制。

两项针对性测试在4080实际实验环境通过（2 passed，2.54秒）：证明只改变半径分布、保留失败计数；拒绝240K节点、其他seed和被清零的timeout。未重复全套测试、未覆盖任何正在运行源码目录。入口已放在独立`source-third-radius-preparation-20261120`，真实300K终点及SHA尚缺，未执行训练命令、未产生扩域输出。

## 后续

A100最后Metric只余约5–6K训练步，按当前约9.16步/秒外推约9–11分钟，另有最终DEV/保存/半小时汇总开销。第三Reach父约余34–35K，按上轮最近20K约20步/秒外推约半小时，仍需实际完成后确认。两个Payload父继续各300K，未用早期失败换seed。

现有900K终点汇总和本地半小时归档仍live；完成后再合并全部19200个DEV回合。关键程序与本报告上传GitHub、紧凑内容归档本地；未发邮件、创建PR、删除证据或使用5080/实机。Reach多seed最终确认、Payload优化、Panda回归、SafeReach前100K目标和新冻结独立评测仍未完成。

状态来源：[Metric145K](third_radius_preparation_20261120/metric_observed.json)、[Reach第三父266K](third_radius_preparation_20261120/reach_parent_observed.json)、[Payload217K](third_radius_preparation_20261120/payload_parent_observed.json)。这些文件是运行中的真实快照，不能替代原生完成收据或完整曲线。
