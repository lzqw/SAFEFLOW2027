# 同硬件两臂已运行：2026-11-19 10:13（UTC+8）快照

4080两条原生训练已启动并实际完成6000/250000 transition，不是仅建立队列。完整P5 PID3272468，Iso PID3272470，队列PID3272465；源码`2c06ca2a`。起始网络SHA与原四臂一致，当前仍为相同早期前缀，不能据此比较方法效果。

训练代码/参数均沿用原250K；单项初始化、共享配置、Iso开关和lateLR时序测试在4080正式环境通过（1 passed），未扩大测试套件。本地Python缺仿真依赖，所以未用本地环境报通过。

4080实测GPU利用率87–99%是整卡瞬时值，包含共享项目，不能全部记作本实验利用率。数据盘剩1210.5GiB；大轨迹、replay和七个milestone全部写`/home/carla/4T/SAFEFLOW2027_SIMULATION_V12/runs-samehardware/`，未放home系统盘。没有停止其他进程或使用5080/实机。

预计这批两条并行训练共约3–5小时，按历史完整训练耗时估计，不用含随机前缀的当前平均sps外推。若共享负载变化会相应延长。半小时只读监测已实际运行，首条记录时间10:13:32，下一次约10:43:32；监测只存状态，不自动修改算法或重试任务。

监测目录`half-hour-monitor-samehardware/`，日志`logs/monitor-samehardware.log`；队列状态`queue_samehardware.json`。本地[启动收据快照](samehardware_launch_20261119.json)是历史记录，后续实时情况以服务器原生状态/receipt为准。

A100两条此前250K均完成，目前数据盘仅约56GiB可用，暂不新增大存档任务；保持40GiB底线和增长余量，不为占卡重复无效实验。没有额外删除任何存档。

[本轮固定预登记](SAMEHARDWARE_CONFIRMATION_PREREGISTRATION_ZH.md)；[已经完成四组消融的全部结果](MATCHED_ABLATION_FINAL_20261119_ZH.md)。本轮尚无新DEV终点，不宣称已达成Reach/Payload/Panda或全baseline目标。
