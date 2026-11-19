# 2026-11-19约15:16（UTC+8）：Panda/SafeReach部署评测已实际启动

五个白名单原生checkpoint及receipt已完整复制到A100数据盘，五个SHA均与登记配置一致，4080源文件未删除。源码`f54300e1`，有限单槽队列PID4097182，原生评测PID4097192；从原生manifest/日志确认已进入Panda真实仿真，不是仅队列等待。输入目录`panda-safereach-mode-inputs/`，输出`panda-safereach-modes-evaluation/`，均在storage0根目录下。

固定Panda两模型、SafeReach三模型，各sampled/zero_noise、100新DEV布局/条件，共3400回合；不纳入正在训练的taper，不加载其他机器人的模型、不更改训练/环境定义。只读模型/计数核对、模式切换/恢复和真实计时/回合字段归一化的小型单项测试通过（1 passed）；未扩大验证套件或执行额外工程rollout。

所有新source/cache/TMP/输出通过storage wrapper写数据盘，显存allocator .025、GPU0共享，不停止已有外部项目。该评测不更新任何训练网络/dual/replay，前后参数哈希和counter检查；GPU1的独立父seed41028/41029保持运行，4080候选也不变。半小时只读监测已启动，目录`half-hour-monitor-panda-safereach-modes/`，队列`queue_panda_safereach_modes.json`，不自动重试或晋升方法。

预计20–60分钟，受共享负载影响；完整结果到齐后同时报告同模型模式变化、同模式Metric/Iso比较、Panda逐条件≤2pp的开发护栏，不能只选P5较好的模式或把固定P5部署检查当最终新训练配置已通过回归。当前尚无完成收据，不报告效果。

已完成Piper4200回合的[全模型配对图](deployment_modes_figures_20261119/piper_deployment_modes.png)及[论文排版PDF](deployment_modes_figures_20261119/piper_deployment_modes.pdf)也已生成：三个模型与两种模式都保留、纵轴从0开始、无单seed伪误差条，虚线仅Reach用户成功阈值。图中四个困难条件之外的nominal/execution仍完整保留在42节点总表，不从验收删除；所有新结果均为DEV，不是独立test。
