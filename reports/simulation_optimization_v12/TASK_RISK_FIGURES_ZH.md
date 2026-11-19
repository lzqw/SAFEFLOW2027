# 完整学习曲线图：同时展示安全收益和任务代价

新增八面板PNG/PDF：训练累计违规、累计return、累计timeout、stall episode、累计安全成功，加nominal/radius065/execution085三个开发成功率面板。绘图不改变训练或结果，仅读取严格完成汇总的所有逐1K训练节点和七个DEV节点；阴影仍只表示预登记150K–200K actor LR下降区间。

历史四臂完整1M参考已经实际生成并检查版面：[PNG](task_risk_reference_figures_20261119/samehardware_learning_curves.png)、[PDF](task_risk_reference_figures_20261119/samehardware_learning_curves.pdf)、[来源收据](task_risk_reference_figures_20261119/provenance.json)。它不是新增实验或候选完成结果，原历史指标保持。

当前终点残差候选/原P5对照的绘图进程独立等待`endpoint-residual-analysis/receipt.json`，每1800秒一次、最多32次；只有严格六臂完整汇总完成，才绘图到数据盘`endpoint-residual-complete-figures/`。六条曲线包含两条当前训练及四条历史同4080参考；图例明确historical，不能冒称全为同期训练。全部原节点保留，不能删除25K候选退化而只展示50K恢复。

新图来源仍仅seed2034，各DEV条件每节点30回合；连线不证明节点之间始终领先，无训练seed置信区间或独立确认。最终方法/Piper/Panda要求与图表无关，继续等待真实训练证据。该任务零训练、零新环境、无GitHub/邮件写入；观察期满只退出绘图等待，不停止或重启训练。旧观察/训练源码目录均不覆盖。
