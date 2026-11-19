# 独立父终点后处理：完整两seed、固定300K，不自动挑好父或启动下游

原A100训练仍固定源码`22d4966f`，该后处理放在独立源码目录，绝不覆盖正在运行的代码或调整参数。只读观察原有限队列，每1800秒一次，最多32次；队列完成且实际native句柄均退出后，才执行一次完整结果整理。失败或观察超时保留证据，不自动重试、启动新训练或把观察超时当作训练失败。

完整模式要求两个seed均有原生`receipt.json`的completed300K、`training_receipt_300000.json`、300个逐1K训练日志、所有60K/120K/180K/240K/300K DEV文件。工程smoke、部分曲线、缺失评测和终点计数不一致必须拒绝。每节点三个条件各30个预登记布局，所有终止类别从完整回合重算；自己的DEV不当作独立测试。

输出保留两个seed完整曲线及全部900个DEV回合，分别记录各节点每条件的等权seed均值、样本SD、最好、最差；零成功seed不能被删除。固定终点`full_300000.pt`按流式读取生成真实SHA、文件大小和路径，以便将来另行登记下游时引用实际本seed父，绝不照搬旧41027父SHA。这里SHA用于身份追踪，不代表模型任务表现或独立科学验证通过。

本次只整理已登记600K父训练，不自动做后续100K扩域或三臂150K确认。不根据较好的60K或单seed表现切换父，不修改300K终点选择规则。原任务的Reach57.5/60mm、多seed稳定性、P5对Metric/Iso、Panda护栏和最终独立测试继续保持未完成。

A100遵守storage wrapper，源程序`source-reach-parent-closure/`、观察状态`reach-fresh-parents-closure/`、结果`reach-fresh-parents-complete-analysis/`均在本用户data root。原大型checkpoint不复制到home或本地；不新增GPU任务，只使用少量CPU读取和哈希，保留40GiB加1GiB整理余量。后处理无GitHub/邮件操作；完成后的紧凑报告另由主会话审核同步。
