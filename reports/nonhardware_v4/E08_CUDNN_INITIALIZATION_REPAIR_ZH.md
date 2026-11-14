# CVPO首次初始化失败与人工加载路径修复

2026-11-14 22:34 UTC。不是科学结果失败，也不是已完成正式训练。

原调度器1842989于22:33接纳worker1863821，日志明确：运行时加载cuDNN9.1，
JAX CUDA插件编译要求9.8及以上；`PointGoalTorch`构造首个环境时失败。
该失败在run_config／输出目录创建、原生reset、replay收集及optimizer更新之前，
因此正式训练transitions=0，不把工程初始化冒充已消费10M预算。
原worker已退出，旧队列尚未轮询发现，不把其旧running字段当活跃证据。

已只停止调度器1842989，保留原日志和待接续三个seed／闭合；
没有向训练worker或其他项目发信号。4080的rho0.5已由原队列正常启动，独立继续。

当前Torch2.7.1+cu118来自原Anaconda环境，JAX/JAXlib0.6.0和
`nvidia-cudnn-cu12`9.28.1.4已在PAPER环境安装。
修复仅给四个CVPO进程指定项目级`LD_LIBRARY_PATH`：
`/home/lzqw/SAFEFLOW2027_PAPER/env/lib/python3.11/site-packages/nvidia/cudnn/lib`。
不更换算法、JAX/Torch版本、系统库或其他项目环境；不下载、删除或升级全局依赖。
库加载路径的作用亦见[JAX官方安装说明](https://docs.jax.dev/en/latest/installation.html)。

一次最小GPU联合探针通过：按原入口顺序import Torch再JAX，
JAX两元素jit加法在cuda:0执行完成，Torch两元素平方梯度为[2,2]，
实际cuDNN版本92801。零新训练、零仿真控制，未宣称它替代正式CRAX训练验证。

人工接续配置`cvpo_queue_a100_v4c.json`保留原科学源码7abb6ebb、
正式配置、三个seed、每seed10M与独立闭合协议；仅增加该加载路径、
新日志名及修复说明。旧原生队列状态将保存在数据盘`cudnn-runtimefix-v1/`，
新调度器继续稳定state路径供原观察器读取；原失败日志不覆盖。
原失败未生成run输出，原三个注册输出身份仍可使用；启动前需确认不存在原生存活worker。
这是针对已定位初始化错误的一次人工修复，不开启自动失败重试；
后续异常保留证据，不盲目重新提交。

22:37已实际接续，新调度器1864682、worker1864692，原生`/proc`命令与run_config均匹配。
原状态已归档，[原错误日志](cvpo_runtimefix/initialization_failure.txt)、[旧队列](cvpo_runtimefix/retired-queue.json)及
[修复收据](cvpo_runtimefix/repair_receipt.json)同步到本地。
22:38已通过原CRAX构造／reset并创建初始完整检查点，首块编译／采集仍在进行，
尚无进度行，不把进程启动或探针通过称为完整训练验证。

后续22:41已实际进入训练，首轮实测36864步、1280官方更新；
[首份取回原生状态](cvpo_runtimefix/first_training_status.json)为49152步、2048更新，
[配置与环境身份](cvpo_runtimefix/native_run_config.json)保留。
22:44进一步复用GPU1原组件释放槽，见[只转移尚未启动seed的接续记录](E08_RELEASED_SLOT_HANDOFF_ZH.md)。
原seed51027 worker始终保留，无再次重启或科学配置变化。
