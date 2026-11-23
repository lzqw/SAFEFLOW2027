# 已按新口径推进：两个有效seed模型的新布局评测正在运行

2026-11-23 11:03（UTC+8）实际核验Reach/Payload两个原生评测进程均live，已生成原生run_config并进入原生仿真/策略计算；当前不是在追加训练，也没有新的最终成功率结果。此前[模型选择说明、完整三seed证据与评测冻结](EVIDENCE_SCOPE_AND_SELECTED_HOLDOUT_20261123_ZH.md)在运行前已提交并上传（科学源码c2069cfb、发布87fa53bb）。

## 运行任务

|任务|选定训练seed / 固定模型|同seed、同预算基线|回合预算|原生PID|
|---|---|---|---:|---:|
|Reach|41029 / P5+2mm固定550K|Metric-only固定550K|四条件×200×两模型=1600|1661850|
|Payload|41027 / P5+4mm固定220K|Metric-only固定220K|四条件×200×两模型=1600|1661852|

两边一律原target actor、zero_noise完整部署模式，使用预先固定的新布局12310000/12320000起，不混合历史DEV或随机部署Metric。新holdout不用于再挑模型或seed；如果之后依据它继续调参，就不再把它当新版本的独立确认。模型/seed是基于历史DEV回顾选择，结果仅对这些选定模型及新布局范围成立，不证明总体跨seed稳定或所有任务全程最优。训练风险不能因部署评测被改写。

11:03整卡快照GPU利用率61%、显存1172MiB，包括共享其他项目负载；不暂停其他项目，不为了显存填满增加重复训练。输出、缓存、临时文件均留4080的4T数据盘，约395G空闲，保留40GiB+2GiB紧凑结果预算。A100数据盘约22G，继续不向该盘追加大文件或回退home。

## 启动前工程修复与不重复选结果

第一次启动在官方URDF资产SHA检查处因字符串没有`Path.open`方法退出，两任务均exit1，**没有创建评测输出目录、没有加载策略或开始仿真回合**。修复仅把资产路径转成Path，不改变固定协议、模型SHA、回合数、模式或布局。原失败队列和日志保留；没有删除不利科学结果或按评测结果重试。

修复后逐个实际加载四个原checkpoint，验证原生格式、计划、arm、seed和完成收据完全匹配，检查官方资产SHA；这些预检查不调用策略、不运行物理。此前两项针对性协议测试已通过，不再重复全套。新源码b6192b9bd264dd4731ee5f9edfcec209edf00f2b在独立source-selected-holdout-20261123-v2运行，原固定评测协议文件SHA保持不变。

显式第二队列1661848，两个进程1661850/1661852、半小时资源监测1661853；状态路径`queue_selected_holdout_20261123_v2.json`，不覆盖第一次失败队列。11:03真实/proc、原生run_config、PyBullet及CUDA计算日志确认这次已越过资产预检查。没有为看到结果而更换seed/模型或重新试验。

## 结果交付与监测

服务器只读资源监测每1800秒一次。本地systemd服务`safeflow-selected-holdout-archive-20261123`已启动（PID1661853是远程资源监测，不是本地服务），按半小时只读观察精确第二队列；全部结束后只下载JSON/CSV，并检验每任务1600回合完成收据。失败/部分任务不会标记为科学完成，不自动启动/重启评测、不写GitHub、不发邮件。归档进程最多48次观察，24小时后结束；若两任务提前完成则收齐后退出。

完整三seed1.35M训练/28800DEV已汇总上传；失败seed、固定P5崩塌、Payload新父零成功仍公开。现在不再为每seed最优继续消耗训练，先让有效案例的新布局对照给出下一步依据。Panda回归、SafeReach早期目标、Payload更高困难几何等剩余要求没有自动完成，整个研究目标仍未结束。

来源：[当前实际队列](selected_holdout_execution_20261123/queue_selected_holdout_20261123_v2.json)、[Reach原生登记](selected_holdout_execution_20261123/Reach/run_config.json)、[Payload原生登记](selected_holdout_execution_20261123/Payload/run_config.json)、[零回合的预检查失败队列](selected_holdout_execution_20261123/preflight_failed_queue.json)、[冻结评测协议](../../configs/paper_crax_payload/selected_seed_frozen_holdout_20261123.json)、[只读半小时归档入口](../../scripts/paper_crax_payload/archive_selected_holdout_20261123.py)。
