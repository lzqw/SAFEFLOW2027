# E08：CVPO官方实现接入核对，尚未训练

2026-11-14。此项独立于已经启动的1.8M组件／参数包；正式预算仍是3×10M=30M。

## 原实现与选型

CVPO论文的作者仓库已标注弃用，并指向同作者的FSRL实现，因此优先采用FSRL，
不自行重写一个近似算法后称为官方CVPO。
来源：[作者原仓库](https://github.com/liuzuxin/cvpo-safe-rl)、
[作者FSRL仓库](https://github.com/liuzuxin/FSRL)。

本次用`git ls-remote`核对FSRL HEAD为`e056fc9498d5d037869533da7cf976acf462f918`。
如进入接入实现与正式运行，必须固定此提交，不能让后续main分支变化悄悄改变算法。

## 不是开箱即用

- FSRL使用Tianshou的Batch／ReplayBuffer与Gymnasium接口；本项目PointGoal是
  JAX/MJX向量状态＋Torch的四返回接口，episode结束后的终态观测与下一回合观测分开。
  必须显式适配terminated/truncated和回放索引，不能直接把自动reset后的状态当Bellman终态。
  依据：[官方CVPO源码](https://github.com/liuzuxin/FSRL/blob/e056fc9498d5d037869533da7cf976acf462f918/fsrl/policy/cvpo.py)、
  [官方Collector](https://github.com/liuzuxin/FSRL/blob/e056fc9498d5d037869533da7cf976acf462f918/fsrl/data/fast_collector.py)。
- 官方Agent要求环境提供`spec.max_episode_steps`；其成本阈值会用gamma和原生horizon换算。
  保持原生成本、预算25和horizon，不直接把25当每步cost-Q阈值。
  依据：[Agent接口](https://github.com/liuzuxin/FSRL/blob/e056fc9498d5d037869533da7cf976acf462f918/fsrl/agent/cvpo_agent.py)。
- A100现有CRAX运行环境在原`PYTHONPATH=/home/lzqw/SAFEFLOW2027_PAPER/CRAX`
  下可找到CRAX/JAX，但没有FSRL与Tianshou。ARM训练环境也没有这两者。
  官方setup依赖包含`tianshou~=0.5.0`及旧protobuf约束，不能对正在共享运行的环境直接升级／降级。
  依据：[官方依赖](https://github.com/liuzuxin/FSRL/blob/e056fc9498d5d037869533da7cf976acf462f918/setup.py)。

## 接入时的最小工作

使用数据盘上的独立环境与固定源码，桥接原生环境，不换成Safety-Gym同名任务。
保留观测、目标刷新、动作边界、原生成本25及共同独立评测规则；不提供额外几何信息。
记录官方算法实际更新次数、墙钟时间、失败前缀及所有训练成本；正式seed沿用51027/51028/51029。
先验证一次真实采样与官方学习更新，再单独冻结30M计划后启动，不能把接口测试当成正式结果。

当前状态：完成文档与环境兼容性核对；未安装、未接入训练、未消耗30M预算。
