# E08：CVPO官方实现接入核对与正式协议

后续已补齐正式入口、完整收集边界恢复、五节点共同验证及验证后一次独立测试程序，
见[正式协议](E08_FORMAL_PROTOCOL_ZH.md)。正式30M尚未消耗；控制器1831775已实际等待
组件释放后的共享槽，非训练worker。
下文“没有正式训练入口”为此前工程检查时的历史状态，不再代表当前代码。

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

## 后续实际工程验证

已将固定官方源码与最小依赖装入数据盘独立overlay，不更改现有环境。
Tianshou0.5.1使用NumPy1.26.4／Numba0.61.2／llvmlite0.44.0；
其他新依赖包括Gymnasium0.28.1和日志导入依赖。未运行官方setup的全依赖安装，
未将其旧protobuf约束强加给共享环境，官方CVPO算法源码未修改。
数据盘`noexec`不能加载C扩展，因此NumPy／grpc／Numba／llvmlite的小型执行副本在
`/tmp/safeflow-cvpo-exec.n27CHx`，源码、wheel、输出和检查点仍在数据盘；重启后需重建该临时副本。
这与用户要求大文件不堆入home一致，不做系统挂载改动。

真实结果：

- [合成桥接收据](e08/synthetic-smoke-v1.json)：官方CVPO两次学习更新，
  12个合成fixture转移，原生采样0；终止不bootstrap、超时保留终态bootstrap已数值核对；
  raw replay不改写，样本使用当前Acme归一化；成本阈值由25和H1000/gamma0.99换算为2.499892。
- [原生工程收据](e08/native-smoke-v1.json)：原版CRAX63504b5的safe_goal_point/level1，
  原生62维观测、2维动作、目标刷新、原生成本和H1000；CPU精确32个真实转移、
  官方CVPO两次学习更新，零实机。初始化／编译之后部分耗时79.04秒，
  不用短CPU工程样本预测GPU正式吞吐，也不把32步零奖励／零成本当方法表现。

`cvpo_bridge.py`的工程入口仅支持合成或32步原生smoke，**没有启动正式10M训练的入口**。
原生环境／replay／优化器／dual／normalizer／RNG已写完整工程快照，保留服务器数据盘。
官方`get_extra_state`未保存dual与优化器；桥接代码显式另存这些字段，不冒称仅Actor即可续训。
实际工程转移32另计，不混进1.8M组件包或尚未消耗的30M正式预算。

下一步是登记并实现完整训练／评测／恢复协议，然后在共享槽释放后运行正式基线。
当前状态：接口和真实学习更新已接通，正式基线尚未运行，没有CVPO任务效果结论。
