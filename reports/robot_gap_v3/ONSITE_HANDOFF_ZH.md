# 下一步现场交接：尚未执行，不是实机结果

仿真结果见[本轮总汇总](NEXT_STAGE_RESULT_ZH.md)。E01/E02/E03和日志复用已完成，
不再追加训练。当前唯一P0缺口是实际部署电脑计时、现场标定和受保护Reach90。
本文件仅提供后续操作步骤；本次没有登录5080、安装依赖、打开机器人接口或发送邮件。

## 1. 获得5080短时空闲窗口后：只做离线CPU计时

使用用户指定的home下`LZQW`项目目录、现有Python环境；不要为此升级共享环境。
先确认目录确实是SAFEFLOW仓库且含冻结bundle，代码使用本分支已交付版本。
模型是原生Piper34维，不可拿本次Panda45维模型替代。

以下从SAFEFLOW项目根目录运行，输出示例目录必须尚不存在。程序只读取仓库内的
仿真观测流，不采集现场传感，不连接CAN，不打开SDK，不发送动作，不使用GPU。
三个方法同样200次计时＋20次warmup。此命令未在5080执行。

```bash
test ! -e reports/deployment_pc_offline_window
for method in iso iso-low ea; do
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m deployment.piper.replay_policy \
    --bundle "reports/robot_gap_v2/reach_adapted_release/bundles/$method" \
    --observations reports/robot_gap_v2/reach_release/observations.jsonl \
    --output "reports/deployment_pc_offline_window/$method.json" \
    --device cpu --benchmark || break
done
```

保留p50/p95/p99、原始时延、CPU型号／线程数／负载和软件版本；结果由服务器下载本地后再上传GitHub。
没有实测控制周期时不传`--period-s`，deadline计数保持null；不拿仿真40ms当作实体标定。
即使在5080完成，该结果也仅是完整策略推理，不包括观测获取、通信、反馈、执行延迟；
因此不能标记E09全链路完成。程序中的“未认证部署机器”保守标签无需为了结果好看而删除。

## 2. 必须由现场测量／确认，不通过远程猜测补全

现有`deployment/piper/hardware_pending.json`仍含未知项，先保留原文件，
在新的现场记录中附日期、测量方式及负责人：

- 序列号／固件、夹爪实际开度、TCP相对法兰的米制位姿、基坐标变换及四元数xyzw约定。
- 动作单位／尺度、实际反馈与控制周期、观测和通信端到端时延、时钟同步方式。
- 当前负载、保守速度／工作空间边界、受保护标准布局、急停测试和现场操作人员。

标准Piper及标准夹爪已经由用户确认，不必再次询问型号。
仿真极限值／通用厂家极限不能替代现场配置；未提供值保持null，不开启运动权限。
离线计时窗口不等于实体运动授权。

## 3. Reach90执行前需要新的现场安排

现有90回合计划为3方法×3条件×10轮，全部仍是planned/not executed。
当前仿真Reach几何压力条件风险高，不能把它们直接照搬为现场无保护布局。
现场应先标定并验证有安全裕量的共同标准布局与保护层，再单独确认真实条件与执行授权。
实机接口／观测映射尚未实现并验收；现有offline程序根本不能执行实体动作。

实际记录必须区分策略原提议、保护后命令、测得执行、无介入安全成功、保护介入、
真实违规／碰撞、最小间距与失败封顶用时。Iso-Low的零任务成功也保留，
不能静默替换成另一更容易获胜的对照。冻结策略比较不顺带开展在线RL或Payload实机。

## 当前所需外部条件

先安排不影响5080其他工作的短时离线计时窗口；然后由现场人员提供标定与保护条件，
并另行确认实体接口测试／受保护试验安排。在这些条件到位前，保持实体交互为0。
不新建重复仿真任务填补等待；不以GPU满载替代这些缺失证据。
