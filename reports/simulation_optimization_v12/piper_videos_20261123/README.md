# Piper 仿真视频

## 常规与困难：同一个 SAFEFlow Reach 模型

[播放 45 mm vs. 60 mm 对比](reach_nominal45_vs_hard60.mp4) · [预览](reach_nominal45_vs_hard60.png) · [回放数据](reach_nominal45_vs_hard60.json)

左侧为常规 45 mm，右侧为困难 60 mm（均指障碍球半径）。起点、目标、障碍中心及检查点相同；两边均安全到达。末端在困难布局中抬高约 8.9 mm，最小净距从 26.14 mm 降到 20.57 mm，仍高于 15 mm 阈值。

## SAFEFlow 与 Metric-only

[播放 70 mm 障碍对比](reach_70mm_safeflow_vs_metric.mp4) · [预览](reach_70mm_safeflow_vs_metric.png) · [数据](reach_70mm_safeflow_vs_metric.json)。同一原布局（12310075），只把障碍半径提高到 70 mm；左侧 SAFEFlow 安全到达，净距 17.78 mm；右侧 Metric-only 在第 2 步低于 15 mm 阈值（14.33 mm）停止，未碰撞。这是单布局压力测试，不是 70 mm 成功率。

以下三段左侧 SAFEFlow，右侧 Metric-only；同一布局、无噪声部署，半速播放。红色是障碍，绿色是目标，黄色是夹持负载。

1. [Reach：60 mm 障碍绕行](reach_60mm_comparison.mp4) — SAFEFlow 到达，Metric-only 触发安全停止。
2. [Payload：57.5 mm 障碍搬运](payload_57_5mm_comparison.mp4) — SAFEFlow 到达，Metric-only 触发安全停止。
3. [Payload：SAFEFlow 失败反例](payload_57_5mm_safeflow_failure.mp4) — SAFEFlow 触发安全停止，Metric-only 到达。

安全停止表示净距低于 15 mm 阈值，不等于发生碰撞。这些是[固定新布局评测](../SELECTED_HOLDOUT_COMPLETE_20261123_ZH.md)中的选例，不是新增评测统计或实机录像。[回放来源与检查点哈希](manifest.json)。
