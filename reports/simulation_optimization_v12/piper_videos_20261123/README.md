# Piper 仿真视频

左侧 SAFEFlow，右侧 Metric-only；同一布局、无噪声部署，半速播放。红色是障碍，绿色是目标，黄色是夹持负载。

1. [Reach：60 mm 障碍绕行](reach_60mm_comparison.mp4) — SAFEFlow 到达，Metric-only 触发安全停止。
2. [Payload：57.5 mm 障碍搬运](payload_57_5mm_comparison.mp4) — SAFEFlow 到达，Metric-only 触发安全停止。
3. [Payload：SAFEFlow 失败反例](payload_57_5mm_safeflow_failure.mp4) — SAFEFlow 触发安全停止，Metric-only 到达。

安全停止表示净距低于 15 mm 阈值，不等于发生碰撞。这些是[固定新布局评测](../SELECTED_HOLDOUT_COMPLETE_20261123_ZH.md)中的选例，不是新增评测统计或实机录像。[回放来源与检查点哈希](manifest.json)。
