# Piper 仿真视频

## 连续半径扫描：Reach → Payload

[播放完整视频](continuous/piper_reach_then_payload_radius_progression.mp4) · [预览](continuous/piper_reach_then_payload_radius_progression.png) · [逐档汇总](continuous/summary.csv) · [逐布局结果](continuous/rollouts.csv) · [模型及视频校验信息](continuous/manifest.json)

两项任务的几何覆盖训练均使用 **45、50、55 mm** 的球形障碍半径。视频先展示 Reach（45 → 50 → 55 → 60 → 65 → 70 → 75 mm），再展示 Payload（45 → 50 → 55 → 57.5 → 60 → 65 → 70 mm）。左为 SAFEFlow，右为 Metric-only。模型冻结、无噪声推理，同一起点、目标和障碍中心，仅改变障碍半径。

每项任务展示一个已选布局，并另用 20 个固定布局做逐档配对统计；回放布局不计入这 20 个。原布局不能安全初始化时不替换，也不计为途中失败。安全停止表示净距跌破 15 mm 阈值，不等于碰撞。这是每任务一个已选训练 seed、复用已有布局的探索性扫描，不是独立测试或实机结果。

**片段时间：** 00:00 Reach 训练设置，00:26 Reach 70 mm，00:34 Reach 汇总，00:40 Payload 训练设置，00:59 Payload 57.5 mm，01:12 Payload 汇总。

- **Reach 70 mm：** 20 个布局有 7 个可初始化，SAFEFlow 安全到达 7 个，Metric-only 到达 4 个。到 75 mm 时，演示布局不能初始化，统计批次也仅剩 1 个可运行布局。
- **Payload 60 mm：** 演示布局上两者均安全停止；另测的 20 个布局中有 17 个可初始化，SAFEFlow 安全到达 17 个，Metric-only 到达 13 个。65 mm 时只有 13 个可初始化，两者分别到达 6 个和 5 个。

[评测与渲染脚本](../../../scripts/paper_crax_payload/render_piper_radius_progression.py)使用原冻结检查点和 Piper 仿真资产；逐布局 CSV 记录评测布局和终止类型，清单记录检查点哈希。

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
