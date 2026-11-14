# 固定终点动作结构：离线描述性诊断

协议与程序在`51b0bf99`提交，测量源码SHA写入各结果。没有改动运行中的训练、
300K模型身份、独立评测布局或候选选择。它不重复旧Reach/PointGoal机制实验。

共同状态为原Iso训练轨迹前32条episode初始记录，全部保留，不按后续终类筛选；
是训练来源的开发诊断，不是独立确认或on-policy最终状态分布。
每状态64个动作，Gaussian seed2026111406、K4，六臂复用相同增量。
每个状态独立估计无偏3×3样本协方差，再汇总trace；不把不同状态的均值变化算成策略噪声。

| 固定300K方法 | 状态×draws | 平均最终动作协方差trace | 至少一维在限幅边界的动作比例 |
|---|---:|---:|---:|
| Iso |32×64|0.03401518|0.513671875|

Iso平均协方差对角为[0.01477945, 0.00624003, 0.01299571]。
参考噪声每维方差为0.04、trace0.12；经过策略动力学及限幅后的最终动作方差不同，
因此不能把训练`mean_reference_vn/vt`叫作最终动作`VN/VT`。
边界比例不是违规率；此诊断没有环境查询或真实风险法向投影。
其余五臂尚未完成同预算固定终点，不用中间模型补齐，不预判优劣。

原始[共同状态池](common-pool.json)和[Iso逐状态结果及模型SHA](iso.json)保留。
所有采样在原4080服务器单CPU执行，无训练、optimizer更新、环境导入、仿真控制或实体交互。
完整检查点和replay未传回本地。几何保持输入梯度，未使用全局inference_mode。
一个数值fixture核对无偏协方差、平移不变性和非有限输入拒绝，通过；未重复全套测试。

复现入口（使用新输出路径，程序拒绝覆盖）：

```bash
python -m scripts.paper_crax_payload.component_action_structure \
  --run /data/runs/iso --freeze-pool --output /data/action_structure/common-pool.json
python -m scripts.paper_crax_payload.component_action_structure \
  --run /data/runs/iso --pool /data/action_structure/common-pool.json \
  --output /data/action_structure/iso.json
```

测量工具与结果原路径：
`/home/carla/4T/SAFEFLOW2027_NONHARDWARE_V4/action-structure-51b0bf99/`。
后续在各臂原服务器使用同一池及同版本程序补齐，仅运行已完成300K收据对应模型；
不用该诊断更换状态、seed、参数或检查点。全部完成后再解释组件间动作结构。
