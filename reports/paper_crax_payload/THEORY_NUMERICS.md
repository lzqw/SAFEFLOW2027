# M4：小型线性 Gaussian 数值实验

实际运行：`python -m scripts.paper_crax_payload.mechanism_theory --output reports/paper_crax_payload/THEORY_NUMERICS.json`。
种子20261112；每项 Gaussian 实验200000次 synthetic draws；共400000次 Gaussian draws、
40000个随机 SPD 候选。新增训练、模拟控制查询、真实交互均为0。

| 核对 | 实际结果 |
|---|---|
| 终点方差 `dt gᵀΦDΦᵀg` 与 pullback `dt(Φᵀg)ᵀD(Φᵀg)` |解析恒等；Monte Carlo 相对误差0.069864%|
| 法向错位 `||Φᵀg||²[vN+(vT-vN)sin²θ]` |0/15/30/45/60/90度全部解析矩阵核对通过|
| 固定trace constrained log-det |active/inactive两种约束；KKT stationarity最大残差1.42e-14与3.55e-15；无可行随机候选超过解析最优|
| 同 covariance Gaussian 的 KL |解析1.501840672；Monte Carlo误差0.837个标准误差|

`constrained_logdet` 明确以单位法向的方差上限为约束。若约束写成一般梯度
`gᵀDg≤c`，需先用 `c/||g||²` 归一化。固定trace不是固定所有方向方差；
椭球体积/log-det也不是任意非线性机器人动力学安全或泛化的证明。

相关最小数值与采样语义测试实际5通过。理论实验原始矩阵、每个角度误差、
Monte Carlo标准误差、全部解析最优与KKT残差见 [THEORY_NUMERICS.json](THEORY_NUMERICS.json)。
