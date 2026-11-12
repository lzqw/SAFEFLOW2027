# 新机制实验实现与执行边界

## 已实际执行

M4四项线性数值实验已经完成且通过，见 THEORY_NUMERICS.json/md。只有 synthetic
Gaussian/SPD draws；0新增训练、0模拟控制、0真实交互。最小数值/采样语义测试5通过。

## 已实现、尚未启动物理查询

`scripts/paper_crax_payload/mechanism_sampling.py` 不接受 oracle 参数。它复用被冻结
Actor的velocity和原有真实clamp deterministic tail；不改变旧训练和历史代码。

- M1：Iso-High、Iso-Low、Normal-Shrink、Directional-Full。有效q=1位置的默认谱分别为
  `.04/.04/.04`、`.01/.01/.01`、`.01/.04/.04`、`.01/.055/.055`。
  High与Full匹配注入trace；Low/Shrink/Full匹配法向方差，但切向与trace不同。
  不声称终点trace或所有匹配条件同时相等。
- M2：Local、Endpoint-Direct、Endpoint-Pullback、Random。每个干预使用共同Actor、
  预先生成的共同noise，Random使用独立且单动作全程固定的随机轴。q=1和rho相同；
  learned梯度无效时fallback可破坏全体谱匹配，因此另列联合精确谱匹配样本数和风险。
- Sampling与energy使用同一个实际D。冻结更改reference variance/rho只作为采样器干预，
  不假称temperature重新训练；M1能量只诊断，所有alpha学习均冻结。
- K=2/4/8、rho=0.1/0.25/0.5/1.0只作为冻结模型采样器敏感性，不是完整重训。

`scripts/paper_crax_payload/mechanism_arm.py` 是可运行Reach机制runner：输入全部四个
已完成固定1M的Iso/EA旧策略，通过原生receipt与配置检查后，构造**新的**128个不同状态、
每状态64draws；每个来源policy/条件16状态，同时保留安全内部和近边界。
新池布局seed260027系列，不把旧64状态复制填充。Oracle在动作已经生成并detach后才调用，
只做共同真风险/投影轴评估；先验选action路径没有oracle。

该runner仅对保留的Reach checkpoint直接支持；不能声称它已经处理Payload或CRAX，
两者需要各自真实环境oracle/状态重建适配。PointGoal只需少量共态分析，不复制完整矩阵。

可选M3 `--routes`：每Actor × native/common-Iso/common-Pullback ×两条件 ×30episode。
共同270027布局，统计安全成功/违规/超时、return、capped时间、成功路径扇区和3cm voxel
安全成功路线覆盖。覆盖是小型代理，不等同entropy；分离Actor与test-time sampler效果。

延迟测量为同一冻结Actor单动作100次，CUDA同步、5次warmup、p50/p95。
此测量不冒充完整训练SPS；训练速度/编译/显存由主训练程序另记。

实际运行必须由主协调者安排资源和台账：

```bash
python -m scripts.paper_crax_payload.mechanism_arm \
  --runs /path/to/iso2027 /path/to/iso2028 /path/to/ea2027 /path/to/ea2028 \
  --output /new/analysis/output --source-commit ACTUAL_SOURCE_COMMIT \
  --device cuda:0 --workers 10 --routes --parameter-sensitivity
```

输出目录拒绝覆盖。receipt实际计算新池采样区间、有限差分控制区间、margin-only模拟查询、
路线区间以及0训练/0optimizer/0真实交互。未运行时不生成成功receipt，不把代码准备完毕
写成机制证据完成。父级协调者尚未授权启动该runner，故当前M1/M2/M3物理成本为0。
