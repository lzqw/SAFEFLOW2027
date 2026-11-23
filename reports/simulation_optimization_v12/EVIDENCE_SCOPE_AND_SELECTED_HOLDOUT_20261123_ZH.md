# 证据口径更新：不要求每个seed最优，保留完整结果并推进有效模型

2026-11-23用户明确：“可以继续推进，不必每个seed都去最优的，只要有seed作为证据就可以”。因此后续不再把每seed同时最优、最差seed阈值或SD上限作为继续推进/采用成功案例的必要条件，不再为救好每个seed追加训练。它们仍作为描述性稳定性信息保留；**不改变历史失败计数、不只发布最好seed，不把成功案例写成总体稳定性证明**。

## 完整三seed开发对照已收齐

三个真实独立父seed41028/41029/41030、每seed三个方法，各新增150K，共1.35M新增训练transition、288个条件/部署节点、28800个DEV回合。11月21日最后三方法与Payload恢复都已exit0完成，自动归档进程收齐后正常退出；目前不是还在跑原队列。

|方法|sampled57.5mm三seed|sampled60mm三seed|zero_noise57.5mm三seed|zero_noise60mm三seed|
|---|---|---|---|---|
|固定P5|97/100/0%|86/98/0%|100/100/0%|99/100/0%|
|P5+2mm|98/99/88%|91/97/74%|100/100/96%|97/100/88%|
|Metric-only|92/93/32%|86/85/26%|96/100/37%|94/95/32%|

固定P5第三seed的四条件、两部署模式全部0成功，大量timeout/stall，不能删掉该seed。+2mm在第三seed恢复成功，nominal/execution085两模式均100%，困难条件也明显优于同seed Metric，但sampled60mm74%仍不足90%。全seed均值sampled60mm为候选87.33%对Metric65.67%；zero_noise60mm95%对73.67%。这支持存在有效配置和任务优势，而不是所有seed都达到绝对阈值。

|同预算三seed合计训练后缀|违规|安全成功|timeout|stall episode|累计return|
|---|---:|---:|---:|---:|---:|
|固定P5|240|12190|719|739|127292.89|
|P5+2mm|199|14560|4|5|154779.49|
|Metric-only|1342|12788|547|568|123741.24|

候选合计违规少85.17%、安全成功多1772、timeout/stall显著更少、return高25.08%；这些全部包含失败seed。由于第三seed Metric很差，它对合计差异影响大，不能用总体计数掩盖前两seed候选训练成功减少和timeout增加。另在第三seed单独比较，候选79违规、4859成功、1timeout/2stall，对Metric767/2790/546/567，支持这一seed同时改善安全与任务。以上仍是400K→550K后缀，不是SafeReach从头学习曲线。

数据：[全三seed汇总收据](three_seed_complete_20261123/receipt.json)、[全288节点](three_seed_complete_20261123/development_nodes.csv)、[28800回合](three_seed_complete_20261123/development_episodes.csv)、[完整1K曲线](three_seed_complete_20261123/training.csv)、[逐seed训练终点](three_seed_complete_20261123/training_endpoints.csv)、[三seed等权终点均值/最好/最差/SD](three_seed_complete_20261123/three_seed_fixed_endpoints.csv)。

## Payload：保留成功案例，但不能把新失败父冒充已经修好

Payload原有效seed41027的+4mm、zero_noise在此前DEV55/57.5mm为96%/85%，同模式Metric90%/78%；nominal/execution均100%。该正面证据可用于继续推进，不必等42028/42029也最优。与此同时两个新EA父300K均零训练成功，42028恢复后207违规/2904timeout、42029为270/2929；最终评测也均0/30成功，不将其删除或算作安全进步。

已有+4mm训练阶段stall2对Metric0的代价仍须披露，部署成功率不能改写训练风险。现在停止追加新随机seed或盲目延长两个失败父，先检验已有有效模型的新布局表现，再决定是否需要新的训练修改。

## 下一项实际工作：先冻结、再一次性新布局同模式比较

本次选择已经参考历史DEV，是**回顾性的模型/seed选择**，不是事先随机抽到的代表性seed。选择依据、任务模型、SHA、模式、评测seed区间及回合数先写入固定协议并提交，之后才允许运行新评测。成功只支持这些已选模型在新布局上的表现，不能宣称整个训练方法跨seed稳定。

- Reach选seed41029固定550K的P5+2mm及同seed Metric；Payload选seed41027固定220K的P5+4mm及同seed Metric，候选/对照同父同训练预算。
- 四个checkpoint使用此前已有的真实SHA，未替换中间最佳节点。选择seed/model本身有DEV择优，明确披露；不伪称从未做过选择。
- 两任务均只用原target actor的zero_noise部署，Metric完全相同模式，不比较候选无噪声与基线随机。
- 每方法四个条件各200回合，每任务1600回合，共3200；Reach布局12310000起、Payload12320000起，动作seed各12311000/12321000起。已知历史计划中未使用这些新评测seed区间；不声称仅凭seed不同就能证明连续状态空间完全不重叠。
- 所有条件、回合、成功/违规/timeout/stall、return、capped时间和推理计时均保存。原horizon/安全距离/首次违规终止规则不变，失败按原horizon计入capped时间。
- 这批是事先冻结的、seed区间分离的新布局holdout，只能对被选模型作确认；若看到结果后继续改模型，这批须作为已用开发证据，另设后续独立集，不允许重用它再称“独立确认”。
- 不训练、没有optimizer更新、不调用实机；只允许这一次固定四模型评测，无按结果换模型/加回合/重复选最佳机制。

固定协议：[selected_seed_frozen_holdout_20261123.json](../../configs/paper_crax_payload/selected_seed_frozen_holdout_20261123.json)；[原生模型加载与同模式评测入口](../../scripts/paper_crax_payload/evaluate_selected_seed_holdout_20261123.py)。两项针对性测试在4080环境通过，确认同模式、同预算与冻结协议，不运行训练全套测试。

评测仅在4080的4T数据盘执行，两任务共享GPU0；保存前保留40GiB+2GiB紧凑输出余量，不占用其他项目。A100数据盘当前仅约22G，继续不写大存档、不回退home、不删除无关文件。新评测计划不意味着已开始或已完成，真实进程登记另行记录；每30分钟资源监测继续沿用只读机制。

## 当前完成口径

“每个seed最优”已不再是用户要求；后续报告将区分：成功案例的可行性证据、全seed描述性结果、新布局对所选模型的确认。Panda≤2pp回归、SafeReach训练期目标、Payload更高困难几何成功率等原任务指标不因本次口径调整自动通过。仍不能声称“所有任务、所有seed、所有几何、所有训练时刻全面最优”，也不因为选定成功seed就把整体研究标为完成。
