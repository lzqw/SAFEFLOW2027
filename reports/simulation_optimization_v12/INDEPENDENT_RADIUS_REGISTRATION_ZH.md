# 独立谱系第二阶段：两个固定300K父各相同100K有限半径覆盖

独立父600K已完整结束，两个固定300K实际SHA见`simulation_v12_reach_independent_radius.json`。本阶段采用原已规划谱系的下一部分：seed41028、41029各增加100K环境transition，到固定400K，总新预算200K。两条父55mm结果较差（6/30、15/30）都保留，不删除41028、不按较好父或较好checkpoint选模。

只把训练sphere radius扩成原生45/50/55mm deterministic round-robin family；原任务/奖励、安全15mm、首次违规终止、horizon、2env/batch256/UTD1、EA actor/critics/EMA、optimizer/alpha/dual/replay与计数均沿用严格native geometry-adaptation。配置与各自真实300K父只差`training_radius_values`一个字段。未引入已失败的loss taper/终点残差、P5方向权重或新的批处理实现，不更改旧loader。

同原生distribution fork，环境开始新episode，native runner按原seed重新初始化随机流；**不是完整父RNG精确resume**。父实际保存只暴露一个GPU，其CUDA RNG index是0；登记不照抄旧41027双GPU的index1，且新400K也只保存本slot index0。父网络/optimizer/replay保留，不把重复原生计数当新收集；native receipt明确initial300K/new100K/total400K。

该native分支原本只在400K评测：继续相同DEV随机流11990000/11991000、三个条件各30回合，不修改原有训练函数去加选模节点。55mm此阶段已在训练覆盖，400K的55mm不能再称几何外推；真正57.5/60mm仍不进入训练、后续协议另行固定。此阶段不是P5/Metric三臂结果，不把两个父的成功率改善当最终方法通过；不使用独立测试集。

固定每条完整100K，不科学提前停止、自动重试或继续增加预算；仅原生NaN/OOM等工程保护。所有seed/所有失败/固定400K端点均保留，不选更好节点。完成后记录真实400K SHA供随后单独登记的固定P5/+2mm/Metric三臂150K确认使用；该200K队列**不自动启动后续900K**，也不意味着最终配置已冻结。

资源：A100两个GPU目前都无本实验/外部计算负载，各分一个共享小allocator .025 slot，不停止外部任务。新来源、resolved配置、输出/checkpoint/log/cache/TMP均在用户data root，并由storage wrapper启动。启动前剩余约51GiB，两个新目录保守总增长6GiB（原两条300K共938MiB），准入≥46GiB，保持40GiB底线；不新增删除、不复制巨大父到home或本地、不用5080或实体机器人。

实现通过最少必要针对性测试：配置只变radius、保留两个seed及RNG0；错误SHA/seed/240K替代/参数改写/RNG映射/收据计数均拒绝。此原生distribution-fork路径已用于历史100K训练，无额外大规模重复smoke或科学校验。正式启动另记录实际PID和首个新采集步数，不能把当前准备当正在训练。
