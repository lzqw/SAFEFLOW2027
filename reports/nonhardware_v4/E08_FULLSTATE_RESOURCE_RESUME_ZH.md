# CVPO资源续跑：事前登记，不改变训练预算或选择规则

2026-11-15 04:31 UTC登记。4080第三seed完整10M及接纳A100已完成。
A100原seed51027／51028于04:30为4,368,640／4,454,656步，累计约206／215SPS；
4080第三seed完整1881.678SPS，当前约40%GPU／3879MiB占用，有原共享槽可用。
共享负载不同，速度不是硬件或算法的受控比较，但足以支持资源调度。

## 此次范围

只迁移原A100 seed51027的未完成后缀；原seed51028继续A100，原闭合控制器仍等待。
这是原三seed30M预算内部的资源续跑，不是额外训练、科学早停、换seed或失败重试。
采用原科学源码`7abb6ebb8e139992fc88a9bcbd2fcebc8e55227e`的`--resume`，
configuration字节、环境、batch、更新量和五节点验证／500独立测试规则不变。
不同GPU/Python补丁版本不承诺跨硬件逐位复现，诚实披露资源变化。

## 安全边界与保留规则

1. 用原PID1864692、完整cmdline和内核starttime核对目标，确认没有completed收据。
2. 只向该训练worker发送一次SIGTERM。原处理器完成当前采集／官方post_update及可能的节点验证后，
   保存`interrupted_full.pt`，记录`engineering_stop_at_complete_collection_boundary`再自然退出。
   不使用SIGKILL、不碰其他项目或另一原worker；无完整边界则不启动续跑。
3. 原prefix目录保留。将interrupted完整断点（含模型、全部优化器／dual、replay、normalizer、
   原环境／key、未完回合、RNG、累计步数／成本／updates／evaluated节点）通过管道直传4080数据盘。
   源目标SHA一致才调用原训练入口，使用新output，不覆盖prefix；没有本地模型文件。
4. 已评测2M／4M节点原样保留，不重新评测；原训练器从断点evaluated列表继续剩余节点。
   实际新增transitions只等于10,000,128减prefix已消耗值，累计训练仍为原固定10M。
5. 原4080namespace wrapper复用注册逻辑路径，物理新输出放4T盘，原环境／源码只读不改。
   启动前检查至少5GiB空闲显存、56GiB数据盘余量；继续共享，不停止外部项目。
6. 续跑完整后，将原prefix与新suffix的validation／policy按互斥节点合并为完整五节点包，
   prefix／suffix原manifest及日志均保存，累计计数对齐；不能用suffix假称从头10M。
   A100原prefix先归档到同盘明确位置，再原子接纳新完整canonical目录，不能删除旧数据。
7. 仅原三seed全预算齐备后由原闭合控制器执行原冻结规则／500独立回合。
   如果续跑失败，保留全部历史、报告工程故障，不自动重试或额外从头训练。

当前仅事前登记；实际停止、断点步数、SHA、续跑worker及完成结果须以随后原生证据为准。
