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

## 04:43 UTC：已按完整状态实际续跑

04:32原worker1864692在4,401,408步、274064次官方更新边界完整保存并退出，
状态为`engineering_stop_at_complete_collection_boundary`，evaluated仍为2M／4M。
另一worker1866873的完整命令SHA及内核starttime前后相同。
原A100 Python构建未暴露`os.pidfd_open`（两次调用在任何信号／操作目录建立之前失败），
因此使用同一Linux pidfd系统调用绑定原PID，再只发一次SIGTERM；没有训练失败重试。
全部[事前进程记录](cvpo_resource_resume_51027/before.json)及
[完整边界](cvpo_resource_resume_51027/boundary.json)保留。

原完整断点与必要小型记录在04:40之前通过压缩管道直接传到4080的
`e08-fsrl/resume-prefix-51027-v1`，原文件仍保留A100。
源／目标interrupted_full的SHA均为
`69502891c4d9eff39118450a9c136e1b9197c03ca77c53524bd7783b3a95436e`。
科学train_cvpo.py与注册configuration SHA仍为原
`4ba046e2fde2c80209f9d2e52df36623dbc41857c99ff8cb6355c3cbdaf89742`／
`619d7920a5cb40b327bd506303e9649e0ceaa8f64c2d6464681577f5d6ae40b0`。

04:40:23 UTC用原namespace入口及`--resume`启动新output
`runs/cvpo-51027-resume4080-v1`，wrapper3643537、原生worker3643539，
启动时GPU空闲11690MiB、4T盘余量超过56GiB，所有输出均在4T盘。
[原始启动记录](cvpo_resource_resume_51027/launch.json)保留命令、时间和原预算。
04:43真实cmdline匹配，已推进到4,516,096步／281232次更新，
evaluation_controls保持500000、evaluated仍2M／4M，证明原节点没有重跑。
[新manifest](cvpo_resource_resume_51027/resume_run_config.json)及
[首份实际学习心跳](cvpo_resource_resume_51027/first_learning_status.json)已收回，
不是只启动一个空shell或从头随机训练。

仅需新增5,598,720步，累计目标仍10,000,128；没有新增seed或隐藏旧累计成本。
`close_cvpo_resource_resume.py`已订阅原生worker3643539的退出事件：仅完整完成后，
验证prefix／suffix节点互斥、原policy SHA、累计623984官方更新和1250000评测控制，
保存两份原manifest／日志并拼接完整时间顺序记录。
再通过小型压缩管道传A100，将原prefix完整归档后原子接纳canonical完整结果。
原A100闭合控制器不变；它仍须等全部三seed完整收据。

两项新CPU契约检查通过（0.02秒）：完整后缀合并保持5,598,720新增步及原始记录，
拒绝重复节点；原子接纳保留完整原prefix。零模型执行／原生rollout。
真实后缀完成及最终接纳仍待观察，不把这些工程检查或首次心跳冒充完整10M。

## 05:35 UTC：同协议接续最后一条的事前登记

05:30原seed51027续跑已至9,372,160步，6M／8M节点完整；原seed51028在A100为
5,286,144步、329360更新，仍处于原进程1866873。
为缩短总体墙钟，最后seed51028也使用完全相同的完整边界／原注册`--resume`协议，
先保存原边界并直传断点，只有原4080 seed51027真正完成、空闲显存满足门槛后才启动。
不并发新增第三个learner，不改变五节点或500独立闭合；旧2M／4M不重跑。

新prefix物理路径为4080`e08-fsrl/resume-prefix-51028-v1`，
新suffix为`e08-fsrl/runs/cvpo-51028-resume4080-v1`；原A100prefix保留。
`close_cvpo_resource_resume.py`仅增加seed／完整prefix步数及update数参数，
51027原等待的默认身份不变；51028只能显式登记其真实边界计数再接纳。
只有两个合法训练seed51027／51028、4M之后6M之前的完整prefix可用，
其余原注册科学设置和全部模型／环境状态仍交给原训练器严格校验。
对应路径、归档和新增步数按seed派生，不复用或覆盖另一条输出。
两项原CPU契约检查在参数化后再次通过，零新增模型执行／rollout。
此段是事前登记，不代表51028已经迁移或完成；须保留随后的真实边界与新PID。

05:38 UTC：51027原worker3643539已自然结束，真实10M收据和五个节点通过互斥合并／SHA核对，
完整包已原子接纳A100；旧prefix全部保存在`runs/cvpo-51027-prefix-a100-v1`，没有数据删除。
[全部结果及原生连续性收据](e08/completed_seed51027/README.md)已保存。
51028原worker1866873也已在5,363,968步／334224官方更新完整边界退出，
evaluated仍2M／4M，剩余4,636,160步；[边界证据](cvpo_resource_resume_51028/boundary.json)保留。
原控制器1866846仅在其worker已经退出、子进程全部为已退出Z状态后退役，避免旧队列误报或重复接纳。
原闭合控制器1866848命令不变；最后断点传输进行中，不提前打开独立测试。

05:46:46 UTC：最后51028的完整断点也已直传且SHA一致：
`07c0bc57e4c4cf75f535048125028505d97206f38e7e539c3241bdce63dbbaae`。
确认51027原生worker已退出且completed收据存在、GPU空闲15557MiB、磁盘大于56GiB后，
用原入口／配置／科学源码启动51028原状态后缀，wrapper3672459、原生worker3672461。
[启动记录](cvpo_resource_resume_51028/launch.json)／
[新manifest](cvpo_resource_resume_51028/resume_run_config.json)／
[首份学习心跳](cvpo_resource_resume_51028/first_learning_status.json)均已保存。
05:48命令匹配，已推进5,449,984步／339600更新，evaluated仍2M／4M、evaluation_controls500000，
表明没有从头重跑或重复旧评测。只需后缀4,636,160步，累计仍原10M。
新退出事件订阅已绑定原生worker3672461，完整后用显式seed51028／prefix步数与updates接纳，
原A100闭合控制器继续等最后一份完整收据；不在本阶段挑选策略／打开独立测试。
