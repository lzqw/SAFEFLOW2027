# 三臂入口已实现并做针对性测试，仍未启动900K

当前两个独立原生EA半径覆盖任务仍使用`10ba914c`原源码运行；新入口放在独立源码，不修改活跃进程、配置或预算。400K父尚未完成，故没有编译真实resolved计划、没有创建执行登记/队列、没有追加900K或开启独立测试。

## 原生算法复用与公平评测

`train_reach_crossseed_v12.py`复用现有`ComparedPiperLearner`与严格native full-parent driver，包含原P5、固定+2mm、Metric各150K；新增子类只保存seed/真实父SHA/模板/预算/部署身份，没有新网络、梯度规则、critic或optimizer。原P5/Metric采样、能量、权重、方向损失和原60–80K LR下降规则不改；100K以后所有臂保持相同floor，旧100K节点也保留。

父加载继续要求完整原生34D、匹配EA方法/seed/父SHA/400K步数/环境数；CUDA父RNG index0，保留完整网络/optimizer/replay/dual，按原分支相同处理部分episode。新格式不允许旧100K或其他seed/candidate检查点冒充本确认。

在100K及150K的完整DEV中，每个臂均执行sampled和zero_noise；其余四个早期节点仅sampled。四个条件原名保留，以`condition + deployment_mode`共同标识（不得丢掉一个模式去重）。所有额外回合和物理控制均进入原驱动的evaluation_controls计数；两个模式均隔离RNG、禁止optimizer更新，评测前后actor/target参数digest、计数与dual必须不变。零噪声是不同部署策略，不改写原随机结果或训练风险。

## 不能由准备文件直接开跑

入口要求单独的六任务执行登记：绑定runner源提交/文件SHA、每seed resolved plan文件SHA、每任务host/output/visible GPU、小allocator .025、两个seed各完整三臂/同host、不早停、不自动重试或独立test。模板中的`preparation_only`没有被悄悄改成已执行；独立执行登记只可在真实双父完成及资源到位后建立。父编译证据随resolved plan保存并反向重建exact模板，防止编译后私下调系数、预算或父。

原配对driver在**每次checkpoint前要求40GiB+10GiB，即50GiB**，所以不能只看最初的40GiB底线。新准入还需为该host每条完整分支预留2GiB增长：三臂至少56GiB，六臂至少62GiB。A100当前约51GiB，不足以新增三臂；现原生radius driver及其6GiB登记余量不变。后续优先考虑4080大数据盘或在不删除证据/不停止外部任务前提下单独规划迁移，不能开跑后因检查点空间不足中断。这里没有进行清理、复制巨大父或修改原native存储检查。

## 最少针对性验证

- 本地7项纯配置测试通过：严格双父、预算、RNG0、方法固定、编译后调参拒绝。
- A100原环境4项无物理测试通过：三方法分别复用精确采样/能量/损失/原optimizer/RNG，100K/150K LR不改；跨seed身份严格恢复；双模式顺序、状态保持与全部额外控制计数。
- 新资源登记逻辑另2项小测试通过：错output、多visible GPU、同seed混host拒绝，51GiB不能冒称可容纳三臂checkpoint增长。未重跑整套测试、未增加仿真smoke或训练。

当前只完成软件准备；端到端物理训练、真实父绑定、空间准入、后续完整汇总与最终目标均仍待实际执行，不能用6项软件测试当科学成功率证据。相关程序/报告上传GitHub并本地归档，模型/凭据不上GitHub，无PR/评论/邮件或实体机器人连接。
