# CVPO第三seed：已复用4080已释放槽，实际训练进行中

2026-11-15 01:49 UTC事前登记，02:20 UTC真实运行核对。六臂组件／参数包已完成，
4080本轮共享槽释放；A100两条原seed继续原进程，原第三seed等待控制器1866845已单独退役。
新原生worker3580633真实命令与manifest匹配，已观察73,728步／3584次官方更新；
随后取回的[首份状态](cvpo_4080_handoff/first_training_status.json)为237,568步／13,824次更新。
这是原同一第三seed的正常正式训练，不是新增seed或训练失败重试。

## 保持科学定义不变

本次将从未启动的seed51029移到4080一条共享槽，而不是增加第四个seed。
仍为10,000,128实际步、相同256向量环境、batch／更新量、全部五节点与验证布局。
原科学源码`7abb6ebb8e139992fc88a9bcbd2fcebc8e55227e`、FSRL／CRAX固定源码、
原完整`cvpo_formal_v4.json`均保持原字节，不修改数据路径字段绕过注册检查。
Torch allocator仍为原配置2.5%、JAX不预占；没有暂停其他项目或额外科学评测。

## 仅在4080项目数据盘准备环境

新文件全部在`/home/carla/4T/SAFEFLOW2027_NONHARDWARE_V4/e08-fsrl/`。
A100现有venv依赖、overlay、正式训练源码、官方FSRL和CRAX直接压缩流式复制至该目录；
没有下载到本地主机，不改A100环境或其活跃任务，不替换4080已有环境。
4080既有同版本Torch2.7.1+cu118仅只读复用；Python为3.11.7，A100为3.11.5，
不同硬件与Python补丁版本披露，不宣称逐位相同训练或受控纯算法速度比较。

`cvpo_4080_namespace.sh`使用已可用的非特权bwrap目录映射：
4T项目根在新进程挂载空间内映射为原`/storage_0/.../nonhardware_v4`。
原注册路径因此仍正确，而物理写入4T盘；宿主机不创建`/storage_0`、不新增全局软链接、
不改系统配置。系统／现有环境只读，GPU仅使用原共享单卡。
依赖压缩复制正常完成，原路径可见、Torch版本一致；一次实际微型Torch＋JAX GPU运算通过，
两者sum均为8.0，JAX设备为cuda:0；零原生环境控制／训练更新。
导入出现非致命TensorBoard／TensorFlow protobuf警告，未把警告当作训练结果或隐藏；
实际CUDA运算及原生训练均通过。TensorFlow仅可能作为日志依赖，进程设置显存按需增长，
不改全局依赖或增加额外训练检查。

## 正式接续前置条件（现均满足）

1. 原依赖压缩传输正常结束，只做一次微型Torch＋JAX GPU导入／运算检查，不运行RL或原生rollout。
2. 核对A100第三seed输出仍不存在、1866845只等待且无活跃训练子进程。
3. 只退役该等待控制器并保存原状态，绝不向两条训练worker发信号；原计划标记被新位置取代。
4. 用原run_queue及登记的`cvpo_queue_4080_v4e_seed51029.json`接续，核对真实原生PID／命令及首次学习进度。

实际只给等待调度器1866845发SIGTERM；原生两训练worker在交接前后的命令SHA与内核启动标识
完全相同，未被中断。原等待状态与计划在A100归档，原计划标记superseded以防重复接纳。
新bwrap3580629、内层队列3580631、原生训练3580633，使用真实host447／CUDA_VISIBLE_DEVICES=0。
完整[交接收据](cvpo_4080_handoff/handoff_receipt.json)、
[原manifest](cvpo_4080_handoff/native_run_config.json)和[队列状态](cvpo_4080_handoff/native_queue_state.json)已保存。
大型venv复制最初为无压缩流，随后只结束这两条本地复制SSH并改用压缩；
这不是失败训练重试，没有给任何训练进程发信号，也没有删除任何实验数据。

## 完成后仍由原A100统一闭合

4080完整收据齐备后，将原manifest、receipt、progress、五个validation及五个policy文件
直接流式传至A100原run-root的临时接纳目录，完成后原子重命名，再让现有闭合队列接纳。
policy SHA由原CVPO campaign核对；不先打开独立测试、不改选定规则或重新训练。
大型full／replay检查点留4080的4T盘；代码／紧凑结果上传GitHub并镜像本地。
科学程序／环境桥接／注册配置的SHA均与A100原文件一致，见交接收据；没有修改原科学源码。
原A100独立闭合控制器继续等待，不能把三条均启动写成已完成30M或独立测试。
新失败仍不自动重启；若发生异常，先保留原始前缀、检查完整恢复边界，不追加完整重训。
