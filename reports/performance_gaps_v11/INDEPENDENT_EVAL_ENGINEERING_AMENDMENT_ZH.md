# 独立评测工程修复记录

首个 Reach 独立评测启动后只写入 registration，没有执行任何回合。原因是评估器读取了旧 checkpoint 顶层的 `model_source_urdf_sha256` / `model_commit` 字段；原生 v11 100K checkpoint 的 `configuration` 中保留model commit，但未重复保存URDF SHA。第一次修复后仍因SHA缺失在首回合前拒绝进入评测。两次失败均保留注册和消费标记；未改checkpoint、训练、最终选模或任务定义。

修复仅将评估器的身份读取改为“兼容旧顶层字段，否则读取 checkpoint configuration”；对历史 checkpoint 未保存的 URDF SHA 字段，仅在字段存在时比较，官方资产文件仍独立执行 SHA 校验，model commit 必须一致。Piper Payload 覆盖候选的训练计划校验不变。原始失败 registration 与消费标记保留在旧 freeze 目录，不删除、不覆盖；每次修复都使用新的 amended freeze，并明确保留此前结果。若 amended 评测再次发生工程错误，不自动重试。

排查期间曾意外在第一个预留独立seed上reset并执行一次online Actor控制（非完整回合，无结果选择），计为1次额外仿真控制和20个物理子步；未据此改变模型、训练配置或最终选择。完整独立评测仍在已冻结选择下只运行一次；报告需披露这项协议偏离。原始v2 freeze引用的是本修复说明当时版本，原文及SHA保留在服务器v2源码快照与Git历史，不篡改已消费freeze。
