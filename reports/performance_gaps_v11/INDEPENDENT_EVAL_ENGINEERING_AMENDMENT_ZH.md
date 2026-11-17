# 独立评测工程修复记录

首个 Reach 独立评测启动后只写入 registration，没有执行任何回合。原因是评估器读取了旧 checkpoint 顶层的 `model_source_urdf_sha256` / `model_commit` 字段；原生 v11 100K checkpoint 将这两个固定身份保存在 `configuration` 中。该错误发生在第一条 Piper Reach 案例进入物理回合之前，未产生独立结果、未改 checkpoint、未改训练或任务定义。

修复仅将评估器的身份读取改为“兼容旧顶层字段，否则读取 checkpoint configuration”；对历史 checkpoint 未保存的 URDF SHA 字段，仅在字段存在时比较，官方资产文件仍独立执行 SHA 校验，model commit 必须一致。Piper Payload 覆盖候选的训练计划校验不变。原始失败 registration 与消费标记保留在旧 freeze 目录，不删除、不覆盖；每次修复都使用新的 amended freeze，并明确保留此前结果。若 amended 评测再次发生工程错误，不自动重试。
