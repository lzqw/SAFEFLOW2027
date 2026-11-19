# 固定400K半径覆盖的有限后处理：保留双seed、只计新增100K

新增只读closure从独立源码目录观察原`queue_independent_radius.json`，每1800秒检查、最多24次；要求源码`10ba914c`原两条真实native进程退出且队列completed，才做一次汇总。观察超时或失败不会重新训练、改变原训练状态或启动下游。

汇总严格要求两个seed的原生initial300K/new100K/total400K完成receipt、完整301K–400K共100个1K日志、原`training_receipt_400000.json`与唯一400K DEV（三条件各30完整回合）。按每seed真实300K父receipt扣除继承计数，逐episode成功/违规/超时相加与episodes一致；不把600K父阶段或400K累计算成本轮200K新增。不换父SHA/seed，不选择更好checkpoint，工程smoke/缺失节点/错误累计数拒绝为完整结果。

输出全量已发生曲线、两个新增窗口/累计指标、180个DEV回合、等权seed成功均值/SD/最好/最差、真实400K checkpoint SHA与大小。检查点只在数据盘流式哈希，不加载或生成动作，零训练/零仿真/零optimizer更新。55mm明确标为trained_radius，不称外推，不隐去400K任务失败或用平均值覆盖较差seed。

后处理source与operation/output都在A100用户data root，由storage wrapper启动，保持40GiB+1GiB余量。它与活跃训练源码`source-independent-radius/`分开，主训练进程和7项既有原生测试结果不改变；新计数/terminal-handle相关11项轻量测试通过，不新增物理工程smoke。

输出预定`independent-radius-complete-analysis/`，观察记录`independent-radius-closure/`。后续P5/+2mm/Metric各150K仍需用两条真实400K父SHA另行固定，**没有自动启动900K方法确认，也没有开启新独立集或宣称全面最优**。GitHub仅上传程序与协议，完整结果在实际产生后再发布；本文件不是完成收据。
