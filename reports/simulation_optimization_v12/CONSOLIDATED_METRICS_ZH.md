# 完整终点统一指标：18条，累计新增1.8M实际训练步

本表只汇总已有六批compact数据，不运行新环境、不产生训练步。18个完成100K端点，包括Reach8条、Piper Payload4条、fresh Panda SafeReach6条；最终DEV58个条件节点、4540回合。并非18个训练seed：Piper共同父seed41027，fresh SafeReach seed2033，跨seed稳定性未证明。

- `consolidated_endpoints_1412/training_endpoints.csv`：全部完整终点训练违规、安全成功、超时、stall、普通终端成功及累计return。
- `consolidated_endpoints_1412/development_endpoints.csv`：逐方案/条件safe与普通成功、违规、timeout、stall、return、horizon-capped回合时间及单独action耗时。
- `consolidated_endpoints_1412/receipt.json`：各输入SHA、实际纳入的完成数、排除的E4部分训练（不伪装100K终点）、缺失值与测量口径。

推理时间是每回合action调用p50/p95的均值，不是将所有action合并后的全局p50/p95，不包含物理仿真、不是实机推理。共享GPU负载与采集时段不同，不能拿跨批毫秒差直接证明算法快慢。early SafeReach原screen没有独立action计时，CSV留空而非填0；旧v10冻结CPU基准不补填到新实验。

统一表保留所有失败方案，不以最好中间节点替代完整100K。Reach task目标和训练风险尚未同时达标，Payload仍不足，Panda新参考与E3/E4未完成（Panda参考未纳入18条）。普通任务成功只按终止时实际目标距离判断，允许同时违规但没有违规后动作，不能等同于safe成功。

“违规率更低”仍必须同看单位实际步的训练违规、累计任务成功、超时和stall；不把几乎不完成任务的低违规当进展。完整学习曲线、先前逐节点比较与单seed限制继续有效。这份汇总只是分析入口，不是最终独立验收报告。
