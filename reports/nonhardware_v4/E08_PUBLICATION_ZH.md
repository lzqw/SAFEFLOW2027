# E08结果交付入口：四方法主表与学习曲线

此页登记结果完成后的交付方式；CVPO已复用两个释放槽并行训练，尚无正式30M完整结果。
原首次库初始化失败的修复与后续排队安排见[实际接续记录](E08_RELEASED_SLOT_HANDOFF_ZH.md)。
报告工具不部署到正在运行的训练源码，不改变CVPO配置、模型选择或独立测试。

## 只同步紧凑数据

原A100数据盘`nonhardware_v4/e08-fsrl/`中的三条run各同步：
`run_config.json`、`receipt.json`、`progress.jsonl`、五个`validation_<实际步数>.json`。
报告直接读取原生验证JSON中的全部回合与指标，不需要下载policy、full检查点或replay。
最终闭合目录`final-closure-v1/`同步：`receipt.json`、`selection.json`、`fixed_10m.json`、
`independent.json`和五个`independent_<评测seed>.json`。所有完整模型仍留A100数据盘。

`report_cvpo_campaign.py`读取以上小文件，并复用原`reports/robot_gap_v2/paper/`的
三方法完整CSV，不修改或重新生成历史候选身份。它在建立任何输出目录前要求：

- 三训练seed51027/51028/51029各真实10,000,128完成收据，原注册配置完全一致。
- 全15个共同验证候选、五个实际快照、每候选五seed×50完整回合。
- 每个validation文件SHA及policy SHA与测试前冻结记录一致。
- 测试前freeze digest、选定身份与预定选择规则相符；报告只核对原选择，不更新它。
- 独立五seed×100共500完整回合，与冻结身份绑定；回合原生指标和汇总值一致。
- 原三方法45个学习节点、九个固定终点和三个选中策略完整保留。

真实大policy字节的SHA核对已由原服务器`cvpo_campaign`在独立评测前执行；
本地不下载模型，因此这里核对的是已记录SHA／文件链和完整原生回合，
不把小文件一致性称为又一次独立模型执行验证。

## 完成后运行一次，不覆盖已有报告

```bash
python -m scripts.paper_crax_payload.report_cvpo_campaign \
  --runs /local/compact/e08-fsrl/runs \
  --closure /local/compact/e08-fsrl/final-closure-v1 \
  --historical reports/robot_gap_v2/paper \
  --output /local/new-cvpo-four-method-report
```

输出固定10M逐seed和均值±样本SD、全部学习节点、四方法完整学习曲线和
选中单策略独立结果，各表分开，不以选择后分数替换固定终点。
同时记录训练累计成本、完整训练墙钟与SPS、输入文件SHA和零新增训练／仿真收据。
`cvpo_training_efficiency.csv`单列每seed官方policy更新数、原生评测控制数和实际新增步数；
更新单位与旧SAC／Flow不同，不将其冒充等价optimizer机会。
共享GPU、后端和更新组织不同均披露，不能把实际墙钟差当纯算法加速比。

一个合成全流程CPU测试通过，覆盖60个总学习节点、12个固定终点、四个独立策略、
图表输出和独立回合分母缺失拒绝；未运行额外训练或原生仿真。
合成数据不随科学结果目录发布，不充当正式CVPO得分。
真实结果尚未齐备，当前不生成空白或预测分数表、不宣称已完成四方法主比较。
