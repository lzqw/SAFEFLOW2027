"""Summarize a complete E01 final test; read-only inputs, no policy loading."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def wilson(count, total):
    z = 1.959963984540054
    p = count / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0., center-radius), min(1., center+radius)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous result reports')
    receipt = json.loads((args.input/'receipt.json').read_text())
    freeze = json.loads((args.input/'freeze.json').read_text())
    if (receipt['status'] != 'completed_independent_test' or receipt['independent_episodes'] != 1600 or
            receipt['independent_arms'] != 4 or receipt['final_test_reselection'] is not False or
            receipt['new_training_transitions'] != 0):
        raise ValueError('Require genuine final completion, not a prefix')
    with (args.input/'independent_summary.csv').open(newline='') as stream:
        summaries = list(csv.DictReader(stream))
    with (args.input/'independent_episodes.csv').open(newline='') as stream:
        episodes = list(csv.DictReader(stream))
    conditions = freeze['conditions']
    labels = [arm['label'] for arm in freeze['frozen_arms']]
    expected = {(label, condition, seed) for label in labels for condition in conditions
                for seed in range(91027, 91127)}
    observed = [(r['label'], r['condition'], int(r['layout_seed'])) for r in episodes]
    if len(summaries) != 16 or len(episodes) != 1600 or set(observed) != expected or len(set(observed)) != 1600:
        raise ValueError('Require every frozen arm/condition/layout exactly once')
    for row in summaries:
        part = [r for r in episodes if r['label'] == row['label'] and r['condition'] == row['condition']]
        successes = sum(r['safe_success'] == 'True' for r in part)
        violations = sum(r['violation'] == 'True' for r in part)
        if (int(row['episodes']) != 100 or int(row['safe_success_count']) != successes or
                int(row['violation_count']) != violations):
            raise ValueError('Native summary and all episode counts must agree')
        low, high = wilson(violations, 100)
        row.update(violation_wilson95_low=low, violation_wilson95_high=high)
    args.output.mkdir(parents=True)
    for name in ['receipt.json', 'freeze.json', 'independent_episodes.csv']:
        (args.output/name).write_bytes((args.input/name).read_bytes())
    with (args.output/'independent_summary.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summaries)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(args.input.glob('*')) if p.suffix in ('.json', '.csv')}
    (args.output/'report_receipt.json').write_text(json.dumps(dict(status='completed_report',
        source_sha256=sources, retained_episodes=len(episodes), retained_conditions=len(summaries),
        new_rollouts=0, new_training_transitions=0, candidate_reselection=False,
        source_raw_files_are_retained_on_data_disk_and_local_delivery=True), indent=2)+'\n')
    lines = ['# Panda Payload E01：最终独立测试', '',
        '四条原生分支各新增100K、20候选共同验证及验证后审议完成后，冻结候选再一次性测试。',
        '每条件100个新布局91027–91126，共1600回合；未依据测试换候选或继续训练。',
        '训练seed41028；父年龄EA240K、Iso120K、SAC300K不同，仅新增适应预算相同。', '',
        '| 冻结分支 | 新增候选年龄 | 条件 | 安全成功 | 违规 | 失败封顶平均用时 |',
        '|---|---:|---|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['label']} | {int(row['new_checkpoint_steps'])//1000}K | {row['condition']} | "
            f"{row['safe_success_count']}/100 | {row['violation_count']}/100 | "
            f"{float(row['time_to_goal_capped_seconds']):.4f}秒 |")
    lines += ['', '失败、违规或超时均计完整原生时域，不只统计成功回合。',
        '零观测违规不是未来零风险：0/100的Wilson95%上界约3.70%。',
        '这是共同验证选出的好候选，不是跨训练seed均值、训练稳定性或实机保证。',
        '保留全部原生物理子步与碰撞检查；沿用原生逐条件动作随机流，',
        '不声称在回合长度不同时仍逐布局逐动作噪声配对。70mm属于修复训练范围，',
        '不叫未见半径外推；旧偏移条件不在本次最终测试，未证明合法替代路线。', '',
        '[全部条件与风险区间](independent_summary.csv) · [全部1600回合](independent_episodes.csv)',
        '· [冻结身份与测试前审议](freeze.json) · [原生完成收据](receipt.json)',
        '· [来源哈希](report_receipt.json) · [测试前阶段决策](../E01_STAGE1_REVIEW_ZH.md)', '',
        '完整恢复检查点留4080的4T数据盘；本报告仅整理已保存数据，没有新仿真／RL／实体动作。', '']
    (args.output/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
