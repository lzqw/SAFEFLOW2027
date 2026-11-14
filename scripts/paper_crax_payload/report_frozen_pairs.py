"""Posthoc paired-layout description of validation-frozen candidates; no rollout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CONDITIONS = ('nominal', 'radius_055', 'execution_scale_085')
METRICS = ('time_to_goal_capped_seconds', 'path_length_m', 'minimum_distance_m')


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_rows(rows, selection):
    scales = {}
    for method in ('Iso', 'EA'):
        choices = selection['selected'][method].values()
        agreed = {float(c['action_scale']) for c in choices}
        if len(agreed) != 1:
            raise ValueError('Risk-cap selections differ: do not silently pick one after test')
        scales[method] = agreed.pop()
    groups = {}
    for row in rows:
        method = row['method']
        if (row['stage'] != 'E02' or row['split'] != 'independent_test' or
                method not in scales or float(row['action_scale']) != scales[method]):
            continue
        if row['static_admissible'] != 'True':
            raise ValueError('Do not silently exclude inadmissible selected-policy layouts')
        key = (row['condition'], int(row['layout_seed']), method)
        if key in groups:
            raise ValueError('Duplicate pair identity')
        groups[key] = row
    expected = {(c, s, m) for c in CONDITIONS for s in range(91027, 91127)
                for m in ('Iso', 'EA')}
    if set(groups) != expected:
        raise ValueError('Require all prespecified conditions and exactly 100 common layouts each')
    pairs = []
    for condition in CONDITIONS:
        for seed in range(91027, 91127):
            row = dict(condition=condition, layout_seed=seed)
            for method in ('Iso', 'EA'):
                source = groups[condition, seed, method]
                for metric in METRICS:
                    row[f'{method}_{metric}'] = float(source[metric])
                for metric in ('safe_success', 'violation'):
                    row[f'{method}_{metric}'] = source[metric] == 'True'
            pairs.append(row)
    return pairs, scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve published analyses')
    source = args.input / 'episodes.csv'
    selection_path = args.input / 'selection.json'
    receipt = json.loads((args.input / 'receipt.json').read_text())
    selection = json.loads(selection_path.read_text())
    if receipt['status'] != 'completed' or not selection['no_final_test_reselection']:
        raise ValueError('Require completed evaluation and unchanged validation selection')
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != receipt['attempted_episodes']:
        raise ValueError('Require the full completed episode source')
    pairs, scales = pair_rows(rows, selection)
    rng = np.random.default_rng(20261114)
    summaries = []
    for condition in CONDITIONS:
        subset = [r for r in pairs if r['condition'] == condition]
        indices = rng.integers(0, len(subset), size=(10000, len(subset)))
        for metric in METRICS:
            iso = np.array([r[f'Iso_{metric}'] for r in subset])
            ea = np.array([r[f'EA_{metric}'] for r in subset])
            delta = ea - iso
            low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
            summaries.append(dict(condition=condition, metric=metric, paired_layouts=len(subset),
                Iso_mean=float(iso.mean()), EA_mean=float(ea.mean()),
                EA_minus_Iso_mean=float(delta.mean()),
                paired_percentile95_low=float(low), paired_percentile95_high=float(high),
                EA_lower_layouts=int((delta < -1e-12).sum()),
                equal_layouts=int((abs(delta) <= 1e-12).sum()),
                EA_higher_layouts=int((delta > 1e-12).sum())))
    args.output.mkdir(parents=True)
    write_csv(args.output / 'pairs.csv', pairs)
    write_csv(args.output / 'paired_summary.csv', summaries)
    manifest = dict(analysis='posthoc_descriptive_common_layout_bootstrap',
        source_episodes_sha256_lf_normalized=hashlib.sha256(source.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        selected_scales=scales, rng_seed=20261114, bootstrap_resamples=10000,
        resampling_unit='layout within condition, keeping the two policies paired',
        interval='percentile95; unadjusted descriptive intervals, not preregistered hypothesis tests',
        training_seed_count=1, paired_layouts=len(pairs), new_rollouts=0,
        new_training_transitions=0, policy_reselection=False,
        risk_equivalence_test=False)
    (args.output / 'receipt.json').write_text(json.dumps(manifest, indent=2) + '\n')
    lines = ['# 冻结候选：相同布局的事后描述分析', '',
        '仅复用已完成独立测试，使用验证选定的尺度，不根据测试重新选模型。',
        '每条件100个共同布局，10000次布局配对bootstrap，固定分析随机种子20261114。',
        '下表差值为EA减Iso，单位秒；包含失败完整4秒封顶，不只计算成功回合。', '',
        '| 条件 | 平均用时差 | 配对百分位95%区间 | EA更快布局 |',
        '|---|---:|---:|---:|']
    for row in summaries:
        if row['metric'] == 'time_to_goal_capped_seconds':
            lines.append(f"| {row['condition']} | {row['EA_minus_Iso_mean']:.4f} | "
                f"[{row['paired_percentile95_low']:.4f}, {row['paired_percentile95_high']:.4f}] | "
                f"{row['EA_lower_layouts']}/100 |")
    lines += ['', '这是看到测试后增加的描述分析，三个条件均公开，不是预登记显著性检验；',
        '区间只描述这对冻结策略在布局间的变化，没有包含训练seed不确定性，未作多重比较校正。',
        '不对零违规数据做退化bootstrap，不把0/100当真实零风险或风险等价；',
        '风险及其Wilson区间见[完整原结果](../frozen_final/INTERPRETATION_ZH.md)。',
        '几何条件EA风险更高，时间差不构成同风险优势。路径与间距差异完整保留CSV；',
        '更短路径不一定更安全，更小间距也不是优势。没有新增仿真、RL或实体交互。', '']
    (args.output / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
