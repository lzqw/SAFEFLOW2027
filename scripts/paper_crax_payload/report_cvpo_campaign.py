"""Append completed CVPO evidence to historical PointGoal tables, never resample.

Reads compact native records only; no model/simulator imports or model selection.
Final publication requires all three complete10M runs and the frozen500-episode test.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


SEEDS = [51027, 51028, 51029]
MILESTONES = [2000128, 4000000, 6000128, 8000000, 10000128]
METRICS = ('mean_native_return', 'mean_episode_cost', 'mean_goals_per_episode',
           'mean_budget_compliant_fraction')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def near(left, right):
    if not math.isfinite(float(left)) or not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Published aggregate does not match underlying native records')


def evaluation_means(results, seeds, count):
    if [r['seed'] for r in results] != seeds:
        raise ValueError('Wrong evaluation seeds or order')
    metric_records = dict(mean_native_return='native_return', mean_episode_cost='episode_cost',
                          mean_goals_per_episode='goals')
    for result in results:
        records = result['records']
        if result['episodes'] != count or len(records) != count:
            raise ValueError('Incomplete evaluation denominator')
        for metric, key in metric_records.items():
            near(result[metric], mean(r[key] for r in records))
        near(result['budget_compliant_fraction'], mean(r['episode_cost'] <= 25 for r in records))
    return {metric: mean(r[metric.replace('mean_budget_compliant_fraction', 'budget_compliant_fraction')]
                         for r in results) for metric in METRICS}


def prepare(runs, closure, historical):
    inputs = []

    def read(path):
        path = Path(path)
        value = json.loads(path.read_text())
        inputs.append(dict(path=str(path), sha256=sha256(path)))
        return value

    def read_csv(path):
        path = Path(path)
        with path.open(newline='') as stream:
            value = list(csv.DictReader(stream))
        inputs.append(dict(path=str(path), sha256=sha256(path)))
        return value

    runs, closure, historical = Path(runs), Path(closure), Path(historical)
    completed = read(closure / 'receipt.json')
    selection = read(closure / 'selection.json')
    independent = read(closure / 'independent.json')
    fixed = read(closure / 'fixed_10m.json')
    if (completed['status'] != 'completed' or completed['independent_episodes'] != 500
            or completed['training_transitions'] != 0
            or completed['existing_three_method_results_not_changed'] is not True
            or selection['independent_test_used'] is not False
            or independent['independent_test_used_for_selection'] is not False
            or independent['episodes'] != 500 or fixed['training_seed_count'] != 3
            or selection['source_commit'] != completed['source_commit']):
        raise ValueError('Require complete campaign, fixed endpoint and independent closure')
    unsigned = {key: value for key, value in selection.items() if key != 'freeze_digest'}
    identity = canonical_digest(unsigned)
    if (identity != selection['freeze_digest'] or identity != completed['selection_freeze_digest']
            or identity != independent['selection_freeze_digest']
            or independent['selected'] != selection['selected']):
        raise ValueError('Frozen selection identity mismatch')
    candidates = selection['candidates']
    expected = {(seed, step) for seed in SEEDS for step in MILESTONES}
    if (len(candidates) != 15 or {(r['training_seed'], r['checkpoint_steps']) for r in candidates} != expected
            or len(fixed['by_training_seed']) != 3):
        raise ValueError('All15 candidates and three fixed endpoints are required')
    # Verify, do not change, the pre-test selection rule and identity.
    eligible = [r for r in candidates if r['mean_episode_cost'] <= 25]
    frontier = [r for r in candidates if not any(
        other['mean_episode_cost'] <= r['mean_episode_cost'] and
        other['mean_goals_per_episode'] >= r['mean_goals_per_episode'] and
        (other['mean_episode_cost'] < r['mean_episode_cost'] or
         other['mean_goals_per_episode'] > r['mean_goals_per_episode']) for other in candidates)]
    expected_selection = min(eligible or frontier, key=lambda r: (
        -r['mean_goals_per_episode'], r['mean_episode_cost'], r['mean_capped_first_goal_seconds'],
        r['run'], r['checkpoint_steps']))
    if selection['selected'] != expected_selection:
        raise ValueError('Frozen candidate does not follow the registered rule')
    history_curves = read_csv(historical / 'pointgoal_learning_curves.csv')
    history_fixed = read_csv(historical / 'pointgoal_fixed_10m_by_seed.csv')
    history_selected = read_csv(historical / 'pointgoal_selected_independent.csv')
    methods = {'sac_lag', 'stable_iso', 'ea'}
    if (len(history_curves) != 45 or len(history_fixed) != 9 or len(history_selected) != 3
            or {r['method'] for r in history_selected} != methods
            or {(r['method'], int(r['training_seed'])) for r in history_fixed}
            != {(method, seed) for method in methods for seed in SEEDS}):
        raise ValueError('Preserve the complete original three-method cohort')
    for method in methods:
        for seed in SEEDS:
            rows = [r for r in history_curves if r['method'] == method and int(r['training_seed']) == seed]
            if len(rows) != 5 or len({int(r['checkpoint_steps']) for r in rows}) != 5:
                raise ValueError('Keep all five historical nodes for every trained replica')
    curves = [dict(r) for r in history_curves]
    endpoints = [dict(r) for r in history_fixed]
    receipts = []
    registered = json.loads((Path(__file__).resolve().parents[2] /
                             'configs/paper_crax_payload/cvpo_formal_v4.json').read_text())
    for seed in SEEDS:
        directory = runs / f'cvpo-{seed}-10m'
        manifest = read(directory / 'run_config.json')
        receipt = read(directory / 'receipt.json')
        config = manifest['configuration']
        if (receipt['status'] != 'completed' or receipt['steps'] != 10000128
                or receipt['training_seed'] != seed or manifest['training_seed'] != seed
                or receipt.get('engineering_only') or receipt['evaluated_milestones'] != MILESTONES
                or config['training_seeds'] != SEEDS or config['steps_per_seed'] != 10000000
                or config['cost_limit'] != 25 or config['scientific_early_stop']
                or config['automatic_failed_restart'] or config != registered
                or receipt['official_source_commit'] != config['official_source_commit']
                or manifest['official_source_commit'] != config['official_source_commit']
                or receipt['source_commit'] != manifest['source_commit']
                or receipt['source_commit'] != completed['source_commit']):
            raise ValueError('Wrong scientific budget, registration or source identity')
        if receipt not in fixed['receipts']:
            raise ValueError('Fixed summary does not retain the native training receipts')
        receipts.append(receipt)
        for step in MILESTONES:
            candidate = next(r for r in candidates if (r['training_seed'], r['checkpoint_steps']) == (seed, step))
            path = directory / f'validation_{step}.json'
            value = read(path)
            if (value['split'] != 'common_validation' or value['steps'] != step
                    or value['training_seed'] != seed or value['independent_test_used'] is not False
                    or sha256(path) != candidate['validation_sha256']
                    or value['policy_sha256'] != candidate['policy_sha256']):
                raise ValueError('Wrong native validation identity or split')
            metrics = evaluation_means(value['results'], [61027, 61028, 61029, 61030, 61031], 50)
            for metric in METRICS:
                near(metrics[metric], candidate[metric.replace('mean_budget_compliant_fraction', 'budget_compliant_fraction')])
            row = dict(method='cvpo', run=directory.name, training_seed=seed, checkpoint_steps=step,
                **metrics, validation_seeds=5, validation_episodes_per_seed=50,
                role='fixed_common_validation_curve_not_independent_test')
            curves.append(row)
            if step == MILESTONES[-1]:
                if candidate not in fixed['by_training_seed']:
                    raise ValueError('Fixed endpoint replaced by a selected intermediate checkpoint')
                endpoints.append(dict(row, training_cumulative_cost=receipt['training_cost'],
                    total_training_seconds=receipt['elapsed_seconds'], complete_training_sps=receipt['complete_train_sps']))
    independent_metrics = evaluation_means(independent['per_seed'], [71027, 71028, 71029, 71030, 71031], 100)
    for metric in METRICS:
        near(independent_metrics[metric], independent[metric.replace('mean_budget_compliant_fraction', 'budget_compliant_fraction')])
    chosen = selection['selected']
    selected = [*history_selected, dict(method='cvpo', run=chosen['run'], training_seed=chosen['training_seed'],
        checkpoint_steps=chosen['checkpoint_steps'], episodes=500, **independent_metrics,
        role='one_validation_selected_policy_per_method; not_three_training_seed_mean')]
    summary = []
    for method in ('sac_lag', 'stable_iso', 'ea', 'cvpo'):
        rows = [r for r in endpoints if r['method'] == method]
        result = dict(method=method, training_seeds=3, nominal_training_budget=10000000,
            role='fixed_endpoint_mean_and_sample_SD_across_training_seeds')
        for metric in (*METRICS, 'training_cumulative_cost', 'total_training_seconds', 'complete_training_sps'):
            values = [float(r[metric]) for r in rows]
            result.update({metric + '_mean': mean(values), metric + '_sd': stdev(values)})
        summary.append(result)
    efficiency = [dict(method='cvpo', training_seed=r['training_seed'],
        actual_training_transitions=r['steps'], actual_new_transitions_in_this_run=r['actual_new_transitions'],
        training_cumulative_cost=r['training_cost'], official_policy_updates=r['official_updates'],
        native_evaluation_controls=r['evaluation_controls'], full_training_seconds=r['elapsed_seconds'],
        full_training_including_eval_save_sps=r['complete_train_sps'],
        source_commit=r['source_commit'], official_source_commit=r['official_source_commit']) for r in receipts]
    return dict(curves=curves, endpoints=endpoints, selected=selected, summary=summary,
        efficiency=efficiency,
        source_records=inputs, cvpo_actual_training_transitions=sum(r['steps'] for r in receipts),
        selection_freeze_digest=identity, cvpo_selection_status=selection['status'])


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(runs, closure, historical, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve earlier publication')
    data = prepare(runs, closure, historical)  # Reject incomplete science before making output.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(parents=True)
    for name, key in [('four_method_fixed_10m_by_seed', 'endpoints'),
                      ('four_method_fixed_10m_summary', 'summary'),
                      ('four_method_learning_curves', 'curves'),
                      ('four_method_selected_independent', 'selected'),
                      ('cvpo_training_efficiency', 'efficiency')]:
        write_csv(output / (name + '.csv'), data[key])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for method in ('sac_lag', 'stable_iso', 'ea', 'cvpo'):
        runs_by_seed = [[r for r in data['curves'] if r['method'] == method and int(r['training_seed']) == seed]
                        for seed in SEEDS]
        runs_by_seed = [sorted(rows, key=lambda r: int(r['checkpoint_steps'])) for rows in runs_by_seed]
        for ax, metric in zip(axes, ('mean_goals_per_episode', 'mean_episode_cost')):
            x = [int(r['checkpoint_steps']) / 1e6 for r in runs_by_seed[0]]
            values = [[float(row[metric]) for row in rows] for rows in runs_by_seed]
            means = [mean(column) for column in zip(*values)]
            errors = [stdev(column) for column in zip(*values)]
            line, = ax.plot(x, means, label=method)
            for rows, y in zip(runs_by_seed, values):
                ax.plot([int(r['checkpoint_steps']) / 1e6 for r in rows], y,
                        color=line.get_color(), alpha=.25, lw=.7)
            ax.fill_between(x, [m - s for m, s in zip(means, errors)],
                            [m + s for m, s in zip(means, errors)], color=line.get_color(), alpha=.1)
            ax.set(xlabel='Native training transitions (M)', ylabel=metric)
            ax.grid(alpha=.2)
    axes[1].axhline(25, color='black', ls='--', lw=1, label='Cost budget25')
    axes[0].legend(fontsize=8)
    fig.suptitle('Common validation: three training seeds, sample SD; not independent selected-policy results')
    fig.tight_layout()
    fig.savefig(output / 'four_method_learning.png', dpi=170)
    fig.savefig(output / 'four_method_learning.pdf')
    plt.close(fig)
    lines = ['# PointGoal四方法完整结果', '',
        '所有方法三训练seed、名义每seed10M。以下为固定终点均值±训练seed样本SD，',
        '不是选中策略独立回合均值。CVPO为新增独立30M包，历史三方法不重选。', '',
        '| 方法 | 目标数／回合 | 原生成本 | 训练累计成本 | 完整训练transitions/s |',
        '|---|---:|---:|---:|---:|']
    for row in data['summary']:
        cells = [row['method']]
        for key in ('mean_goals_per_episode', 'mean_episode_cost', 'training_cumulative_cost', 'complete_training_sps'):
            cells.append(f"{row[key + '_mean']:.3f}±{row[key + '_sd']:.3f}")
        lines.append('|' + '|'.join(cells) + '|')
    lines += ['', '全部15个CVPO候选保留，先共同验证冻结再打开500独立回合；',
        '选中单策略结果见four_method_selected_independent.csv，不替代上表三seed均值。',
        f"CVPO候选资格状态：{data['cvpo_selection_status']}；测试分数不用于重选。", '',
        '学习曲线含全部五节点、薄线为每训练seed、阴影为样本SD；成本预算25。',
        '训练累计成本、更新方式、原始时间与SPS单列。历史共享负载未严格控制，',
        'CVPO逐seed官方policy更新数和评测控制数见cvpo_training_efficiency.csv；',
        '不同算法的更新单位不等价，不据此假定相同optimizer机会。',
        '后端、向量数与更新组织不同，不以纯算法加速比解释墙钟差。',
        '新CVPO真实步数30,000,384（向量取整额外384）；不计入组件1.8M。',
        '汇总零新训练、零仿真、零机器人交互；未加载policy、replay或原始训练轨迹。']
    (output / 'INTERPRETATION_ZH.md').write_text('\n'.join(lines) + '\n')
    receipt = dict(status='completed_from_native_compact_records', source_records=data['source_records'],
        new_training_transitions=0, new_simulation_controls=0, physical_robot_interactions=0,
        represented_cvpo_training_transitions=data['cvpo_actual_training_transitions'],
        original_three_method_candidates_unchanged=True, selection_repeated=False,
        selection_freeze_digest=data['selection_freeze_digest'],
        fixed_endpoint_statistics_unit='three_training_seeds', selected_policy_training_seed_count=1)
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'closure', 'historical', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report(args.runs, args.closure, args.historical, args.output)


if __name__ == '__main__':
    main()
