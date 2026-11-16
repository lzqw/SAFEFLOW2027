"""Plot exact completed three-arm suffixes, without simulation or model imports."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    results = comparison['results']
    if (len(results) != 3 or {r['arm'] for r in results}
            != {'A-fixed', 'B-loss-anneal', 'C-joint-anneal'}
            or any(r['new_training_transitions'] != 150000 for r in results)
            or comparison['independent_confirmation']):
        raise ValueError('Require all three original complete paired DEV suffixes')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    maximum_common = min(r['violations'] for r in results)
    for result in results:
        curve = result['curve']
        line, = axes[0].step([0, *[r['violations'] for r in curve]],
                             [0, *[r['successes'] for r in curve]], where='post', label=result['arm'])
        axes[0].plot(result['violations'], result['successes'], 'o', color=line.get_color())
        rows = comparison['equal_suffix_violation_budget']
        points = [next(r for r in row['arms'] if r['arm'] == result['arm']) for row in rows]
        axes[1].plot([row['maximum_new_training_violations'] for row in rows],
                     [point['successes'] for point in points], 'o-', label=result['arm'])
    axes[0].axvline(maximum_common, color='gray', linestyle=':', label='Common maximum risk budget')
    axes[0].set(title='Every actual terminal prefix; no interpolation of logs',
                xlabel='New training violations, absolute100K to250K', ylabel='New safe successes')
    axes[1].set(title='Same suffix violation budget, each at its own prefix',
                xlabel='Maximum new training violations', ylabel='Safe successes before budget exceeded')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle('v6 complete150K suffixes from one observed100K parent; one seed, DEV only')
    fig.tight_layout(rect=(0, 0, 1, .94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), new_simulation_controls=0,
                         new_training_transitions=0, maximum_common_violation_budget=maximum_common)))


if __name__ == '__main__':
    main()
