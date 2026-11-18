"""Descriptive paired comparisons at every saved early-screen node, not seed inference."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def save(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    dev = read(args.analysis/'dev_nodes.csv'); train = read(args.analysis/'training.csv')
    candidate = 'P5-fixed-2033'; comparisons = []
    for value in dev:
        if value['run'] != candidate:
            continue
        for baseline in ('iso-2033', 'metric-only-2033'):
            ref = next(r for r in dev if (r['run'], r['condition'], r['new_steps']) ==
                       (baseline, value['condition'], value['new_steps']))
            delta = {key: float(value[key])-float(ref[key]) for key in
                     ('safe_successes', 'violations', 'timeouts', 'stall_count', 'mean_return', 'mean_capped_seconds')}
            no_worse = (delta['safe_successes'] >= 0 and delta['violations'] <= 0 and
                        delta['timeouts'] <= 0 and delta['stall_count'] <= 0 and
                        delta['mean_return'] >= 0 and delta['mean_capped_seconds'] <= 0)
            comparisons.append(dict(candidate=candidate, baseline=baseline,
                transitions=int(value['new_steps']), condition=value['condition'],
                episodes=int(value['episodes']), **{'delta_'+k: v for k, v in delta.items()},
                descriptive_all_metrics_no_worse=no_worse))
    paired_train = []
    for value in train:
        if value['run'] != candidate:
            continue
        for baseline in ('iso-2033', 'metric-only-2033'):
            ref = next(r for r in train if (r['run'], r['steps']) == (baseline, value['steps']))
            delta = {key: float(value[key])-float(ref[key]) for key in
                     ('training_violations', 'safe_successes', 'safe_timeouts', 'training_stall_episodes')}
            no_worse = (delta['training_violations'] <= 0 and delta['safe_successes'] >= 0 and
                        delta['safe_timeouts'] <= 0 and delta['training_stall_episodes'] <= 0)
            paired_train.append(dict(candidate=candidate, baseline=baseline, transitions=int(value['steps']),
                **{'delta_'+k: v for k, v in delta.items()}, descriptive_count_metrics_no_worse=no_worse))
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'development_pairwise.csv', comparisons)
    save(args.output/'training_pairwise.csv', paired_train)
    summary = []
    for baseline in ('iso-2033', 'metric-only-2033'):
        d = [r for r in comparisons if r['baseline'] == baseline]
        t = [r for r in paired_train if r['baseline'] == baseline]
        summary.append(dict(baseline=baseline, development_nodes=len(d),
            development_all_metrics_no_worse_nodes=sum(r['descriptive_all_metrics_no_worse'] for r in d),
            training_nodes=len(t), training_count_metrics_no_worse_nodes=sum(r['descriptive_count_metrics_no_worse'] for r in t),
            first_training_violation_excess=next((r['transitions'] for r in t if r['delta_training_violations'] > 0), None)))
    (args.output/'summary.json').write_text(json.dumps(dict(candidate=candidate, comparisons=summary,
        interpretation='descriptive only;one training seed;no universal dominance or statistical significance claim',
        earliest_node=1000, unobserved_between_nodes=True, random_prefix_may_tie=True,
        new_training_steps=0, new_simulation_controls=0, independent_test_used=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
