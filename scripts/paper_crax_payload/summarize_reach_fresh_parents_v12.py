"""Read-only learning diagnostics for both registered fresh native Reach parents.

No checkpoint selection, retries, new physics, or modification of live runs.
An incomplete curve is explicitly partial, even if its last logged step is 300K.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.paper_crax_payload.summarize_simulation_v12 import output


PROJECT = Path(__file__).resolve().parents[2]
PLAN = PROJECT/'configs/paper_crax_payload/simulation_v12_reach_fresh_parents.json'
COUNTERS = ('episodes', 'training_safe_successes', 'training_violation_count',
            'training_safe_timeouts', 'learner_updates', 'reward_updates', 'actor_updates')


def diagnose(rows, budget):
    if not rows:
        raise ValueError('A saved native progress curve is required')
    previous = None
    for row in rows:
        if (not 0 < row['steps'] <= budget
                or any(isinstance(v, float) and not math.isfinite(v) for v in row.values())
                or any(not isinstance(row[k], int) or row[k] < 0 for k in COUNTERS)
                or row['episodes'] != sum(row[k] for k in COUNTERS[1:4])):
            raise ValueError('Invalid finite native progress/episode accounting')
        if previous is not None and (row['steps'] <= previous['steps']
                or row['cumulative_elapsed_seconds'] <= previous['cumulative_elapsed_seconds']
                or any(row[k] < previous[k] for k in COUNTERS)):
            raise ValueError('Do not join retries or nonmonotone counters')
        previous = row
    last = rows[-1]
    # Recent wall-clock throughput includes any native evaluation within the window.
    # It does not extrapolate from the random-only, much faster training prefix.
    anchor = next((r for r in reversed(rows[:-1]) if last['steps']-r['steps'] >= 20000), None)
    sps = ((last['steps']-anchor['steps']) /
           (last['cumulative_elapsed_seconds']-anchor['cumulative_elapsed_seconds'])) if anchor else None
    first = next((r['steps'] for r in rows if r['training_safe_successes']), None)
    return dict(steps=last['steps'], budget=budget, **{k:last[k] for k in COUNTERS},
        first_success_logged_at_or_before_step=first,
        no_success_observed_through_step=last['steps'] if first is None else None,
        recent_safe_success=last['recent_safe_success'], recent_return=last['recent_return'],
        alpha=last['alpha'], safety_multiplier=last['safety_multiplier'],
        q_policy_mean=last['q_policy_mean'], policy_cost=last['policy_cost'],
        actor_updates_advancing_in_last_log_interval=(len(rows)>1 and
            last['actor_updates']>rows[-2]['actor_updates']),
        elapsed_seconds=last['cumulative_elapsed_seconds'], recent_window_sps=sps,
        approximate_remaining_hours=(budget-last['steps'])/sps/3600 if sps else None,
        eta_includes_future_evaluation_overhead=False,
        independent_test_used=False, winner_confirmation=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(PLAN.read_text())
    base = json.loads((PROJECT/plan['base_config']).read_text())
    summaries, curves, nodes, episodes, provenance = [], [], [], [], []

    def read(path):
        raw = path.read_bytes()
        provenance.append(dict(file=str(path.relative_to(args.runs)),
            sha256=hashlib.sha256(raw).hexdigest()))
        return raw.decode()

    for seed in plan['training_seeds']:
        directory = args.runs/f'Reach-EA-{seed}'
        manifest = json.loads(read(directory/'run_config.json'))
        expected = json.loads(json.dumps(base))
        expected['development_seed'] = seed
        expected['evaluation']['development_layout_seed_start'] = plan['development_layout_seed_start']
        expected['evaluation']['action_sampling_seed'] = plan['action_sampling_seed']
        if (manifest['training_seed'] != seed or manifest['initial_steps'] != 0
                or manifest['parent_checkpoint'] is not None
                or manifest['initialization'] != 'fresh_Piper_no_Panda_transfer'
                or manifest['source_commit'] != '22d4966f'
                or manifest['total_transition_budget'] != plan['training_steps_each']
                or manifest['configuration'] != expected):
            raise ValueError('Require both registered independent fresh native parents')
        rows = [json.loads(line) for line in read(directory/'progress.jsonl').splitlines()]
        summary = dict(seed=seed, **diagnose(rows, plan['training_steps_each']))
        # A logged final step alone is not a completed training/evaluation receipt.
        final = directory/f"training_receipt_{plan['training_steps_each']}.json"
        summary['final_native_receipt_present'] = False
        if final.exists():
            receipt = json.loads(read(final))
            if (receipt['status'] != 'completed_native_checkpoint'
                    or receipt['training_seed'] != seed
                    or receipt['training_steps'] != plan['training_steps_each']
                    or rows[-1]['steps'] != plan['training_steps_each']
                    or any(receipt[k] != rows[-1][k] for k in COUNTERS[1:4])):
                raise ValueError('Final native receipt and curve disagree')
            summary['final_native_receipt_present'] = True
        summaries.append(summary)
        curves.extend(dict(seed=seed, **row) for row in rows)
        for path in sorted(directory.glob('evaluation_*.json')):
            step = int(path.stem.split('_')[-1])
            if step > rows[-1]['steps']:
                raise ValueError('Evaluation snapshot is ahead of its progress snapshot; resync')
            raw = json.loads(read(path))
            if (raw['new_training_transitions'] != 0 or raw['adaptation_updates'] != 0
                    or raw['training_rng_preserved'] is not True
                    or [c['condition'] for c in raw['conditions']] !=
                        [c['name'] for c in expected['evaluation']['conditions']]):
                raise ValueError('Require frozen native evaluation with all registered conditions')
            for condition in raw['conditions']:
                records = condition['records']
                n = len(records)
                if (n != expected['evaluation']['development_episodes_per_condition']
                        or [r['seed'] for r in records] != list(range(plan['development_layout_seed_start'],
                            plan['development_layout_seed_start']+n))
                        or any(r['safe_success'] and r['violation'] for r in records)):
                    raise ValueError('Require every paired DEV layout and terminal class')
                counts = dict(safe_success_count=sum(r['safe_success'] for r in records),
                    violation_count=sum(r['violation'] for r in records),
                    safe_timeout_count=sum(not r['safe_success'] and not r['violation'] for r in records))
                if any(condition[k] != v for k,v in counts.items()):
                    raise ValueError('Native DEV totals disagree with per-episode outcomes')
                nodes.append(dict(seed=seed, steps=step, condition=condition['condition'], episodes=n,
                    **counts, mean_return=sum(r['episode_return'] for r in records)/n))
                episodes.extend(dict(training_seed=seed, steps=step, condition=condition['condition'],
                    **{k:v for k,v in r.items() if k != 'trajectory'}) for r in records)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, values in (('parents.csv', summaries), ('training.csv', curves),
            ('development_nodes.csv', nodes), ('development_episodes.csv', episodes)):
        output(args.output/filename, values)
    receipt = dict(status=('native_parent_learning_diagnostic_receipts_present'
        if all(r['final_native_receipt_present'] for r in summaries)
        else 'partial_native_parent_learning_diagnostic'), parents=summaries,
        completed_parents=sum(r['final_native_receipt_present'] for r in summaries),
        observed_training_steps=sum(r['steps'] for r in summaries), registered_steps=600000,
        new_training_steps=0, new_simulation_controls=0, optimizer_updates=0,
        independent_test_used=False, winner_confirmation=False, automatic_retry=False,
        omitted_registered_seeds=[], provenance=provenance)
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
