"""One compact table of complete DEV endpoints, with missing metrics explicit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


SOURCES = ('round1_analysis', 'round2_analysis', 'round3_analysis',
           'payload_round1_analysis', 'early_complete_analysis', 'followup_1357_analysis')


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def truth(value):
    if value not in ('True', 'False', 'true', 'false'):
        raise ValueError('Do not turn missing measurements into false/zero')
    return value.lower() == 'true'


def measured_mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key, '') != '']
    return mean(values) if len(values) == len(rows) else None


def measured_count(rows, key):
    return sum(truth(r[key]) for r in rows) if all(r.get(key, '') != '' for r in rows) else None


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); train = []; nodes = []; provenance = []; pending = []
    for source in SOURCES:
        folder = args.reports/source
        receipt = json.loads((folder/'receipt.json').read_text())
        complete = {r['run']: r for r in receipt['completed_native_receipts']}
        progress = read(folder/'training.csv'); episodes = read(folder/'dev_episodes.csv')
        pending.extend(dict(source=source, run=name, reason='no complete100Kreceipt;excluded_from_endpoint_table')
                       for name in sorted({r['run'] for r in progress}-complete.keys()))
        for name, record in complete.items():
            if record['status'] != 'completed' or record.get('new_training_steps', record.get('new_training_transitions')) != 100000:
                raise ValueError('Only actual complete registered100K endpoints belong here')
            counts = record.get('segment_counts', record)
            metrics = record.get('training_metrics', {})
            native = 'segment_counts' in record
            task = record.get('line', 'panda_safereach')
            # Native forks retain one registered parent seed per task. Fresh
            # screen receipts carry the actual initialization seed directly.
            seed = record.get('training_seed')
            stage_plan = {'round1_analysis': 'simulation_v12_piper_round1.json',
                'round2_analysis': 'simulation_v12_piper_round2.json',
                'round3_analysis': 'simulation_v12_piper_round3.json',
                'payload_round1_analysis': 'simulation_v12_payload_round1.json'}
            plan_name = stage_plan.get(source)
            if source == 'followup_1357_analysis':
                plan_name = 'simulation_v12_followup_'+task+'.json'
            if native:
                plan_path = args.reports.parents[1]/'configs/paper_crax_payload'/plan_name
                seed = json.loads(plan_path.read_text())['lines'][task]['seed']
            if seed is None:
                raise ValueError('Training seed provenance must be explicit')
            final = [r for r in progress if r['run'] == name and
                     int(r.get('new_steps') or r.get('steps')) == 100000]
            if len(final) != 1:
                raise ValueError('Need one complete native endpoint progress row')
            violation_key = 'violations' if native else 'training_violations'
            if int(counts[violation_key]) != int(final[0][violation_key]):
                raise ValueError('Native receipt and endpoint progress disagree')
            row = dict(source=source, run=name, task=task, training_seed=seed,
                new_training_transitions=100000, training_violations=counts[violation_key],
                training_safe_successes=counts['safe_successes'], training_safe_timeouts=counts['safe_timeouts'],
                training_stall_episodes=metrics.get('training_stall_episodes'),
                ordinary_terminal_task_successes=metrics.get('ordinary_terminal_task_successes'),
                training_cumulative_return=metrics.get('training_cumulative_return'),
                source_commit=record['source_commit'], independent_test_used=False)
            train.append(row)
            final_episodes = [r for r in episodes if r['run'] == name and int(r['new_steps']) == 100000]
            for condition in sorted({r['condition'] for r in final_episodes}):
                group = [r for r in final_episodes if r['condition'] == condition]
                success = sum(truth(r['safe_success']) for r in group)
                bad = sum(truth(r['violation']) for r in group)
                timeouts = len(group)-success-bad
                if any(truth(r['safe_success']) and truth(r['violation']) for r in group):
                    raise ValueError('Terminal classes must remain exclusive')
                capped = 'capped_seconds' if 'capped_seconds' in group[0] else 'time_to_goal_capped_seconds'
                nodes.append(dict(source=source, run=name, task=task, training_seed=seed,
                    condition=condition, development_episodes=len(group), safe_success_count=success,
                    safe_success_percent=100*success/len(group), violation_count=bad, timeout_count=timeouts,
                    stall_count=measured_count(group, 'stall'),
                    ordinary_terminal_task_success_count=measured_count(group, 'ordinary_terminal_task_success'),
                    mean_return=measured_mean(group, 'episode_return'),
                    mean_horizon_capped_episode_seconds=measured_mean(group, capped),
                    mean_of_episode_action_p50_ms=measured_mean(group, 'inference_p50_ms'),
                    mean_of_episode_action_p95_ms=measured_mean(group, 'inference_p95_ms'),
                    inference_scope='sharedGPU_full_batch1_action_call' if measured_mean(group, 'inference_p50_ms') is not None else 'not_measured_in_this_screen',
                    independent_test_used=False))
        provenance.extend(dict(source=str(folder/name), sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest())
                          for name in ('receipt.json', 'training.csv', 'dev_episodes.csv'))
    identities = [(r['task'], r['run']) for r in train]
    if len(set(identities)) != len(identities):
        raise ValueError('Do not double count a run copied into multiple snapshots')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'training_endpoints.csv', train); write(args.output/'development_endpoints.csv', nodes)
    (args.output/'receipt.json').write_text(json.dumps(dict(
        completed_runs=len(train), completed_new_training_transitions=sum(r['new_training_transitions'] for r in train),
        development_endpoint_conditions=len(nodes), development_endpoint_episodes=sum(r['development_episodes'] for r in nodes),
        excluded_partial_runs=pending, provenance=provenance, missing_metrics='blank_not_zero',
        inference_aggregation='mean of per-episode action quantiles;not pooled action latency quantiles',
        task_duration_is_not_inference_latency=True, new_training_steps=0, new_simulation_controls=0,
        evaluation_scope='DEV_only;one_training_seed_per_task;no_crossseed_or_independent_claim'), indent=2)+'\n')


if __name__ == '__main__': main()
