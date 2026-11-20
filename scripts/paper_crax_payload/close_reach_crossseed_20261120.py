"""Finite half-hour completion observer and lossless DEV summary for one host.

No training launch, retries, policy calls, checkpoint selection or independent test.
Sampled and zero-noise records always retain distinct condition identities.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import time

from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import sha256, validate_resolved
from scripts.paper_crax_payload.summarize_simulation_v12 import output


def read(path):
    return json.loads(path.read_text())


def summarize_groups(value, plan, arm, step):
    modes = ['sampled', 'zero_noise'] if step in (100000, 150000) else ['sampled']
    conditions = [c['name'] for c in plan['lines']['piper_reach']['conditions']]
    expected = [(mode, condition) for mode in modes for condition in conditions]
    groups = value['conditions']
    if (value.get('independent_test_used') is not False
            or value.get('fixed_node_not_best_checkpoint') is not True
            or value.get('arm') != arm or value.get('new_steps') != step
            or value.get('deployment_modes') != modes
            or [(g['deployment_mode'], g['condition']) for g in groups] != expected):
        raise ValueError('Require every registered condition and deployment mode at the fixed node')
    seed = plan['compiled_training_seed']
    start = plan['lines']['piper_reach']['development_layout_seed_start']
    nodes, episodes = [], []
    for group in groups:
        records = group['records']
        if len(records) != 100 or [r['seed'] for r in records] != list(range(start, start+100)):
            raise ValueError('Keep all 100 registered layout seeds including failures')
        if any(r['safe_success'] and r['violation'] for r in records):
            raise ValueError('Safe success and violation cannot overlap')
        identity = dict(training_seed=seed, arm=arm, new_steps=step,
            condition=group['condition'], deployment_mode=group['deployment_mode'])
        nodes.append(dict(**identity, episodes=100,
            safe_successes=sum(bool(r['safe_success']) for r in records),
            violations=sum(bool(r['violation']) for r in records),
            timeouts=sum(not r['safe_success'] and not r['violation'] for r in records),
            stall_count=sum(bool(r['stall']) for r in records),
            mean_return=statistics.mean(r['episode_return'] for r in records),
            mean_capped_seconds=statistics.mean(r['time_to_goal_capped_seconds'] for r in records),
            independent_test_used=False))
        episodes.extend(dict(**identity,
            **{k: v for k, v in r.items()
               if k not in ('trajectory', 'trajectory_tcp_xyz', 'control_details', 'layout')})
            for r in records)
    return nodes, episodes


def summarize(root, admission, destination):
    jobs = [j for j in admission['jobs'] if j['host'] == platform.node()]
    if len(jobs) != 3 or len({j['seed'] for j in jobs}) != 1:
        raise ValueError('One host must retain exactly one seed and all three methods')
    training, nodes, episodes, receipts, provenance = [], [], [], [], []
    for job in jobs:
        directory = Path(job['output'])
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError('Only this admitted data disk may be read')
        manifest, receipt = read(directory/'run_config.json'), read(directory/'receipt.json')
        plan = manifest['plan']
        line = validate_resolved(plan)
        if (manifest['host'] != job['host'] or manifest['arm'] != job['arm']
                or line['seed'] != job['seed'] or manifest['parent_sha256'] != line['parent_sha256']
                or manifest['source_commit'] != admission['runner_source_commit']
                or receipt['source_commit'] != admission['runner_source_commit']
                or receipt['status'] != 'completed' or receipt['steps'] != 550000
                or receipt['new_training_steps'] != 150000):
            raise ValueError('Require complete registered native 150K suffixes and exact provenance')
        rows = [json.loads(s) for s in (directory/'progress.jsonl').read_text().splitlines()]
        if [r['new_steps'] for r in rows] != list(range(1000, 150001, 1000)):
            raise ValueError('Retain every 1K learning curve node')
        for row in rows:
            if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                raise ValueError('Nonfinite native learning curve')
            if row['episodes'] != row['safe_successes']+row['violations']+row['safe_timeouts']:
                raise ValueError('Retain all terminal classes')
        if any(rows[-1][k] != v for k, v in receipt['segment_counts'].items()):
            raise ValueError('Receipt and full native curve disagree')
        if sorted(p.name for p in directory.glob('development_*.json')) != sorted(
                f'development_{step}.json' for step in plan['milestones']):
            raise ValueError('Exactly the six fixed development nodes required')
        training.extend(dict(training_seed=job['seed'], arm=job['arm'], **r) for r in rows)
        for step in plan['milestones']:
            ns, es = summarize_groups(read(directory/f'development_{step}.json'), plan, job['arm'], step)
            nodes.extend(ns); episodes.extend(es)
        receipts.append(dict(training_seed=job['seed'], **receipt))
        provenance.append(dict(seed=job['seed'], arm=job['arm'],
            parent_sha256=line['parent_sha256'], receipt_sha256=sha256(directory/'receipt.json'),
            manifest_sha256=sha256(directory/'run_config.json'),
            endpoint_checkpoint_sha256=sha256(directory/'full_new_150000.pt')))
    destination.mkdir(parents=True, exist_ok=False)
    for name, records in [('training.csv', training), ('development_nodes.csv', nodes),
                          ('development_episodes.csv', episodes)]:
        output(destination/name, records)
    (destination/'receipt.json').write_text(json.dumps(dict(
        status='complete_crossseed_one_host_development_summary', host=platform.node(),
        training_seed=jobs[0]['seed'], completed_runs=3,
        completed_new_training_transitions=450000,
        inherited_parent_transitions_not_new=400000,
        progress_rows=len(training), development_nodes=len(nodes),
        development_episodes=len(episodes), native_receipts=receipts, provenance=provenance,
        new_training_steps=0, new_simulation_controls=0, independent_test_used=False,
        winner_confirmation=False, all_registered_nodes_retained=True,
        physical_robot_interactions=0), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--maximum-checks', type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.maximum_checks <= 48:
        raise ValueError('At most 24 hours of finite observation')
    root = args.data_root.resolve(strict=True)
    admission = read(root/'crossseed900K-20261120/execution_registration.json')
    if admission['data_roots_by_host'].get(platform.node()) != str(root):
        raise ValueError('Wrong data disk or native host')
    expected = [j['output'] for j in admission['jobs'] if j['host'] == platform.node()]
    for check in range(args.maximum_checks):
        queue = read(root/'queue_crossseed900K_20261120.json')
        if queue['source_commit'] != admission['runner_source_commit']:
            raise ValueError('Wrong native queue source')
        print(json.dumps(dict(check=check+1, queue_status=queue['status'],
            interval_seconds=1800, training_launches=0, automatic_retry=False)), flush=True)
        if queue['status'] in ('completed', 'completed_with_failures'):
            if (queue['status'] != 'completed' or queue['pending'] or queue['live_native_pids']
                    or [j['output'] for j in queue['jobs']] != expected
                    or any(j['status'] != 'completed' or j['exit_code'] != 0 for j in queue['jobs'])):
                raise ValueError('Preserve failed/partial campaigns; do not summarize as complete')
            for job in queue['jobs']:
                cmdline = Path('/proc', str(job['pid']), 'cmdline')
                try:
                    command = cmdline.read_bytes()
                except FileNotFoundError:
                    command = b''
                if job['output'].encode() in command:
                    raise ValueError('Native training still live; no terminal summary')
            if shutil.disk_usage(root).free < 41*1024**3:
                raise OSError('Preserve 40GiB floor plus summary growth')
            summarize(root, admission, root/'crossseed900K-complete-analysis-20261120')
            return
        if check+1 < args.maximum_checks:
            time.sleep(1800)
    raise TimeoutError('Finite observer ended; training remains untouched')


if __name__ == '__main__':
    main()
