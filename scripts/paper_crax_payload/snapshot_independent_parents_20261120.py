"""Read-only compact snapshots of the three registered native parents, no models."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

from scripts.paper_crax_payload.summarize_reach_fresh_parents_v12 import diagnose
from scripts.paper_crax_payload.summarize_simulation_v12 import output


PROJECT = Path(__file__).resolve().parents[2]
CASES = {
    41030: ('447', 'Reach', 'runs-third-parent-20261120/Reach-EA-41030',
            '58bf769dceb1eabd2eec07c627decf68d810c2aa', 11990000, 11991000),
    42028: ('CTAlab', 'Payload', 'runs-payload-parents-20261120/Payload-EA-42028',
            '870e7e81ce29ab0b06a24fe546544d1f848ad666', 12010000, 12011000),
    42029: ('447', 'Payload', 'runs-payload-parents-20261120/Payload-EA-42029',
            '870e7e81ce29ab0b06a24fe546544d1f848ad666', 12010000, 12011000),
}
ROOTS = {'CTAlab': Path('/storage_0/SAFEFLOW2027_lzqw_uid1007/simulation_v12'),
         '447': Path('/home/carla/4T/SAFEFLOW2027_SIMULATION_V12')}


def complete_rows(raw):
    # Never parse a partially appended final JSON line as a failed run.
    lines = raw.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n')]


def summarize_node(value, seed, step, layout_start, task):
    if (value['new_training_transitions'] != 0 or value['adaptation_updates'] != 0
            or value['training_rng_preserved'] is not True or value['task'] != task
            or value['training_seed'] != seed or value['checkpoint_steps'] != step
            or value['method'] != 'EA'
            or [g['condition'] for g in value['conditions']] !=
                ['nominal', 'new_geometry', 'execution_scale_085']):
        raise ValueError('Only this native fixed development node and every registered condition')
    nodes, episodes = [], []
    for group in value['conditions']:
        records = group['records']
        if (len(records) != 30 or group['episodes'] != 30
                or [r['seed'] for r in records] != list(range(layout_start, layout_start+30))
                or any(r['safe_success'] and r['violation'] for r in records)):
            raise ValueError('Retain all registered layouts and failures')
        identity = dict(training_seed=seed, task=task, steps=step, condition=group['condition'])
        counts = dict(safe_success_count=sum(r['safe_success'] for r in records),
            violation_count=sum(r['violation'] for r in records),
            safe_timeout_count=sum(not r['safe_success'] and not r['violation'] for r in records))
        if any(group[k] != v for k,v in counts.items()):
            raise ValueError('Native totals must match every episode')
        nodes.append(dict(**identity, episodes=30, **counts,
            mean_return=sum(r['episode_return'] for r in records)/30,
            mean_capped_seconds=sum(r['time_to_goal_capped_seconds'] for r in records)/30))
        episodes.extend(dict(**identity, **{k:v for k,v in r.items() if k != 'trajectory'}) for r in records)
    return nodes, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    host = platform.node()
    root = ROOTS[host].resolve(strict=True)
    if not args.output.resolve().is_relative_to(root) or args.output.exists():
        raise ValueError('New snapshot on this host data disk only; preserve previous evidence')
    if shutil.disk_usage(root).free < 41*1024**3:
        raise OSError('Preserve40GiB plus compact snapshot allowance')
    training, nodes, episodes, summaries, provenance, handles = [], [], [], [], [], []
    base_path = PROJECT/'configs/paper_crax_payload/piper_sim_v1.json'
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != '84ff1e945d3003a30e58fb5d8e5202371b16cafd9401d00d286ccc4c5102285a':
        raise ValueError('Original recipe changed')
    for seed, (case_host, task, relative, source, layout_start, action_start) in CASES.items():
        if case_host != host:
            continue
        directory = root/relative
        def read(path):
            raw = path.read_bytes()
            provenance.append(dict(training_seed=seed, file=str(path.relative_to(root)),
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
            return raw
        manifest = json.loads(read(directory/'run_config.json'))
        expected = deepcopy(json.loads(base_path.read_text()))
        expected['development_seed'] = seed
        expected['evaluation']['development_layout_seed_start'] = layout_start
        expected['evaluation']['action_sampling_seed'] = action_start
        if (manifest['host'] != host or manifest['task'] != task or manifest['method'] != 'EA'
                or manifest['training_seed'] != seed or manifest['source_commit'] != source
                or manifest['configuration'] != expected or manifest['initial_steps'] != 0
                or manifest['total_transition_budget'] != 300000
                or manifest['parent_checkpoint'] is not None or manifest['engineering'] is not False):
            raise ValueError('Require the exact fresh scientific parent, not an adapted or replacement run')
        # Evaluations are atomically finalized. Read them before the growing log,
        # so each archived evaluation has its logged progress prefix available.
        evaluations = [(path, json.loads(read(path))) for path in sorted(directory.glob('evaluation_*.json'))]
        rows = complete_rows(read(directory/'progress.jsonl'))
        if [r['steps'] for r in rows] != list(range(1000, rows[-1]['steps']+1, 1000)):
            raise ValueError('Whole observed native1K prefix required')
        summary = dict(training_seed=seed, task=task, **diagnose(rows, 300000))
        proc = Path('/proc', str(manifest['native_pid']), 'cmdline')
        try:
            command = proc.read_bytes()
        except FileNotFoundError:
            command = b''
        handles.append(dict(training_seed=seed, native_pid=manifest['native_pid'],
            native_handle_live=str(directory).encode() in command and b'train_' in command))
        summary['complete_native_receipt_observed'] = False
        if (directory/'receipt.json').exists():
            receipt = json.loads(read(directory/'receipt.json'))
            if (receipt['status'] != 'completed' or receipt['training_steps'] != 300000
                    or receipt['training_seed'] != seed or receipt['initial_steps'] != 0
                    or receipt['newly_collected_transitions'] != 300000
                    or receipt['training_source_commit'] != source or rows[-1]['steps'] != 300000):
                raise ValueError('A final logged step alone does not prove completion')
            summary['complete_native_receipt_observed'] = True
        summaries.append(summary)
        training.extend(dict(training_seed=seed, task=task, **r) for r in rows)
        for path, value in evaluations:
            step = int(path.stem.split('_')[-1])
            if step not in (60000, 120000, 180000, 240000, 300000) or step > rows[-1]['steps']:
                raise ValueError('Registered milestone cannot exceed observed progress')
            ns, es = summarize_node(value, seed, step, layout_start, task)
            nodes.extend(ns)
            episodes.extend(es)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in [('parents.csv',summaries), ('training.csv',training),
                          ('development_nodes.csv',nodes), ('development_episodes.csv',episodes)]:
        output(args.output/name, records)
    (args.output/'receipt.json').write_text(json.dumps(dict(status='observed_native_parent_development_snapshot',
        host=host, unix_time=time.time(), parents=summaries, native_handles=handles, provenance=provenance,
        development_nodes=len(nodes), development_episodes=len(episodes),
        complete_three_seed_confirmation=False, independent_test_used=False, winner_confirmation=False,
        new_training_steps=0, new_simulation_controls=0, physical_robot_interactions=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
