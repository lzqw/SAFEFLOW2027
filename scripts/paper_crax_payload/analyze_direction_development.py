"""Exact chronology from completed v5 data-disk traces; no new interactions.

Use on the original server. Only sufficient JSON summaries are copied home.
All five registered arms are required for a final comparison, including failures
and timeouts; this tool never selects a new policy or candidate checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.analyze_component_trajectories import (
    analyze, compact_summary, compare_summaries)
from scripts.paper_crax_payload.train_direction_development import validate_config


def summarize_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'run_config.json').read_text())
    receipt = json.loads((directory/'receipt.json').read_text())
    config = manifest['configuration']; validate_config(config)
    registered = next(a for a in config['arms'] if a == manifest['arm'])
    if (manifest['format'] != 'direction_development_v5_full' or manifest['engineering_only']
            or receipt['engineering_only'] or receipt['status'] != 'completed'
            or receipt['steps'] != config['steps_per_run'] or receipt['arm'] != registered
            or receipt['training_seed'] != manifest['training_seed']
            or manifest['training_seed'] not in config['development_training_seeds']
            or receipt['source_commit'] != manifest['source_commit']
            or receipt['initial_network_sha256'] != manifest['initial_network_sha256']
            or receipt['independent_endpoint_test_executed']):
        raise ValueError('Require the original complete development run, never engineering/test replacement')
    milestones = [0, *config['evaluation']['milestones']]
    trajectory_analysis = dict(windows=[list(pair) for pair in zip(milestones[:-1], milestones[1:])],
                               world_grid_m=.02, arclength_spacing_m=.005)
    result = compact_summary(analyze(directory, trajectory_analysis=trajectory_analysis))
    if not result['explicit_trace_end'] or result['steps'] != config['steps_per_run']:
        raise ValueError('Completed receipt without a closed, complete trajectory')
    result.update(training_seed=manifest['training_seed'], independent_confirmation=False,
        trajectory_scope='all_actual_training_steps_including_failures_and_censored_prefixes',
        geometry_scope='descriptive_TCP_grid_not_joint_space_topology',
        manifest_sha256=hashlib.sha256((directory/'run_config.json').read_bytes()).hexdigest(),
        receipt_sha256=hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest(),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_analysis_sha256=hashlib.sha256(Path(__file__).with_name('analyze_component_trajectories.py').read_bytes()).hexdigest(),
        trajectory_analysis=trajectory_analysis,
        new_training_transitions=0, new_simulation_controls=0)
    return result


def combine(config, results):
    validate_config(config)
    expected = {a['name'] for a in config['arms']}
    if (len(results) != len(expected) or {r['arm'] for r in results} != expected
            or len({r['initial_network_sha256'] for r in results}) != 1
            or len({r['training_seed'] for r in results}) != 1
            or len({r['analysis_script_sha256'] for r in results}) != 1
            or len({r['shared_analysis_sha256'] for r in results}) != 1
            or len({json.dumps(r['trajectory_analysis'], sort_keys=True) for r in results}) != 1
            or any(r['steps'] != config['steps_per_run'] or not r['completed_receipt']
                   or not r['explicit_trace_end'] or r['independent_confirmation'] for r in results)):
        raise ValueError('Require all five same-initialization complete development arms')
    result = compare_summaries(results, matching_arms=[a['name'] for a in config['arms']])
    result.update(protocol=config['protocol'], split='development_only', method_selected=False,
                  independent_confirmation=False, new_training_transitions=0, new_simulation_controls=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', type=Path)
    group.add_argument('--summaries', type=Path, nargs='+')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve completed analysis evidence')
    if args.run: result = summarize_run(args.run)
    else:
        if args.config is None: parser.error('Combining requires the registered configuration')
        result = combine(json.loads(args.config.read_text()),
                         [json.loads(path.read_text()) for path in args.summaries])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), new_training_transitions=0, new_simulation_controls=0)))


if __name__ == '__main__': main()
