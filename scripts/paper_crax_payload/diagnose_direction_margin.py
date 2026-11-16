"""Frozen150K margin calibration on the last10K ALREADY COLLECTED controls.

No environment imports, simulator calls, policy action generation or updates.
These are retrospective training actions, not independent tests or calibration
coverage guarantees. Reconstruct the pre-action observation per env/episode;
the trace step's observation is the next observation, NOT the model input.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.paper_crax_payload.direction_development import build_development
from scripts.paper_crax_payload.train_direction_development import validate_config


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def terminal_records(directory, start=140000, end=150000):
    states = {}; selected = []; total = 0; closed = False
    with gzip.open(directory/'trajectories.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            kind = row['record_type']
            if kind == 'initial': states[row['env'], row['episode_id']] = row['observation']
            elif kind == 'step':
                total += 1
                if row['transitions'] != total: raise ValueError('Missing or duplicate actual controls')
                key = (row['env'], row['episode_id']); observation = states[key]
                if start < total <= end:
                    selected.append(dict(steps=total, obs=observation, action=row['executed_action'],
                        margin=row['margin'], violation=row['violation']))
                if row['done']: del states[key]
                else: states[key] = row['observation']
            elif kind == 'end': closed = True
    if not closed or total != end or len(selected) != end-start:
        raise ValueError('Require the complete original150K trace and every last10K actual action')
    return selected


def calibration_statistics(heads, actual):
    heads = np.asarray(heads, dtype=float); actual = np.asarray(actual, dtype=float)
    mean = heads.mean(0); std = heads.std(0, ddof=0)
    groups = dict(all=np.ones_like(actual, dtype=bool), unsafe_actual=actual > 0.,
        near_boundary_actual=np.abs(actual) <= .005, safe_interior_actual=actual < -.005)
    result = {}
    for name, mask in groups.items():
        count = int(mask.sum())
        if not count:
            result[name] = dict(count=0); continue
        error = mean[mask]-actual[mask]
        result[name] = dict(count=count, mean_actual_margin_mm=float(actual[mask].mean()*1000),
            mean_predicted_margin_mm=float(mean[mask].mean()*1000),
            rmse_mm=float(np.sqrt(np.mean(error**2))*1000),
            mae_mm=float(np.abs(error).mean()*1000), mean_head_std_mm=float(std[mask].mean()*1000),
            predicted_unsafe_count=int((mean[mask] > 0.).sum()),
            both_heads_predict_safe_count=int((heads[:, mask] <= 0.).all(0).sum()),
            mean_plus_one_head_std_predicts_safe_count=int((mean[mask]+std[mask] <= 0.).sum()))
    # No invented probability interpretation for head disagreement.
    result['unsafe_actual']['false_safe_mean_count'] = int(((actual > 0.) & (mean <= 0.)).sum())
    return result


def diagnose(directory, output):
    if output.exists(): raise FileExistsError('Preserve the original fixed-window diagnosis')
    manifest = json.loads((directory/'run_config.json').read_text())
    receipt = json.loads((directory/'receipt.json').read_text()); config = manifest['configuration']
    validate_config(config)
    if (receipt['status'] != 'completed' or receipt['steps'] != 150000 or receipt['engineering_only']
            or receipt['arm'] != manifest['arm'] or receipt['source_commit'] != manifest['source_commit']
            or receipt['training_seed'] != 2028 or manifest['engineering_only']):
        raise ValueError('Use an original completed registered development run')
    torch.set_num_threads(1)
    checkpoint_path = directory/'full_150000.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if (checkpoint['steps'] != 150000 or checkpoint['configuration'] != config
            or checkpoint['arm'] != manifest['arm'] or checkpoint['source_commit'] != manifest['source_commit']):
        raise ValueError('Do not swap the final checkpoint for an earlier better-scoring node')
    arm = manifest['arm']
    learner = build_development(config, arm, 2028, 'cpu')
    learner.load_state_dict(checkpoint['learner']); del checkpoint
    records = terminal_records(directory)
    obs = torch.tensor([r['obs'] for r in records], dtype=torch.float32)
    actions = torch.tensor([r['action'] for r in records], dtype=torch.float32)
    actual = np.asarray([r['margin'] for r in records])
    if not np.isfinite(actual).all() or any((r['margin'] > 0.) != r['violation'] for r in records):
        raise ValueError('Actual signed margin/terminal violation convention changed')
    values = {}
    with torch.no_grad():
        for name, model in [('geometry_ema', learner.margin_geometry), ('online_margin', learner.margin_online)]:
            predictions = torch.cat([model(obs[i:i+1024], actions[i:i+1024])
                                     for i in range(0, len(obs), 1024)], dim=1).numpy()
            if not np.isfinite(predictions).all(): raise FloatingPointError('Nonfinite frozen margin output')
            values[name] = calibration_statistics(predictions, actual)
    result = dict(protocol='direction_v5_fixed_last10K_margin_diagnosis', arm=arm, training_seed=2028,
        checkpoint_steps=150000, source_commit=manifest['source_commit'], observations=len(records),
        actual_step_window=[140001, 150000], sample_selection='all_actual_controls_in_fixed_last10K_no_filtering',
        signed_margin='safety_distance_minus_actual_control_interval_minimum_distance;positive_is_unsafe',
        model_site='frozen_terminal_margin_on_executed_action_not_active_sampler_pullback_weight',
        scope='retrospective_training_distribution;possibly_in_sample;not_independent_confirmation',
        head_std_is_not_calibrated_uncertainty=True, results=values,
        checkpoint_sha256=digest_file(checkpoint_path), trace_sha256=digest_file(directory/'trajectories.jsonl.gz'),
        script_sha256=digest_file(Path(__file__)), run_receipt_sha256=digest_file(directory/'receipt.json'),
        model_loader_sha256=digest_file(Path(__file__).with_name('direction_development.py')),
        new_training_transitions=0, new_simulation_controls=0, generated_policy_actions=0,
        independent_test=False, physical_robot_interactions=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), observations=len(records), actual_unsafe=int((actual > 0.).sum()),
                          new_training_transitions=0, new_simulation_controls=0)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); diagnose(args.run, args.output)


if __name__ == '__main__': main()
