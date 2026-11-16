"""One-shot v5 confirmation: freeze ALL nine final identities before new layouts.

Preparation only until a real, justified development selection is registered.
Never retries, selects checkpoints, retunes methods, or trains. Test episodes
are paired across policies/conditions: 2700 episodes, NOT 2700 unique layouts.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import torch

from scripts.paper_crax_payload.train_direction_confirmation import (
    FORMAT, build_confirmation, validate_confirmation)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def save_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def inspect_policies(config, directories):
    expected = {(a['name'], s) for a in config['arms'] for s in config['training_seeds']}
    if len(directories) != 9 or len(expected) != 9:
        raise ValueError('Require all nine registered policies, not the first available subset')
    policies = []; seen = set(); sources = set(); paired_initialization = {}
    for directory in directories:
        directory = directory.resolve()
        manifest_path = directory/'run_config.json'; receipt_path = directory/'receipt.json'
        manifest = json.loads(manifest_path.read_text()); receipt = json.loads(receipt_path.read_text())
        arm = manifest['arm']; seed = manifest['training_seed']; key = (arm['name'], seed)
        if (key not in expected or key in seen or manifest['configuration'] != config
                or arm != next(a for a in config['arms'] if a['name'] == arm['name'])
                or manifest['format'] != FORMAT or manifest['engineering_only']
                or manifest['target_transitions'] != 300000
                or receipt['status'] != 'completed' or receipt['engineering_only']
                or receipt['steps'] != 300000 or receipt['new_training_transitions'] != 300000
                or receipt['training_seed'] != seed or receipt['arm'] != arm
                or receipt['source_commit'] != manifest['source_commit']
                or receipt['evaluations'] != config['evaluation']['milestones']
                or receipt['independent_endpoint_test_executed']):
            raise ValueError('Require matching original completed fixed300K confirmation runs')
        initial = manifest['initial_network_sha256']
        if seed in paired_initialization and paired_initialization[seed] != initial:
            raise ValueError('Paired methods must retain the same per-seed initialization')
        paired_initialization[seed] = initial
        checkpoint_path = directory/'full_300000.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if (checkpoint['format'] != FORMAT or checkpoint['configuration'] != config
                or checkpoint['arm'] != arm or checkpoint['seed'] != seed
                or checkpoint['steps'] != 300000 or checkpoint['source_commit'] != manifest['source_commit']):
            raise ValueError('Never substitute an earlier or differently configured checkpoint')
        # Validate all actor/geometry/model settings before any test environment.
        learner = build_confirmation(config, arm, seed, 'cpu')
        learner.load_state_dict(checkpoint['learner']); del learner, checkpoint
        policies.append(dict(arm=arm, training_seed=seed, checkpoint_steps=300000,
            checkpoint=str(checkpoint_path), full_checkpoint_sha256=sha256(checkpoint_path),
            run_manifest_sha256=sha256(manifest_path), run_receipt_sha256=sha256(receipt_path),
            source_commit=manifest['source_commit'], initial_network_sha256=initial))
        sources.add(manifest['source_commit']); seen.add(key)
    if seen != expected or len(sources) != 1:
        raise ValueError('Freeze all nine complete policies from the same scientific source')
    return sorted(policies, key=lambda p: (p['arm']['name'], p['training_seed']))


def open_seed_plan(path, expected_digest):
    # Called ONLY after the all-policy registry has been persisted.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError('Fresh seed plan must match the pretraining byte commitment')
    plan = json.loads(raw)
    if (set(plan) != {'layout_seed_start', 'action_seed', 'fresh_layouts_not_previously_evaluated'}
            or plan['fresh_layouts_not_previously_evaluated'] is not True
            or type(plan['layout_seed_start']) is not int or type(plan['action_seed']) is not int
            or not 100000000 <= plan['layout_seed_start'] <= 2**31-101
            or not 0 <= plan['action_seed'] <= 2**31-20101):
        raise ValueError('Use a declared fresh reserved layout range, disjoint from old registered ranges')
    # Range check prevents known registered reuse; freshness still requires an
    # honest pretraining provenance review, not this declaration alone.
    return plan


def close(config, directories, seed_path, output):
    validate_confirmation(config)
    if output.exists(): raise FileExistsError('Preserve every admitted closure; no automatic retry')
    torch.set_num_threads(1)
    policies = inspect_policies(config, directories)
    output.mkdir(parents=True)
    registry_path = output/'all_final_policies_frozen.json'
    save_new(registry_path, dict(protocol='direction_confirmation_v5_all_final_policies',
        configuration=config, policies=policies, script_sha256=sha256(Path(__file__)),
        independent_test_opened=False, no_checkpoint_reselection=True))
    registry_sha = sha256(registry_path)
    try:
        plan = open_seed_plan(seed_path, config['independent_closure']['fresh_layout_seed_commitment_sha256'])
        save_new(output/'opened_seed_plan.json', plan)
        # Importing the simulator occurs after all nine policies are frozen.
        from scripts.paper_crax_payload.train_component_study import evaluate
        evaluation_config = deepcopy(config)
        evaluation_config['evaluation'].update(independent_episodes_each=100,
            independent_layout_seed_start=plan['layout_seed_start'], action_seed=plan['action_seed'])
        controls = episodes = 0
        for policy in policies:
            if sha256(registry_path) != registry_sha:
                raise ValueError('All-policy freeze changed after test opening')
            checkpoint_path = Path(policy['checkpoint'])
            if sha256(checkpoint_path) != policy['full_checkpoint_sha256']:
                raise ValueError('Final policy checkpoint changed after all-policy freeze')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            learner = build_confirmation(config, policy['arm'], policy['training_seed'], 'cpu')
            learner.load_state_dict(checkpoint['learner']); del checkpoint
            result = evaluate(learner, evaluation_config, 300000); del learner
            result.update(arm=policy['arm'], training_seed=policy['training_seed'],
                all_final_policies_freeze_sha256=registry_sha,
                full_checkpoint_sha256=policy['full_checkpoint_sha256'],
                independent_of_method_selection=True, paired_layouts_across_policies_and_conditions=True)
            save_new(output/f"independent-{policy['arm']['name']}-seed{policy['training_seed']}.json", result)
            controls += result['native_controls']; episodes += result['episodes']
        if episodes != 2700 or sha256(registry_path) != registry_sha:
            raise ValueError('Require all2700 episodes with unchanged all-policy freeze')
        save_new(output/'receipt.json', dict(status='completed', policies=9, independent_test_episodes=episodes,
            unique_layout_seeds=100, paired_layouts_across_policies_and_conditions=True,
            native_controls=controls, new_training_transitions=0, physical_robot_interactions=0,
            all_final_policies_freeze_sha256=registry_sha, checkpoint_reselection=False,
            automatic_retry=False, seed_plan_sha256=sha256(seed_path)))
    except BaseException as error:
        save_new(output/'interruption.json', dict(status='interrupted_no_automatic_retry',
            error_type=type(error).__name__, error=str(error), preserve_all_partial_results=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs=9, required=True)
    parser.add_argument('--seed-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    close(json.loads(args.config.read_text()), args.runs, args.seed_plan, args.output)


if __name__ == '__main__':
    raise SystemExit('Original300K confirmation closure superseded by the v6 budget reallocation; '
                     'do not open independent tests before a new250K selection/freeze.')
