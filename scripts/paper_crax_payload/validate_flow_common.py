"""Frozen common validation for Flow PointGoal checkpoints.

Every Flow method/seed receives the same five saved checkpoints and validation
seeds. This loads policy-only state, performs no optimizer or dual updates, and
never writes back to the training run.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path


VALIDATION_SEEDS = (61027, 61028, 61029, 61030, 61031)
INDEPENDENT_TEST_SEEDS = (71027, 71028, 71029, 71030, 71031)
CHECKPOINT_STEPS = (2_000_128, 4_000_000, 6_000_128, 8_000_000, 10_000_128)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summarize(result):
    return {key: result[key] for key in (
        'seed', 'episodes', 'native_return', 'episode_cost', 'goals_per_episode',
        'at_least_one_goal', 'budget_compliant_fraction', 'budget_compliant_return',
        'budget_compliant_episodes', 'capped_first_goal_seconds',
        'conditional_first_goal_seconds', 'physical_controls', 'elapsed_seconds')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(VALIDATION_SEEDS))
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--split', choices=['validation', 'independent_test'], default='validation')
    parser.add_argument('--selected-checkpoint', type=int)
    parser.add_argument('--selection-freeze', type=Path)
    args = parser.parse_args()
    expected_seeds = VALIDATION_SEEDS if args.split == 'validation' else INDEPENDENT_TEST_SEEDS
    if tuple(args.seeds) != expected_seeds:
        raise ValueError('Use exactly the predeclared seed family for the selected split')
    if args.split == 'validation' and args.episodes != 50:
        raise ValueError('Common validation requires 50 episodes per validation seed')
    if args.split == 'independent_test' and (args.episodes != 100 or args.selected_checkpoint is None
                                             or args.selection_freeze is None or len(args.runs) != 1):
        raise ValueError('Independent test requires one exact candidate, 100 episodes, and a selection freeze')
    args.output.mkdir(parents=True, exist_ok=True)

    import torch
    from scripts.paper_crax_payload.configuration import load_configuration
    from scripts.paper_crax_payload.crax_env import PointGoalTorch
    from scripts.paper_crax_payload.crax_models import CumulativeFlowLearner
    from scripts.paper_crax_payload.observation_normalizer import ObservationNormalizer
    from scripts.paper_crax_payload.train_crax import evaluate

    master = load_configuration(args.config)
    cfg = master['pointgoal']
    profile = cfg['profiles']['balanced']
    shared = cfg['shared_learner']
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Flow common validation must use an explicit GPU')
    torch.cuda.set_device(device)
    rows = []
    planned = 0
    started = time.monotonic()
    selection = json.loads(args.selection_freeze.read_text()) if args.selection_freeze else None
    if selection is not None and (selection.get('status') != 'complete_frozen_selection'
                                  or selection.get('independent_test_used') is not False):
        raise ValueError('Independent test requires an untouched complete selection freeze')
    for run in args.runs:
        if not run.is_dir():
            raise FileNotFoundError(run)
        for checkpoint in CHECKPOINT_STEPS:
            if args.selected_checkpoint is not None and checkpoint != args.selected_checkpoint:
                continue
            policy_path = run / f'policy_{checkpoint}.pt'
            if not policy_path.is_file():
                raise FileNotFoundError(policy_path)
            policy = torch.load(policy_path, map_location=device, weights_only=False)
            method = policy['method']
            seed = int(policy['seed'])
            if int(policy['steps']) != checkpoint or method not in ('stable_iso', 'ea'):
                raise ValueError('Flow policy/checkpoint provenance mismatch')
            env = PointGoalTorch(profile['E'], seed, str(device))
            normalizer = ObservationNormalizer(env.obs_dim, device, cfg['observation_normalization'])
            normalizer.load_state_dict(policy['normalizer'])
            learner = CumulativeFlowLearner(env.obs_dim, env.action_dim, method, str(device),
                width=shared['hidden_width'], actor_lr=shared['actor_lr'],
                critic_lr=shared['critic_lr'], variance=shared['reference_variance'],
                rho=shared['normal_variance_ratio'], cost_budget=cfg['episodic_cost_budget'],
                discount=cfg['discount'], critic_only_updates=shared['critic_only_updates'],
                flow_steps=shared['flow_steps'], dual_rule=shared.get('dual_rule', 'episode_mean'),
                episode_length=env.native.episode_length,
                lagrangian_lr=shared.get('lagrangian_lr', .01))
            learner.load_state_dict(policy['learner'])
            for module_name in ('actor', 'target_actor', 'q', 'qt', 'cost_q', 'cost_qt', 'margin', 'margin_geometry'):
                getattr(learner, module_name).eval().requires_grad_(False)
            learner.log_alpha.requires_grad_(False)
            cache = {}
            policy_hash = digest(policy_path)
            if selection is not None:
                method_record = selection['methods'][method]['selected']
                if (method_record['run'] != run.name or method_record['checkpoint_steps'] != checkpoint
                        or method_record['policy_sha256'] != policy_hash):
                    raise ValueError('Independent test policy is not the exact frozen selected Flow candidate')
            for validation_seed in args.seeds:
                target = args.output / f'{run.name}__{checkpoint}__{validation_seed}.json'
                if target.exists():
                    raise FileExistsError(f'Preserve existing validation artifact: {target}')
                result = evaluate(learner, args.episodes, validation_seed, normalizer, cache)
                result.update(run=run.name, method=method, training_seed=seed,
                    checkpoint_steps=checkpoint, policy_sha256=policy_hash,
                    training_source_commit=policy['source_commit'], validation_seed=validation_seed,
                    frozen_actor=True, frozen_normalizer=True, optimizer_updates=0,
                    dual_updates=0, training_transitions=0,
                    role='independent_test' if args.split == 'independent_test' else 'development_common_validation_not_independent_test',
                    analysis_source_commit=args.source_commit)
                target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
                row = summarize(result) | {k: result[k] for k in (
                    'run', 'method', 'training_seed', 'checkpoint_steps', 'policy_sha256',
                    'training_source_commit', 'validation_seed', 'analysis_source_commit')}
                rows.append(row)
                planned += args.episodes * env.native.episode_length
                print(json.dumps(row, allow_nan=False), flush=True)
            del env, learner, normalizer
            torch.cuda.empty_cache()
    fields = list(rows[0])
    with (args.output / 'COMMON_VALIDATION_RESULTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    receipt = dict(status='completed', schema='pointgoal_flow_common_validation_v1',
        analysis_source_commit=args.source_commit, split=args.split, training_transitions=0,
        optimizer_updates=0, dual_updates=0, real_interactions=0,
        physical_controls=planned, result_rows=len(rows), runs=len(args.runs),
        checkpoints_per_run=len(CHECKPOINT_STEPS), seeds=list(args.seeds),
        episodes_per_seed=args.episodes, independent_test_executed=args.split == 'independent_test',
        original_checkpoints_modified=False, elapsed_seconds=time.monotonic()-started)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
