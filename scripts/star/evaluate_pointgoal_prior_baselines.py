#!/usr/bin/env python3
"""Evaluate PointGoal SACLag and official SHAPO-PIDLag final checkpoints.

The evaluator uses the same five base environment seeds and sequential 20-episode
streams as the frozen P4.7 final evaluation.  Each baseline keeps its native
stochastic policy, with a fixed Torch RNG stream per base seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import torch


ENVIRONMENT = 'SafetyPointGoal1-v0'
EXPECTED_SHAPO_COMMIT = '1bb770437bbbe40a89938661343744243e164bf9'
DEFAULT_BASE_SEEDS = (22_027, 22_028, 22_029, 22_030, 22_031)
DEFAULT_EPISODES_PER_BASE_SEED = 20
DEFAULT_POLICY_SEED_OFFSET = 1_000_003
SUPPORTED_ALGORITHMS = ('SACLag', 'SHAPO-PIDLag')


class RolloutBackend(Protocol):
    """Small common interface for the two independently implemented baselines."""

    auto_resets: bool

    def reset(self, seed: int | None) -> Any:
        """Reset the evaluation environment and return its observation."""

    def action(self, observation: Any) -> Any:
        """Sample one native stochastic policy action."""

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        """Advance the environment by one transition."""

    def close(self) -> None:
        """Close environment resources."""


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert undefined floating statistics to strict-JSON nulls."""
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write strict JSON through an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(
        f'{json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a homogeneous CSV through an atomic rename."""
    if not rows:
        raise ValueError('cannot write an empty episode table')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def scalar(value: Any) -> float:
    """Convert one scalar NumPy/Torch value to float."""
    tensor = torch.as_tensor(value).detach().cpu().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f'expected one scalar, received shape {tuple(tensor.shape)}')
    return float(tensor.item())


def truthy(value: Any) -> bool:
    """Convert scalar or vector environment flags to one boolean."""
    tensor = torch.as_tensor(value).detach().cpu()
    return bool(tensor.bool().any().item()) if tensor.numel() else False


def tail_mean(values: list[float], fraction: float = 0.2) -> float:
    """Return the empirical upper-tail mean used for cost CVaR-20."""
    if not values:
        raise ValueError('CVaR requires at least one episode')
    count = max(1, math.ceil(len(values) * fraction))
    return statistics.fmean(sorted(values, reverse=True)[:count])


def git_identity(source: Path, *, include_untracked: bool = True) -> tuple[str, str]:
    """Return the exact Git commit and status for one source tree."""
    commit = subprocess.run(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_command = ['git', '-C', str(source), 'status', '--porcelain']
    if not include_untracked:
        status_command.append('--untracked-files=no')
    status = subprocess.run(
        status_command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, status


def load_receipt(path: Path, expected_protocol_sha256: str | None) -> dict[str, Any]:
    """Load and validate one completed formal-training receipt."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    algorithm = payload.get('algorithm')
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f'{path} has unsupported algorithm {algorithm!r}')
    if payload.get('status') != 'complete' or payload.get('phase') != 'formal':
        raise ValueError(f'{path} is not a completed formal receipt')
    if payload.get('environment') != ENVIRONMENT:
        raise ValueError(f'{path} does not belong to {ENVIRONMENT}')
    if int(payload.get('steps', -1)) != 200_000:
        raise ValueError(f'{path} is not a 200k equal-budget checkpoint')
    if int(payload.get('seed', -1)) not in (2027, 2028, 2029):
        raise ValueError(f'{path} has an unexpected training seed')
    protocol_sha = str(payload.get('protocol_sha256', ''))
    if expected_protocol_sha256 is not None and protocol_sha != expected_protocol_sha256:
        raise ValueError(f'{path} has a different protocol identity')
    checkpoint = Path(payload['checkpoint']).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != payload.get('checkpoint_sha256'):
        raise ValueError(f'{path} checkpoint SHA-256 mismatch')
    if algorithm == 'SHAPO-PIDLag':
        normalizer = Path(payload['normalizer']).resolve()
        if not normalizer.is_file():
            raise FileNotFoundError(normalizer)
        if sha256_file(normalizer) != payload.get('normalizer_sha256'):
            raise ValueError(f'{path} observation-normalizer SHA-256 mismatch')
        if payload.get('official_source_commit') != EXPECTED_SHAPO_COMMIT:
            raise ValueError(f'{path} does not use the pinned official SHAPO source')
    payload['_receipt_path'] = str(path.resolve())
    payload['_receipt_sha256'] = sha256_file(path)
    return payload


class SACLagBackend:
    """OmniSafe SACLag actor and its native evaluation wrapper stack."""

    auto_resets = False

    def __init__(self, receipt: dict[str, Any]) -> None:
        import omnisafe  # pylint: disable=import-outside-toplevel

        checkpoint = Path(receipt['checkpoint']).resolve()
        save_dir = checkpoint.parent.parent
        evaluator = omnisafe.Evaluator(render_mode='rgb_array')
        evaluator.load_saved(str(save_dir), checkpoint.name, render_mode='rgb_array')
        if evaluator._cfgs.algo != 'SACLag':  # pylint: disable=protected-access
            raise ValueError('loaded OmniSafe checkpoint is not SACLag')
        if evaluator._cfgs.env_id != ENVIRONMENT:  # pylint: disable=protected-access
            raise ValueError('loaded OmniSafe checkpoint belongs to another environment')
        if evaluator._cfgs.algo_cfgs.obs_normalize:  # pylint: disable=protected-access
            raise ValueError('frozen SACLag baseline unexpectedly normalizes observations')
        if evaluator._actor is None or evaluator._env is None:  # pylint: disable=protected-access
            raise RuntimeError('OmniSafe evaluator did not reconstruct actor and environment')
        self._evaluator = evaluator
        self._actor = evaluator._actor  # pylint: disable=protected-access
        self._env = evaluator._env  # pylint: disable=protected-access
        self._actor.eval()

    def reset(self, seed: int | None) -> Any:
        observation, _ = self._env.reset(seed=seed)
        return observation

    def action(self, observation: Any) -> Any:
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            return self._actor.predict(obs, deterministic=False).reshape(-1)

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


class SHAPOPIDLagBackend:
    """Pinned official SHAPO-PIDLag policy with frozen training observation RMS."""

    auto_resets = True

    def __init__(self, receipt: dict[str, Any], official_source: Path) -> None:
        official_source = official_source.resolve()
        commit, status = git_identity(official_source)
        if commit != EXPECTED_SHAPO_COMMIT:
            raise ValueError(f'expected official SHAPO {EXPECTED_SHAPO_COMMIT}, observed {commit}')
        if status:
            raise ValueError('official SHAPO evaluation source must be clean')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(official_source))
        sys.path.insert(0, str(official_source / 'safepo' / 'single_agent'))

        from safepo.common.env import (  # pylint: disable=import-outside-toplevel
            make_sa_safetygym_env,
        )
        from safepo.common.model import ActorVCritic  # pylint: disable=import-outside-toplevel

        arguments = SimpleNamespace(**receipt['official_arguments'])
        env, observation_space, action_space = make_sa_safetygym_env(
            arguments,
            num_envs=1,
            env_id=ENVIRONMENT,
            seed=None,
        )
        policy = ActorVCritic(
            obs_dim=observation_space.shape[0],
            act_dim=action_space.shape[0],
            hidden_sizes=arguments.hidden_sizes,
            use_actor_layer_norm=arguments.use_actor_layer_norm,
            use_critic_layer_norm=arguments.use_critic_layer_norm,
        )
        state = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=False)
        policy.load_state_dict(state, strict=True)
        policy.eval()

        normalizer = torch.load(receipt['normalizer'], map_location='cpu', weights_only=False)
        running = env.obs_rms
        running.mean = torch.as_tensor(normalizer['mean']).cpu().numpy().copy()
        running.var = torch.as_tensor(normalizer['var']).cpu().numpy().copy()
        running.count = float(normalizer['count'])

        def keep_training_statistics(_observations: Any) -> None:
            return None

        running.update = keep_training_statistics
        self._env = env
        self._policy = policy

    def reset(self, seed: int | None) -> Any:
        observation, _ = self._env.reset(seed=seed)
        return observation

    def action(self, observation: Any) -> Any:
        obs = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            action, _, _, _ = self._policy.step(obs, deterministic=False)
        return action.detach().squeeze().cpu().numpy()

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


def terminal_info(info: dict[str, Any]) -> dict[str, Any]:
    """Return the original terminal info when an auto-reset wrapper replaced it."""
    final = info.get('final_info')
    return final if isinstance(final, dict) else info


def evaluate_backend(  # pylint: disable=too-many-locals
    backend: RolloutBackend,
    algorithm: str,
    training_seed: int,
    base_seeds: list[int],
    episodes_per_base_seed: int,
    policy_seed_offset: int,
    *,
    max_episode_steps: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one baseline under fixed environment and policy RNG streams."""
    if not base_seeds or len(base_seeds) != len(set(base_seeds)):
        raise ValueError('base seeds must be non-empty and unique')
    if episodes_per_base_seed <= 0 or max_episode_steps <= 0:
        raise ValueError('episode counts and horizon must be positive')
    records: list[dict[str, Any]] = []
    total_violation_steps = 0
    total_steps = 0
    with torch.random.fork_rng(devices=[]):
        for base_seed in base_seeds:
            torch.manual_seed(policy_seed_offset + base_seed)
            observation = backend.reset(base_seed)
            for episode_index in range(episodes_per_base_seed):
                episode_return = 0.0
                episode_cost = 0.0
                episode_goals = 0
                episode_violation_steps = 0
                action_norms: list[float] = []
                completed = False
                for step_index in range(max_episode_steps):
                    action = backend.action(observation)
                    action_norms.append(
                        float(torch.linalg.vector_norm(torch.as_tensor(action)).item()),
                    )
                    next_observation, reward, cost, terminated, truncated, info = backend.step(
                        action,
                    )
                    reward_value = scalar(reward)
                    cost_value = scalar(cost)
                    episode_return += reward_value
                    episode_cost += cost_value
                    episode_violation_steps += int(cost_value > 0.0)
                    original_info = terminal_info(info)
                    episode_goals += int(
                        truthy(original_info.get('goal_met', info.get('goal_met', False))),
                    )
                    total_steps += 1
                    total_violation_steps += int(cost_value > 0.0)
                    observation = next_observation
                    if truthy(terminated) or truthy(truncated):
                        completed = True
                        episode_length = step_index + 1
                        break
                if not completed:
                    raise RuntimeError(
                        f'{algorithm} seed {training_seed} exceeded the {max_episode_steps}-step horizon',
                    )
                records.append(
                    {
                        'algorithm': algorithm,
                        'training_seed': training_seed,
                        'base_seed': base_seed,
                        'episode_index': episode_index,
                        'return': episode_return,
                        'cost': episode_cost,
                        'length': episode_length,
                        'goal_met_count': episode_goals,
                        'success': int(episode_goals > 0),
                        'violation_steps': episode_violation_steps,
                        'mean_action_norm': statistics.fmean(action_norms),
                    },
                )
                if episode_index + 1 < episodes_per_base_seed and not backend.auto_resets:
                    observation = backend.reset(None)

    returns = [float(row['return']) for row in records]
    costs = [float(row['cost']) for row in records]
    successes = [int(row['success']) for row in records]
    lengths = [int(row['length']) for row in records]
    summary = {
        'episodes': len(records),
        'mean_return': statistics.fmean(returns),
        'return_std': statistics.stdev(returns) if len(returns) > 1 else 0.0,
        'success_rate': statistics.fmean(successes),
        'mean_cost': statistics.fmean(costs),
        'cost_std': statistics.stdev(costs) if len(costs) > 1 else 0.0,
        'cost_cvar_20': tail_mean(costs),
        'mean_episode_length': statistics.fmean(lengths),
        'violation_ratio': total_violation_steps / max(total_steps, 1),
        'mean_action_norm': statistics.fmean(
            float(row['mean_action_norm']) for row in records
        ),
    }
    return records, summary


def result_path(output_dir: Path, receipt: dict[str, Any]) -> Path:
    """Return one stable method/seed result path."""
    slug = str(receipt['algorithm']).lower().replace('-', '_')
    return output_dir / f'{slug}_seed_{receipt["seed"]}_step_{receipt["steps"]}.json'


def reusable_result(
    path: Path,
    receipt: dict[str, Any],
    base_seeds: list[int],
    episodes_per_base_seed: int,
    policy_seed_offset: int,
) -> dict[str, Any] | None:
    """Return an exact completed result, or refuse an incompatible existing artifact."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    exact = (
        payload.get('status') == 'complete'
        and payload.get('receipt_sha256') == receipt['_receipt_sha256']
        and payload.get('base_seeds') == base_seeds
        and payload.get('episodes_per_base_seed') == episodes_per_base_seed
        and payload.get('policy_seed_offset') == policy_seed_offset
    )
    if not exact:
        raise FileExistsError(f'refusing to overwrite incompatible evaluation {path}')
    return payload


def evaluate_receipt(  # pylint: disable=too-many-arguments
    receipt: dict[str, Any],
    official_source: Path,
    output_dir: Path,
    base_seeds: list[int],
    episodes_per_base_seed: int,
    policy_seed_offset: int,
    evaluation_source_commit: str,
) -> dict[str, Any]:
    """Evaluate one receipt or reuse its exact completed artifact."""
    output = result_path(output_dir, receipt)
    existing = reusable_result(
        output,
        receipt,
        base_seeds,
        episodes_per_base_seed,
        policy_seed_offset,
    )
    if existing is not None:
        return existing
    if receipt['algorithm'] == 'SACLag':
        backend: RolloutBackend = SACLagBackend(receipt)
        normalizer_policy = 'disabled_by_frozen_training_config'
    else:
        backend = SHAPOPIDLagBackend(receipt, official_source)
        normalizer_policy = 'frozen_training_running_mean_and_variance'
    try:
        records, summary = evaluate_backend(
            backend,
            str(receipt['algorithm']),
            int(receipt['seed']),
            base_seeds,
            episodes_per_base_seed,
            policy_seed_offset,
        )
    finally:
        backend.close()
    payload = {
        'status': 'complete',
        'algorithm': receipt['algorithm'],
        'environment': ENVIRONMENT,
        'training_seed': int(receipt['seed']),
        'training_steps': int(receipt['steps']),
        'training_cumulative_cost': float(receipt['training_cumulative_cost']),
        'checkpoint': receipt['checkpoint'],
        'checkpoint_sha256': receipt['checkpoint_sha256'],
        'receipt': receipt['_receipt_path'],
        'receipt_sha256': receipt['_receipt_sha256'],
        'training_source_commit': receipt['source_commit'],
        'evaluation_source_commit': evaluation_source_commit,
        'protocol_sha256': receipt['protocol_sha256'],
        'official_shapo_source_commit': receipt.get('official_source_commit'),
        'evaluation_mode': 'fixed-native-stochastic-policy-stream',
        'evaluation_role': 'primary',
        'environment_stream_policy': (
            'reset once with each base seed, then advance 20 sequential episode RNG streams'
        ),
        'policy_stream_policy': (
            'native stochastic action sampling from one CPU Torch RNG stream per base seed'
        ),
        'policy_seed_offset': policy_seed_offset,
        'base_seeds': base_seeds,
        'episodes_per_base_seed': episodes_per_base_seed,
        'normalizer_policy': normalizer_policy,
        'summary': summary,
        'episode_records': records,
    }
    atomic_json(output, payload)
    return payload


def parse_base_seeds(raw: str) -> list[int]:
    """Parse a comma-separated fixed seed list."""
    seeds = [int(value.strip()) for value in raw.split(',') if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('base seeds must be non-empty and unique')
    return seeds


def parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--receipts', nargs='+', type=Path, required=True)
    result.add_argument('--official-source', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument(
        '--base-seeds',
        default=','.join(str(seed) for seed in DEFAULT_BASE_SEEDS),
    )
    result.add_argument(
        '--episodes-per-base-seed',
        type=int,
        default=DEFAULT_EPISODES_PER_BASE_SEED,
    )
    result.add_argument('--policy-seed-offset', type=int, default=DEFAULT_POLICY_SEED_OFFSET)
    result.add_argument('--expected-protocol-sha256')
    return result


def main() -> None:
    """Evaluate every requested baseline and emit raw plus compact tables."""
    args = parser().parse_args()
    source_root = Path(__file__).resolve().parents[2]
    evaluation_source_commit, source_status = git_identity(source_root)
    if source_status:
        raise ValueError('formal evaluation requires a clean committed evaluator source')
    base_seeds = parse_base_seeds(args.base_seeds)
    if args.episodes_per_base_seed <= 0:
        raise ValueError('episodes-per-base-seed must be positive')
    receipts = [
        load_receipt(path.resolve(), args.expected_protocol_sha256) for path in args.receipts
    ]
    identities = [(payload['algorithm'], int(payload['seed'])) for payload in receipts]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate method/training-seed receipts')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        evaluate_receipt(
            receipt,
            args.official_source,
            args.output_dir,
            base_seeds,
            args.episodes_per_base_seed,
            args.policy_seed_offset,
            evaluation_source_commit,
        )
        for receipt in receipts
    ]
    episode_rows = [row for payload in results for row in payload['episode_records']]
    episode_rows.sort(
        key=lambda row: (
            row['algorithm'],
            row['training_seed'],
            row['base_seed'],
            row['episode_index'],
        ),
    )
    atomic_csv(args.output_dir / 'pointgoal_prior_episode_records.csv', episode_rows)
    manifest = {
        'status': 'complete',
        'evaluation_source_commit': evaluation_source_commit,
        'protocol_sha256': args.expected_protocol_sha256,
        'base_seeds': base_seeds,
        'episodes_per_base_seed': args.episodes_per_base_seed,
        'episodes_per_policy': len(base_seeds) * args.episodes_per_base_seed,
        'policy_seed_offset': args.policy_seed_offset,
        'results': [str(result_path(args.output_dir, receipt)) for receipt in receipts],
    }
    atomic_json(args.output_dir / 'evaluation_manifest.json', manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
