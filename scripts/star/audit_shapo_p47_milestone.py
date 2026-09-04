#!/usr/bin/env python3
"""Strictly validate and compactly archive one preregistered SHAPO P4.7 milestone."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml
from scripts.star.materialize_shapo_p47_config import materialize
from scripts.star.validate_shapo_p47_run import (
    load_json,
    newest,
    sha256_file,
    validate_checkpoint_artifact,
    validate_checkpoint_training_costs,
    validate_evaluation_artifact,
    validate_trajectory_artifact,
)


SELECTION_RULE = (
    'feasible_first;maximum_mean_return;minimum_mean_cost;earliest_milestone;'
    'before_feasibility_minimum_mean_cost_then_maximum_mean_return'
)


def write_json(path: Path, payload: Any) -> None:
    """Write deterministic strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'{json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)}\n',
        encoding='utf-8',
    )


def copied_identity(source: Path, destination: Path, root: Path) -> dict[str, Any]:
    """Copy one compact artifact and return its verified portable identity."""
    expected_sha256 = sha256_file(source)
    expected_bytes = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size != expected_bytes or sha256_file(destination) != expected_sha256:
        raise ValueError(f'copied milestone artifact changed: {source}')
    return {
        'artifact': destination.relative_to(root).as_posix(),
        'source_artifact': str(source),
        'sha256': expected_sha256,
        'bytes': expected_bytes,
    }


def write_checksum_manifest(root: Path) -> None:
    """Seal every compact milestone file except the checksum file and completion marker."""
    excluded = {'checksums.sha256', 'milestone_audit_complete'}
    files = sorted(path for path in root.rglob('*') if path.is_file() and path.name not in excluded)
    lines = [f'{sha256_file(path)}  {path.relative_to(root).as_posix()}' for path in files]
    (root / 'checksums.sha256').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def validate_manifest_contract(
    path: Path,
    *,
    geometry_mode: str,
    geometry_scale: float,
    formal_budget: float,
) -> dict[str, Any]:
    """Require the frozen P4.7 geometry identity in the run manifest."""
    manifest = load_json(path)
    geometry = manifest['p4_5_unified_cost_geometry']
    if geometry['geometry_mode'] != geometry_mode:
        raise ValueError('milestone run manifest geometry mode changed')
    if not math.isclose(float(geometry['cost_normalization_scale']), geometry_scale):
        raise ValueError('milestone run manifest geometry scale changed')
    if not math.isclose(float(geometry['formal_cost_budget']), formal_budget):
        raise ValueError('milestone run manifest formal cost budget changed')
    p47 = manifest['p4_7_safety_aware_geometry']
    if p47.get('enabled') is not True or float(p47.get('high_risk_geometry_fraction')) != 0.2:
        raise ValueError('milestone run manifest does not identify frozen P4.7')
    return manifest


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Validate one locked milestone and atomically create compact scientific evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--env-id', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    matrix = yaml.safe_load(matrix_path.read_text(encoding='utf-8'))
    if args.env_id not in matrix.get('environments', {}):
        raise ValueError(f'environment {args.env_id!r} is not in the locked matrix')
    if args.seed not in [int(seed) for seed in matrix.get('training_seeds', [])]:
        raise ValueError(f'training seed {args.seed} is not in the locked matrix')
    environment = matrix['environments'][args.env_id]
    milestones = [int(step) for step in environment['milestones']]
    if args.milestone not in milestones:
        raise ValueError(f'milestone {args.milestone} is not locked for {args.env_id}')
    if not args.source_commit or len(args.source_commit) != 40:
        raise ValueError('formal source commit must be a full commit identity')

    container = args.container.resolve()
    source_identity_path = container / 'source_commit.txt'
    if source_identity_path.read_text(encoding='utf-8').strip() != args.source_commit:
        raise ValueError('milestone container source commit changed')
    config_path = container / 'resolved_formal_config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if config != materialize(matrix_path, args.env_id):
        raise ValueError('milestone resolved config differs from sealed materialization')
    metadata = config['shapo_four_env_metadata']
    geometry_mode = str(environment['geometry_mode'])
    geometry_scale = float(environment['cost_normalization_scale'])
    formal_budget = float(environment['cost_limit'])
    if metadata['engineering_smoke_only'] is not False:
        raise ValueError('milestone config is labelled as an engineering smoke')
    if metadata['geometry_mode'] != geometry_mode:
        raise ValueError('milestone config geometry mode changed')
    if float(metadata['cost_normalization_scale']) != geometry_scale:
        raise ValueError('milestone config geometry scale changed')
    if float(metadata['formal_cost_budget']) != formal_budget:
        raise ValueError('milestone config formal budget changed')

    base_seeds = [int(seed) for seed in matrix['evaluation']['base_seeds']]
    trajectory_seeds = [int(seed) for seed in matrix['evaluation']['trajectory_base_seeds']]
    episodes_per_seed = int(matrix['evaluation']['episodes_per_base_seed'])
    evaluation_path = newest(
        container,
        f'training/**/artifacts/evaluation_step_{args.milestone}.json',
    )
    evaluation = validate_evaluation_artifact(
        evaluation_path,
        environment=args.env_id,
        milestone=args.milestone,
        base_seeds=base_seeds,
        trajectory_seeds=trajectory_seeds,
        episodes_per_seed=episodes_per_seed,
        geometry_mode=geometry_mode,
    )
    trajectory_path = newest(
        container,
        f'training/**/artifacts/trajectory_step_{args.milestone}.csv',
    )
    trajectory = validate_trajectory_artifact(
        trajectory_path,
        args.milestone,
        trajectory_seeds,
    )

    manifest_path = newest(container, 'training/**/artifacts/run_manifest.json')
    validate_manifest_contract(
        manifest_path,
        geometry_mode=geometry_mode,
        geometry_scale=geometry_scale,
        formal_budget=formal_budget,
    )
    counterfactual_path = newest(
        container,
        f'training/**/artifacts/counterfactual_step_{args.milestone}.json',
    )
    counterfactual = load_json(counterfactual_path)
    if int(counterfactual.get('milestone', -1)) != args.milestone:
        raise ValueError('counterfactual milestone identity changed')
    selection_path = newest(container, 'training/**/artifacts/best_safe_return_selection.json')
    selection = load_json(selection_path)
    if selection.get('selection_rule') != SELECTION_RULE:
        raise ValueError('best-safe-return selection rule changed')
    selected_step = int(selection.get('selected', {}).get('milestone', -1))
    allowed_selection_steps = {step for step in milestones if step <= args.milestone}
    if selected_step not in allowed_selection_steps:
        raise ValueError('best-safe-return selection is not from an observed locked milestone')

    replay_capacity = int(matrix['training_invariants']['replay_size'])
    checkpoint_path = newest(
        container,
        f'training/**/checkpoints/milestone_{args.milestone}.pt',
    )
    allowed_resume_checkpoints = set(container.glob('training/**/checkpoints/latest_full.pt'))
    checkpoint = validate_checkpoint_artifact(
        checkpoint_path,
        label=f'milestone_{args.milestone}',
        environment=args.env_id,
        seed=args.seed,
        expected_steps={args.milestone},
        replay_required=False,
        replay_capacity=replay_capacity,
        expected_custom_config=config['custom_cfgs'],
        allowed_resume_checkpoints=allowed_resume_checkpoints,
    )
    validate_checkpoint_training_costs([evaluation], [checkpoint])

    compact_sources = {
        'evaluation': evaluation_path,
        'trajectory': trajectory_path,
        'counterfactual': counterfactual_path,
        'run_manifest': manifest_path,
        'best_safe_return_selection': selection_path,
        'warmup_status': newest(container, 'training/**/artifacts/warmup_status.json'),
        'evaluation_curve': newest(container, 'training/**/artifacts/evaluation.csv'),
        'resolved_config': config_path,
        'materialization_receipt': container / 'materialize.json',
        'source_commit': source_identity_path,
    }

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite milestone audit: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    try:
        compact_artifacts = {}
        for label, source in compact_sources.items():
            group = (
                'provenance'
                if label
                in {
                    'resolved_config',
                    'materialization_receipt',
                    'source_commit',
                }
                else 'artifacts'
            )
            destination = staging / group / source.name
            compact_artifacts[label] = copied_identity(source, destination, staging)
        audit = {
            'status': 'passed',
            'audit_version': 'shapo-p47-milestone-v1',
            'audit_script_sha256': sha256_file(Path(__file__).resolve()),
            'strict_evaluation_validator_sha256': sha256_file(
                Path(__file__).resolve().with_name('validate_shapo_p47_run.py'),
            ),
            'protocol_version': matrix['protocol_version'],
            'protocol_sha256': matrix['protocol_sha256'],
            'matrix_sha256': sha256_file(matrix_path),
            'frozen_method_canonical_sha256': matrix['frozen_method_canonical_sha256'],
            'source_commit': args.source_commit,
            'environment': args.env_id,
            'training_seed': args.seed,
            'training_seed_is_statistical_unit': True,
            'milestone': args.milestone,
            'formal_cost_budget': formal_budget,
            'geometry_mode': geometry_mode,
            'cost_normalization_scale': geometry_scale,
            'evaluation': {
                key: evaluation[key]
                for key in (
                    'episodes',
                    'mean_return',
                    'mean_cost',
                    'cost_cvar_20',
                    'training_cumulative_cost',
                    'success_rate',
                    'failure_count',
                    'failure_rate',
                    'time_limit_count',
                    'time_limit_rate',
                    'high_risk_vn_vt',
                    'high_risk_dangerous_outward_velocity',
                )
            },
            'trajectory': trajectory,
            'server_only_checkpoint': {
                **checkpoint,
                'server_only': True,
            },
            'best_safe_return_selection': selection,
            'compact_artifacts': compact_artifacts,
            'diagnostic_only': True,
            'used_for_checkpoint_or_configuration_selection': False,
        }
        write_json(staging / 'milestone_audit.json', audit)
        write_checksum_manifest(staging)
        (staging / 'milestone_audit_complete').touch()
        os.replace(staging, output)
    except Exception:
        print(f'milestone audit failed; staging retained at {staging}')
        raise
    print(
        json.dumps(
            {
                'status': 'passed',
                'environment': args.env_id,
                'training_seed': args.seed,
                'milestone': args.milestone,
                'output_dir': str(output),
            },
            indent=2,
            sort_keys=True,
        ),
    )


if __name__ == '__main__':
    main()
