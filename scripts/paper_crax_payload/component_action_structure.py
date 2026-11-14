"""Descriptive final-action covariance on a frozen training-derived state pool.

No environment import, simulator calls, parameter updates or candidate selection.
Common states are the first32 episode starts, before inspecting their outcomes.
Only within-state stochastic covariance is measured, not covariance across states.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def completed_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'run_config.json').read_text())
    receipt = json.loads((directory / 'receipt.json').read_text())
    if (receipt.get('status') != 'completed' or receipt.get('engineering_only')
            or receipt.get('steps') != 300000 or receipt.get('training_seed') != 2027
            or receipt.get('source_commit') != 'b31878c0'
            or manifest['arm'] != receipt['arm']
            or manifest['initial_network_sha256'] != receipt['initial_network_sha256']):
        raise ValueError('Require an original complete300K scientific component arm')
    return manifest, receipt


def write_new(output, value):
    with Path(output).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return value


def freeze_pool(directory, output):
    manifest, _ = completed_run(directory)
    if manifest['arm']['name'] != 'iso':
        raise ValueError('Pool source is fixed to the original Iso run')
    states = []
    with gzip.open(Path(directory) / 'trajectories.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['record_type'] != 'initial':
                continue
            states.append({key: row[key] for key in (
                'env', 'episode_id', 'transitions', 'observation', 'minimum_distance')})
            if len(states) == 32:
                break
    if len(states) != 32 or any(len(row['observation']) != 25 for row in states):
        raise ValueError('Require all32 recorded episode-start observations')
    return write_new(output, dict(
        protocol='component_final_action_structure_v1', states=states,
        source_arm='iso', source_commit=manifest['source_commit'],
        initial_network_sha256=manifest['initial_network_sha256'],
        config_sha256=manifest['config_sha256'],
        state_selection='first32_initial_records_in_native_Iso_training_trace; no outcome filtering',
        source_run_config_sha256=digest_file(Path(directory) / 'run_config.json'),
        independent_test=False, candidate_selection=False,
        draws_per_state=64, gaussian_seed=2026111406,
        new_training_transitions=0, new_simulation_controls=0))


def action_statistics(samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError('Require finite repeated final3D actions at one fixed state')
    covariance = np.cov(samples, rowvar=False, ddof=1)
    return dict(mean_action=samples.mean(axis=0).tolist(),
        sample_covariance=covariance.tolist(), covariance_trace=float(np.trace(covariance)),
        covariance_eigenvalues=np.linalg.eigvalsh(covariance).tolist(),
        final_action_boundary_fraction=float(np.mean(np.any(np.abs(samples) >= 1., axis=-1))))


def measure(directory, pool_path, output):
    import torch
    from scripts.paper_crax_payload.component_models import build_component

    if Path(output).exists():
        raise FileExistsError('Preserve previous diagnostic; no automatic resampling')
    manifest, receipt = completed_run(directory)
    pool = json.loads(Path(pool_path).read_text())
    if (pool['protocol'] != 'component_final_action_structure_v1'
            or pool['initial_network_sha256'] != manifest['initial_network_sha256']
            or pool['config_sha256'] != manifest['config_sha256']
            or len(pool['states']) != 32 or pool['draws_per_state'] != 64):
        raise ValueError('Use the same fixed32-state pool for all component arms')
    torch.set_num_threads(1)
    checkpoint_path = Path(directory) / 'full_300000.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if (checkpoint['steps'] != 300000 or checkpoint['arm'] != manifest['arm']
            or checkpoint['source_commit'] != manifest['source_commit']
            or checkpoint['configuration'] != manifest['configuration']):
        raise ValueError('Wrong fixed endpoint checkpoint')
    arm = manifest['arm']
    learner = build_component(manifest['configuration'], arm['component'], arm['rho'], 'cpu')
    learner.load_state_dict(checkpoint['learner'])
    learner.set_transitions(300000)
    # Discard replay/environment recovery state after loading the actual learner.
    del checkpoint
    learner.actor.eval()
    generator = torch.Generator(device='cpu').manual_seed(pool['gaussian_seed'])
    rows = []
    for index, state in enumerate(pool['states']):
        observations = torch.tensor(state['observation'], dtype=torch.float32).repeat(64, 1)
        noise = torch.randn((4, 64, 3), generator=generator)
        # Geometry explicitly re-enables input autograd; do not use inference_mode.
        with torch.no_grad():
            actions, energy, clips, reference = learner.actor(
                observations, geometry=learner.geometry,
                noise_increments=noise, return_reference=True)
        rows.append(dict(state_index=index, **action_statistics(actions.cpu().numpy()),
            mean_energy=float(energy.mean()), mean_internal_clipping=float(clips.mean()),
            mean_reference_vn=float(reference['vn'].mean()),
            mean_reference_vt=float(reference['vt'].mean())))
    return write_new(output, dict(
        protocol=pool['protocol'], arm=arm, steps=receipt['steps'],
        training_seed=2027, source_commit=receipt['source_commit'],
        analysis_script_sha256=digest_file(__file__),
        device='cpu', torch_version=str(torch.__version__), numpy_version=np.__version__,
        checkpoint_sha256=digest_file(checkpoint_path), pool_sha256=digest_file(pool_path),
        states=rows, draws_per_state=64, generated_actions=2048,
        mean_within_state_covariance_trace=float(np.mean([r['covariance_trace'] for r in rows])),
        scope='descriptive_final_action_distribution_on_training_derived_common_starts',
        risk_projection=False, true_environment_risk_measured=False,
        independent_confirmation=False, cross_training_seed_claim=False,
        new_training_transitions=0, new_simulation_controls=0, physical_robot_interactions=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze-pool', action='store_true')
    parser.add_argument('--pool', type=Path)
    args = parser.parse_args()
    if args.freeze_pool:
        freeze_pool(args.run, args.output)
    elif args.pool:
        measure(args.run, args.pool, args.output)
    else:
        parser.error('Choose --freeze-pool or --pool')


if __name__ == '__main__':
    main()
