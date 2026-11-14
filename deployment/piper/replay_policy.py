"""Run a native frozen bundle against recorded observations; never transmit.

Input JSONL starts with metadata declaring Piper/robot_base observations, then
rows with observation[34] and timestamp_s. Replay uses the recorded clock plus
measured inference latency. Output actions are proposals, never executions.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from deployment.piper.bundle import FrozenPiperPolicy
from deployment.piper.offline import OfflineController, OfflineLimits, Recorder, replay_offline


def read_observations(path):
    with Path(path).open() as stream:
        header = json.loads(next(stream))
        if (header.get('record_type') != 'metadata' or header.get('robot_family') != 'Piper'
                or header.get('frame') != 'robot_base' or header.get('observation_dimension') != 34):
            raise ValueError('Require an explicit Piper34/base-frame observation stream')
        rows = []
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            observation = np.asarray(row['observation'], dtype=np.float32)
            stamp = float(row['timestamp_s'])
            if observation.shape != (34,) or not np.isfinite(observation).all() or not np.isfinite(stamp):
                raise ValueError('Invalid recorded observation/timestamp')
            rows.append((stamp, observation))
    if not rows:
        raise ValueError('No recorded observations')
    return header, rows


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def latency_summary(times, period_s=None):
    times = np.asarray(times,dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError('Require finite nonnegative action latencies')
    if period_s is not None and (not np.isfinite(period_s) or period_s <= 0):
        raise ValueError('A supplied diagnostic deadline must be positive')
    return dict(p50_s=float(np.quantile(times,.5)),p95_s=float(np.quantile(times,.95)),
        p99_s=float(np.quantile(times,.99)),mean_s=float(times.mean()),max_s=float(times.max()),
        raw_action_latencies_s=times.tolist(),diagnostic_period_s=period_s,
        diagnostic_deadline_misses=int((times>period_s).sum()) if period_s is not None else None,
        diagnostic_deadline_miss_rate=float((times>period_s).mean()) if period_s is not None else None,
        period_physically_calibrated=False)


def benchmark(policy, observations, warmup=20, samples=200, period_s=None):
    """Batch1 wall time includes normalization, geometry/tail/VJPs and D2H."""
    if warmup < 0 or samples < 1:
        raise ValueError('Invalid bounded benchmark size')
    if period_s is not None and (not np.isfinite(period_s) or period_s <= 0):
        raise ValueError('A supplied diagnostic deadline must be positive')
    times = []
    for index in range(warmup+samples):
        obs = observations[index % len(observations)][1][None]
        synchronize(policy.device)
        start = time.perf_counter()
        policy.action(obs)
        synchronize(policy.device)
        if index >= warmup:
            times.append(time.perf_counter()-start)
    count = dict(velocity_network_forwards=0, margin_ensemble_forwards=0)
    def velocity_hook(*_):
        count['velocity_network_forwards'] += 1
    def margin_hook(*_):
        count['margin_ensemble_forwards'] += 1
    hooks = [policy.actor.net.register_forward_hook(velocity_hook)]
    if policy.geometry is not None:
        hooks.append(policy.geometry.margin.register_forward_hook(margin_hook))
    try:
        policy.action(observations[0][1][None])
    finally:
        for hook in hooks:
            hook.remove()
    # EndpointGeometry differentiates each ensemble head once per active
    # geometry query. Counted forwards distinguish this from just quoting K.
    count['per_head_input_VJPs'] = 2*count['margin_ensemble_forwards']
    return dict(host=platform.node(), platform=platform.platform(), device=str(policy.device),
        torch_version=str(torch.__version__), threads=torch.get_num_threads(),
        hardware_label='measurement host only; not certified deployment-machine latency',
        batch_size=1, warmup=warmup, measured_actions=samples,
        **latency_summary(times,period_s),
        includes='normalization, base integration, geometry/tail/input gradients, host return',
        excludes='bundle load, physical observation acquisition, transport, actuation',
        base_integration_steps=policy.actor.num_steps, actual_forward_counts=count,
        parameter_updates=0, physical_commands_sent=0)


def replay_bundle(bundle, observations, output, device='cpu'):
    header, rows = read_observations(observations)
    policy = FrozenPiperPolicy(bundle, device)
    limits = OfflineLimits(action_scale_m=policy.manifest['action_scale_m'])
    controller = OfflineController(limits)
    recorder = Recorder(output, dict(robot_family='Piper', source=str(observations),
        source_metadata=header, bundle_manifest=policy.manifest, limits=asdict(limits),
        replay_clock='recorded timestamp plus measured inference wall time',
        physical_motion_enabled=False, real_interactions=0))
    try:
        for stamp, obs in rows:
            synchronize(device)
            start = time.perf_counter()
            raw = policy.action(obs[None])[0]
            synchronize(device)
            latency = time.perf_counter()-start
            proposal = controller.propose(raw, obs[12:15], stamp,
                now_s=stamp+latency, inference_latency_s=latency)
            recorder.append(proposal, metrics=dict(
                source_observation=obs.tolist(), source_is_prerecorded=True,
                new_simulation_step_executed=False))
    finally:
        recorder.close()
    return replay_offline(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--period-s',type=float,
        help='Optional offline diagnostic deadline, never a claim of calibrated control period')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.benchmark:
        if args.output.exists():
            raise FileExistsError('Preserve an existing timing result')
        _, observations = read_observations(args.observations)
        policy = FrozenPiperPolicy(args.bundle, args.device)
        result = benchmark(policy, observations, period_s=args.period_s)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    else:
        result = replay_bundle(args.bundle, args.observations, args.output, args.device)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
