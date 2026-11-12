"""SI/base-frame controller contract, fail-closed dry-run and record/replay.

There is deliberately no SDK import, CAN socket, physical connection, or motion
enable implementation in this module. Offline limits do not certify hardware.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Sequence


def vector3(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Expected three finite SI/base-frame values')
    return tuple(float(x) for x in values)


@dataclass(frozen=True)
class OfflineLimits:
    """Explicit engineering fixture, NOT manufacturer-approved robot limits."""
    period_s: float = 0.04
    timeout_s: float = 0.12
    action_scale_m: float = 0.01
    max_speed_m_s: float = 0.15
    workspace_min_m: tuple = (0.05, -0.45, 0.05)
    workspace_max_m: tuple = (0.65, 0.45, 0.65)

    def __post_init__(self):
        scalar = [self.period_s, self.timeout_s, self.action_scale_m, self.max_speed_m_s]
        if not all(math.isfinite(x) and x > 0 for x in scalar):
            raise ValueError('Control, timeout, action and velocity scales must be positive')
        lo, hi = vector3(self.workspace_min_m), vector3(self.workspace_max_m)
        if any(a >= b for a, b in zip(lo, hi)):
            raise ValueError('Invalid workspace')


def si_pose_to_sdk_units(position_m, euler_rad):
    """Pure arithmetic only, based on pinned SDK units; never transmits."""
    xyz, rpy = vector3(position_m), vector3(euler_rad)
    return tuple(round(x * 1_000_000) for x in xyz) + tuple(
        round(x * 180_000 / math.pi) for x in rpy)


class PhysicalTransportDisabled(RuntimeError):
    pass


def connect_physical_robot(*args, **kwargs):
    raise PhysicalTransportDisabled(
        'This preparation package has no physical transport. Operator confirmation '
        'and a separately reviewed implementation are required; no CAN/SDK was opened.')


class OfflineController:
    def __init__(self, limits: OfflineLimits | None = None):
        self.limits = limits or OfflineLimits()
        self.last_observation_timestamp = None
        self.sequence = 0

    def propose(self, raw_action, tcp_base_m, observation_timestamp_s, *, now_s=None,
                inference_latency_s=0.0):
        """Return an auditable proposal; execution is always absent here.

        Caller must separately record the measured executed action when a
        simulation/authorized transport actually advances. Rejected proposals
        must not be written to replay as executed actions.
        """
        now = time.monotonic() if now_s is None else float(now_s)
        action, position = vector3(raw_action), vector3(tcp_base_m)
        stamp, latency = float(observation_timestamp_s), float(inference_latency_s)
        if not all(math.isfinite(x) for x in [now, stamp, latency]) or latency < 0:
            raise ValueError('Invalid timestamp/latency')
        self.sequence += 1
        reasons = []
        if stamp > now or now - stamp > self.limits.timeout_s:
            reasons.append('observation_timeout_or_future_timestamp')
        if self.last_observation_timestamp is not None and stamp <= self.last_observation_timestamp:
            reasons.append('nonmonotone_observation')
        if latency > self.limits.period_s:
            reasons.append('inference_deadline_miss')
        lo, hi = self.limits.workspace_min_m, self.limits.workspace_max_m
        if any(x < a or x > b for x, a, b in zip(position, lo, hi)):
            reasons.append('observed_tcp_outside_workspace')
        clipped = [max(-1.0, min(1.0, x)) for x in action]
        delta = [x * self.limits.action_scale_m for x in clipped]
        norm = math.sqrt(sum(x*x for x in delta))
        scale = min(1., self.limits.max_speed_m_s*self.limits.period_s/max(norm, 1e-30))
        limited = [x * scale for x in delta]
        target = [max(a, min(b, x+d)) for x, d, a, b in zip(position, limited, lo, hi)]
        proposed = [x-p for x, p in zip(target, position)]
        modification = math.sqrt(sum((d-a*self.limits.action_scale_m)**2
                                     for d, a in zip(proposed, action)))
        accepted = not reasons
        self.last_observation_timestamp = max(stamp, self.last_observation_timestamp or stamp)
        return {
            'schema_version': 1, 'sequence': self.sequence, 'mode': 'offline_dry_run',
            'frame': 'robot_base', 'position_unit': 'm', 'time_unit': 's',
            'observation_timestamp_s': stamp, 'proposal_timestamp_s': now,
            'raw_policy_action': list(action), 'proposed_delta_m': proposed,
            'command_delta_m': proposed if accepted else None,
            'executed_action': None, 'execution_confirmed': False,
            'accepted_offline_proposal': accepted,
            'protected_or_rejected': bool(reasons or modification > 1e-12),
            'action_modification_m': modification, 'rejection_reasons': reasons,
            'inference_latency_s': latency, 'deadline_miss': latency > self.limits.period_s,
            'physical_connection': False, 'physical_commands_sent': 0,
        }


class Recorder:
    def __init__(self, path: Path, metadata: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('x', encoding='utf-8')
        self.file.write(json.dumps({'record_type': 'metadata', **metadata})+'\n')
        self.file.flush()

    def append(self, proposal, *, measured_execution=None, metrics=None):
        if measured_execution is not None and not proposal['accepted_offline_proposal']:
            raise ValueError('Rejected action cannot be labelled executed')
        record = dict(proposal)
        record.update(record_type='control', measured_execution=measured_execution, metrics=metrics)
        self.file.write(json.dumps(record, allow_nan=False)+'\n')
        self.file.flush()

    def close(self):
        self.file.close()


def replay_offline(path: Path):
    """Read-only validation/replay statistics; does not send any action."""
    with Path(path).open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records or records[0].get('record_type') != 'metadata':
        raise ValueError('Missing metadata header')
    controls = [x for x in records if x.get('record_type') == 'control']
    if any(x.get('physical_commands_sent') != 0 for x in controls):
        raise ValueError('This reader only replays explicitly offline records')
    return dict(records=len(controls),
                accepted=sum(x['accepted_offline_proposal'] for x in controls),
                deadline_misses=sum(x['deadline_miss'] for x in controls),
                rejected=sum(not x['accepted_offline_proposal'] for x in controls),
                physical_commands_sent=0, real_interactions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('dry-run').add_argument('--output', type=Path, required=True)
    sub.add_parser('replay').add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'replay':
        print(json.dumps(replay_offline(args.input)))
        return
    controller = OfflineController()
    recorder = Recorder(args.output, dict(robot_family='Piper',
        fixture='synthetic observations, NOT trained policy or measured robot',
        limits=asdict(controller.limits), real_interactions=0))
    try:
        for now, stamp, action, latency in [(10., 10., [.2, 0., 0.], .002),
                                           (10.2, 10.04, [1., 0., 0.], .002),
                                           (10.24, 10.24, [0., 1., 0.], .08)]:
            recorder.append(controller.propose(action, [.3, 0., .3], stamp,
                                               now_s=now, inference_latency_s=latency))
    finally:
        recorder.close()
    print(json.dumps(replay_offline(args.output)))


if __name__ == '__main__':
    main()
