"""Registered single-factor replay screen, matching the fresh early-risk arms."""
import json
from pathlib import Path

from scripts.paper_crax_payload.boundary_replay_v12 import BoundaryReplay
from scripts.paper_crax_payload.train_safereach_early_v12 import build, deployed, SafeReachTracker
from scripts.paper_crax_payload.train_direction_development import main as run

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_safereach_boundary.json'
FORMAT = 'safereach_observed_boundary4_v12_full'


def validate(config):
    if config != json.loads(PLAN.read_text()) or config['replay_priority'] != BoundaryReplay.SETTINGS:
        raise ValueError('Use the exact registered observed-only4x replay screen')


def make_replay(config, arm, capacity, obs_dim):
    return BoundaryReplay(capacity, obs_dim)


if __name__ == '__main__':
    run(validator=validate, builder=build, format_name=FORMAT, evaluator=deployed,
        tracker_factory=SafeReachTracker, replay_factory=make_replay)
