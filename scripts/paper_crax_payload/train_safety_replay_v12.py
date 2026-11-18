"""Fresh100K safety-batch-only4x screen; no priority actor/reward updates."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.paper_crax_payload.boundary_replay_v12 import SafetyOnlyBoundaryReplay
from scripts.paper_crax_payload.train_safereach_early_v12 import build as build_early, deployed, SafeReachTracker
from scripts.paper_crax_payload.train_direction_development import main as run

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_safereach_safety_replay.json'
FORMAT = 'safereach_safety_only_boundary4_v12_full'


class SafetyReplayMixin:
    def update(self, batch):
        ordinary = {k: v for k, v in batch.items() if k != 'safety_batch'}
        super().update(ordinary, safety_batch=batch['safety_batch'])

    def state_dict(self):
        return super().state_dict() | dict(safety_only_replay_v12=deepcopy(self.safety_replay_settings))

    def load_state_dict(self, state):
        if state.get('safety_only_replay_v12') != self.safety_replay_settings:
            raise ValueError('Only identical safety-only learner settings may resume')
        super().load_state_dict(state)


def build(config, arm, seed, device):
    learner = build_early(config, arm, seed, device)
    learner.__class__ = type('SafetyOnly'+type(learner).__name__, (SafetyReplayMixin, type(learner)), {})
    learner.safety_replay_settings = deepcopy(config['replay_priority'])
    return learner


def validate(config):
    if config != json.loads(PLAN.read_text()):
        raise ValueError('Use the exact registered safety-only replay protocol')


def replay_factory(config, arm, capacity, obs_dim):
    return SafetyOnlyBoundaryReplay(capacity, obs_dim)


if __name__ == '__main__':
    run(validator=validate, builder=build, format_name=FORMAT, evaluator=deployed,
        tracker_factory=SafeReachTracker, replay_factory=replay_factory)
