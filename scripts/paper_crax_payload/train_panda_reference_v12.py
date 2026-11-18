"""Matched native45D P5/Metric reference; no Piper weights or Panda tuning."""
import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.paper_crax_payload.component_models import ComponentActor
from scripts.paper_crax_payload.native_p5_adaptation import attach_adaptation, PandaAdaptationLearner
from scripts.paper_crax_payload.payload_environment import PayloadConfig, SafePayloadTransport
from scripts.paper_crax_payload.train_native_p5_adaptation import train
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import TrainingTracker
from scripts.star.arm_safe_entropy.stability_runtime import isolated_rng

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_panda_reference.json'
FORMAT = 'native_panda_explicit_metric_reference_v12_full'


class PandaReferenceLearner(PandaAdaptationLearner):
    def state_dict(self):
        return super().state_dict() | dict(panda_reference_v12=deepcopy(self.reference_identity))

    def load_state_dict(self, state):
        if state.get('panda_reference_v12') != self.reference_identity:
            raise ValueError('Retain the identical native Panda reference arm')
        super().load_state_dict(state)


def attach(learner, plan, line, arm):
    if plan['lines'][line]['robot'] != 'Panda' or arm not in plan['arms']:
        raise ValueError('Only native45D Panda is admitted here')
    base = deepcopy(plan); base['arms'] = ['p5', 'control']
    attach_adaptation(learner, base, line, 'p5' if arm == 'Panda-P5-fixed' else 'control')
    learner.__class__ = PandaReferenceLearner
    learner.reference_identity = dict(arm=arm, robot='Panda', obs_dim=45,
        metric_definition='isotropic_sampling_directional_energy_no_outward_loss')
    if arm == 'Panda-Metric-only':
        for actor in (learner.actor, learner.target_actor):
            actor.__class__ = ComponentActor
            actor.sampling_directional, actor.energy_directional = False, True
    return learner


class PandaTracker(TrainingTracker):
    def __init__(self, n):
        super().__init__(n, success_distance=.035)


def evaluate(learner, config, episodes):
    """Same stochastic target rule, actual centroid progress and executed scale."""
    groups = []
    before = (learner.transitions, learner.updates, learner.actor_updates, learner.safety_multiplier)
    with isolated_rng():
        for ci, condition in enumerate(config['evaluation']['conditions']):
            env = SafePayloadTransport(PayloadConfig(**(dict(config['task']) | condition.get('task_overrides', {}))))
            rows = []
            try:
                for episode in range(episodes):
                    seed = config['evaluation']['development_layout_seed_start'] + episode
                    torch.manual_seed(config['evaluation']['action_sampling_seed']+ci*1000+episode)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(config['evaluation']['action_sampling_seed']+ci*1000+episode)
                    obs, initial = env.reset(seed=seed)
                    distances = deque([float(np.linalg.norm(env.payload_position()-env.payload_goal))], maxlen=11)
                    trajectory = [env.payload_position().tolist()]; durations = []
                    total = 0.; minimum = initial['minimum_distance']; stalled = False
                    for t in range(env.config.horizon):
                        if learner.device.type == 'cuda': torch.cuda.synchronize(learner.device)
                        tick = time.perf_counter_ns(); action = learner.action(obs[None])[0]
                        if learner.device.type == 'cuda': torch.cuda.synchronize(learner.device)
                        durations.append(time.perf_counter_ns()-tick)
                        obs, reward, done, truncated, info = env.step(np.clip(action*condition.get('action_scale', 1.), -1., 1.))
                        total += reward; minimum = min(minimum, info['minimum_distance'])
                        trajectory.append(env.payload_position().tolist()); distances.append(float(info['distance_to_goal']))
                        stalled |= len(distances) == 11 and distances[0]-distances[-1] < .001
                        if done or truncated: break
                    rows.append(dict(seed=seed, episode_return=total, safe_success=bool(info['safe_success']),
                        violation=bool(info['violation']), collision=bool(info['collision']),
                        safe_timeout=bool(not info['safe_success'] and not info['violation']), stall=stalled,
                        ordinary_terminal_task_success=bool(info['distance_to_goal'] < env.config.success_distance),
                        length=t+1, minimum_distance_m=minimum, trajectory=trajectory,
                        time_to_goal_capped_seconds=((t+1) if info['safe_success'] else env.config.horizon)*env.control_dt,
                        inference_p50_ms=float(np.percentile(durations, 50))/1e6,
                        inference_p95_ms=float(np.percentile(durations, 95))/1e6))
                groups.append(dict(condition=condition['name'], episodes=episodes, records=rows,
                    safe_success_count=sum(r['safe_success'] for r in rows),
                    violation_count=sum(r['violation'] for r in rows),
                    safe_timeout_count=sum(r['safe_timeout'] for r in rows),
                    stall_count=sum(r['stall'] for r in rows)))
            finally:
                env.close()
    if before != (learner.transitions, learner.updates, learner.actor_updates, learner.safety_multiplier):
        raise ValueError('Frozen reference evaluation changed the learner')
    return dict(conditions=groups, physical_evaluation_transitions=sum(r['length'] for c in groups for r in c['records']),
        new_training_steps=0, physical_robot_interactions=0, independent_test_used=False,
        evaluation_split='new_development_only_not_Panda_guard_pass',
        inference_timing_scope='sharedGPU batch1 full action call;no physics;not hardware latency',
        stall_definition='any10control window with net centroid-to-goal progress less than1mm')


def validate(plan, args):
    if (plan != json.loads(PLAN.read_text()) or args.line != 'panda_payload'
            or args.arm not in plan['arms'] or args.resume):
        raise ValueError('Use the exact registered paired native Panda reference')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'parent', 'output', 'data-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', required=True); p.add_argument('--line', default='panda_payload')
    p.add_argument('--source-commit', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--cuda-memory-fraction', type=float, default=.04)
    args = p.parse_args(); args.resume = None; args.assets_receipt = None
    train(args, protocol_validator=validate, adaptation_attacher=attach, format_name=FORMAT,
          tracker_factory=PandaTracker, payload_evaluator=evaluate)
