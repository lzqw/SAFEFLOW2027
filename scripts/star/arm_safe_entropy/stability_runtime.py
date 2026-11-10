"""Small state/RNG helpers for native EA-STAB continuation, not fault orchestration."""
from contextlib import contextmanager
from copy import deepcopy
import random

import numpy as np
import torch


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


@contextmanager
def isolated_rng():
    state=capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def capture_environment(env):
    joints=list(range(7))+[9,10]
    states=[env.client.getJointState(env.body,joint) for joint in joints]
    return dict(joints=joints,positions=[s[0] for s in states],velocities=[s[1] for s in states],
        start=env.start.copy(),goal=env.goal.copy(),obstacle=env.obstacle_position.copy(),
        elapsed=env._elapsed,done=env._done,previous_distance=env._previous_distance,
        rng=deepcopy(env.np_random.bit_generator.state),observation=env._obs().copy())


def restore_environment(env,state):
    """Restore observed Markov state and episode/RNG, without advancing physics.

    The next real action replaces motor targets,as in the original controller.
    Solver warm-start caches are not claimed bitwise portable between clients.
    No partial episode is silently reset or omitted from training accounting.
    """
    env.start=np.asarray(state['start']).copy()
    env.goal=np.asarray(state['goal']).copy()
    env.obstacle_position=np.asarray(state['obstacle']).copy()
    env.sim.set_base_pose('target',env.goal,np.array([0.,0.,0.,1.]))
    env.sim.set_base_pose('obstacle',env.obstacle_position,np.array([0.,0.,0.,1.]))
    for joint,position,velocity in zip(state['joints'],state['positions'],state['velocities']):
        env.client.resetJointState(env.body,joint,float(position),targetVelocity=float(velocity))
    env.robot.control_joints(np.asarray(state['positions']))
    env.client.performCollisionDetection()
    env._elapsed,env._done=state['elapsed'],state['done']
    env._previous_distance=state['previous_distance']
    env.np_random.bit_generator.state=deepcopy(state['rng'])
    observation=env._obs()
    if not np.allclose(observation,state['observation'],rtol=0.,atol=2e-6):
        raise ValueError('Native observed-state reconstruction differs; do not silently start a new episode')
    return observation


def build_learner(config,variant,device):
    from scripts.star.arm_safe_entropy.stability_models import StableArmLearner
    shared=config['shared_learner']
    return StableArmLearner(variant=variant,device=device,actor_lr=shared['actor_lr'],
        critic_lr=shared['critic_lr'],critic_only_updates=shared['critic_only_updates'],
        margin_ema_tau=config['geometry']['ema_tau'],actor_interval=shared['actor_update_interval'],
        rho=shared['rho'])
