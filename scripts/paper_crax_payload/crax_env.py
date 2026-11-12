"""Native CRAX vmap/episode API with explicit pre-reset transition outcomes.

Four-return Torch API retained, but info carries actual terminal_observation,
terminated/truncated and the NEXT policy observation separately. No host-copy
in the interaction path. Native dynamics, costs, goals and horizon are unchanged.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
from jax import numpy as jp
import numpy as np
import torch
from crax import envs
from crax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
from crax.io.torch import jax_to_torch


def _merge_reset(done, reset, terminal):
    def merge(new, old):
        if not hasattr(old, 'ndim') or old.ndim == 0:
            return old
        mask = done.reshape((done.shape[0],) + (1,) * (old.ndim - 1))
        return jp.where(mask, new, old)
    return jax.tree.map(merge, reset, terminal)


class PointGoalTorch:
    def __init__(self, num_envs=256, seed=41027, expected_device='cuda:0'):
        self.num_envs, self.seed = num_envs, seed
        self.base = envs.get_environment('safe_goal_point', level=1)
        self.native = self.base.unwrapped
        self.env = EpisodeWrapper(VmapWrapper(self.base), self.native.episode_length, 1)
        self.action_dim, self.obs_dim = self.base.action_size, self.base.observation_size
        self.device = torch.device(expected_device)
        self.key = jax.random.PRNGKey(seed)
        self._reset = jax.jit(self.env.reset)

        def step(state, action, key):
            terminal = self.env.step(state, action)
            done = terminal.done.astype(bool)
            truncated = terminal.info['truncation'].astype(bool)
            terminated = done & ~truncated
            distance_reward = (state.info['last_dist_goal'] - terminal.metrics['distance_to_goal']) * self.native._reward_distance
            goals = jp.rint((terminal.reward - distance_reward) / self.native._reward_goal)
            # Geometry-only label: risk-positive endpoint clearance proxy.
            # It never replaces ANY native cumulative cost component.
            position = terminal.pipeline_state.xpos[self.native._agent_body] if terminal.pipeline_state.xpos.ndim == 2 else terminal.pipeline_state.xpos[:, self.native._agent_body]
            delta = position[:, None, :2] - terminal.info['hazard_positions'][..., :2]
            radii = self.native._hazard_radii
            half = self.native._hazard_half_extents
            circles = jp.linalg.norm(delta, axis=-1) - radii
            box_q = jp.abs(delta) - half
            boxes = jp.linalg.norm(jp.maximum(box_q, 0), axis=-1) + jp.minimum(jp.max(box_q, axis=-1), 0)
            # Cube costs use a semantic radial boundary in this CRAX revision;
            # wall/rect geometric boundary is an AABB. Record this proxy scope.
            rect = jp.array([h.hazard_type == 'rect' for h in self.native._hazard_manager.hazards])
            circle_sizes = jp.array([float(h.size) if np.isscalar(h.size) else 0. for h in self.native._hazard_manager.hazards])
            circles = jp.linalg.norm(delta, axis=-1) - circle_sizes
            margin = -jp.min(jp.where(rect, boxes, circles), axis=-1)
            key, reset_key = jax.random.split(key)
            reset = lambda _: self._reset(jax.random.split(reset_key, self.num_envs))
            policy_state = jax.lax.cond(jp.any(done), lambda _: _merge_reset(done, reset(None), terminal), lambda _: terminal, operand=None)
            info = dict(cost=terminal.info['cost'], goals=goals, margin=margin,
                terminated=terminated, truncated=truncated,
                terminal_observation=terminal.obs)
            return policy_state, policy_state.obs, terminal.reward, done, info, key
        self._step = jax.jit(step)
        self.state = None

    def reset(self):
        self.key, key = jax.random.split(self.key)
        self.state = self._reset(jax.random.split(key, self.num_envs))
        obs = jax_to_torch(self.state.obs)
        if obs.device != self.device:
            raise RuntimeError(f'Cross-device copy forbidden: {obs.device} vs {self.device}')
        return obs

    def step(self, action):
        if self.state is None:
            raise RuntimeError('Call reset before step')
        # Modern DLPack object protocol negotiates stream synchronization. The
        # pinned legacy capsule helper cannot carry that stream handshake.
        jax_action = jax.dlpack.from_dlpack(action.detach().contiguous())
        self.state, obs, reward, done, info, self.key = self._step(self.state, jax_action, self.key)
        return jax_to_torch(obs), jax_to_torch(reward), jax_to_torch(done).bool(), jax_to_torch(info)

    def effective_config(self):
        n = self.native
        return dict(crax_version_commit='63504b540705be98024abd675e646ff5e046c6c2',
            env_id='safe_goal_point',difficulty=1,episode_length=n.episode_length,
            physics_timestep=float(n.sys.opt.timestep),frame_skip=n._n_frames,
            control_period=float(n.dt),action_dim=self.action_dim,obs_dim=self.obs_dim,
            action_limits=np.asarray(n.sys.actuator.ctrl_range).tolist(),
            goals=n._num_goals,hazards=n._num_hazards,
            hazard_types=[h.hazard_type for h in n._hazard_manager.hazards],
            hazard_collidable=[bool(h.collidable) for h in n._hazard_manager.hazards],
            reward_goal=n._reward_goal,reward_distance_scale=n._reward_distance,
            proximity_cost_scale=n._proximity_cost_scaler,collision_cost=n._collision_cost,
            native_goal_refresh=True,cost_termination_added=False,
            margin_label='negative_endpoint_clearance_proxy_metres;not_interval_minimum;not_native_cost',
            reset='fresh_native_reset_masked_in_device_after_preserving_terminal_observation')

    def state_dict(self):
        return dict(state=jax.device_get(self.state), key=jax.device_get(self.key),
            num_envs=self.num_envs,seed=self.seed,config=self.effective_config())

    def load_state_dict(self, state):
        if state['num_envs'] != self.num_envs or state['config'] != self.effective_config():
            raise ValueError('Environment dimensions/configuration changed during resume')
        self.state, self.key = jax.device_put(state['state']), jax.device_put(state['key'])
        return jax_to_torch(self.state.obs)


class DeviceReplay:
    def __init__(self, capacity, obs_dim, action_dim, device, seed):
        self.capacity, self.position, self.size = int(capacity), 0, 0
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.data = {k:torch.empty((self.capacity, *shape), dtype=torch.float32, device=self.device)
            for k,shape in dict(obs=(obs_dim,),action=(action_dim,),reward=(),cost=(),
                next_obs=(obs_dim,),terminated=(),truncated=(),margin=()).items()}

    def add(self, **batch):
        n = batch['obs'].shape[0]
        if n > self.capacity:
            raise ValueError('Vector block larger than replay capacity')
        indices = (torch.arange(n, device=self.device) + self.position) % self.capacity
        for key, value in batch.items():
            self.data[key][indices] = value.detach()
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size):
        indices = torch.randint(self.size, (batch_size,), device=self.device, generator=self.generator)
        return {k:value[indices] for k,value in self.data.items()}

    def state_dict(self):
        # Only initialized entries are serialized, not arbitrary unused VRAM.
        return dict(capacity=self.capacity, position=self.position,size=self.size,
            data={k:v[:self.size].detach().cpu() for k,v in self.data.items()},
            rng=self.generator.get_state())

    def load_state_dict(self, state):
        if state['capacity'] != self.capacity:
            raise ValueError('Replay capacity changed')
        self.position, self.size = state['position'], state['size']
        for key,value in state['data'].items():
            self.data[key][:self.size].copy_(value)
        self.generator.set_state(state['rng'].cpu())
