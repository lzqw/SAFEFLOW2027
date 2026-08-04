# Copyright 2026 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU ring replay with failure and conservative-boundary stratification."""

from __future__ import annotations

from typing import Any

import torch


NORMAL = 0
BOUNDARY = 1
FAILURE = 2
CATEGORY_COUNT = 3


def updates_for_transitions(
    utd_ratio: float,
    new_transitions: int,
    carry: float = 0.0,
) -> tuple[int, float]:
    """Convert newly collected transitions to a concurrency-independent update count."""
    if utd_ratio < 0 or new_transitions < 0 or not 0.0 <= carry < 1.0:
        raise ValueError('UTD, transition count, and fractional carry are invalid')
    accumulated = carry + utd_ratio * new_transitions
    updates = int(accumulated)
    return updates, accumulated - updates


class STARReplayBuffer:  # pylint: disable=too-many-instance-attributes
    """Preallocated CPU replay with O(1) category-pool maintenance."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        batch_size: int,
        num_envs: int = 1,
        failure_fraction: float = 0.2,
        boundary_fraction: float = 0.4,
        importance_beta: float = 0.4,
        boundary_score_threshold: float = 0.5,
        context_dim: int = 2,
        pin_memory: bool = True,
    ) -> None:
        """Allocate every transition field and category pool on CPU."""
        if min(obs_dim, act_dim, capacity, batch_size, num_envs, context_dim) <= 0:
            raise ValueError('replay dimensions and sizes must be positive')
        if capacity <= batch_size:
            raise ValueError('replay capacity must exceed the global batch size')
        if failure_fraction < 0 or boundary_fraction < 0:
            raise ValueError('stratum fractions cannot be negative')
        if failure_fraction + boundary_fraction > 1:
            raise ValueError('failure and boundary fractions cannot exceed one')
        if not 0 <= importance_beta <= 1:
            raise ValueError('importance beta must lie in [0, 1]')
        if not 0 <= boundary_score_threshold <= 1:
            raise ValueError('boundary score threshold must lie in [0, 1]')
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.capacity = capacity
        self.batch_size = batch_size
        self.num_envs = num_envs
        self.failure_fraction = failure_fraction
        self.boundary_fraction = boundary_fraction
        self.importance_beta = importance_beta
        self.boundary_score_threshold = boundary_score_threshold
        self.context_dim = context_dim
        self._pin_memory = bool(pin_memory and torch.cuda.is_available())
        self._ptr = 0
        self._size = 0

        self.data: dict[str, torch.Tensor] = {
            'obs': self._allocate((capacity, obs_dim)),
            'action': self._allocate((capacity, act_dim)),
            'reward': self._allocate((capacity,)),
            'cost': self._allocate((capacity,)),
            'terminated': self._allocate((capacity,)),
            'truncated': self._allocate((capacity,)),
            'reward_done': self._allocate((capacity,)),
            'cost_done': self._allocate((capacity,)),
            'next_obs': self._allocate((capacity, obs_dim)),
            'context': self._allocate((capacity, context_dim)),
            'next_context': self._allocate((capacity, context_dim)),
            'risk_threshold': self._allocate((capacity,)),
            'next_risk_threshold': self._allocate((capacity,)),
            'empirical_cost_to_go': self._allocate((capacity,), fill=float('nan')),
            'violation_label': self._allocate((capacity,), fill=float('nan')),
            'boundary_score': self._allocate((capacity,)),
            'is_boundary': self._allocate((capacity,), dtype=torch.bool),
            'is_failure': self._allocate((capacity,), dtype=torch.bool),
            'source_random': self._allocate((capacity,), dtype=torch.bool),
            'category': self._allocate((capacity,), dtype=torch.long, fill=-1),
            'generation': self._allocate((capacity,), dtype=torch.long),
        }
        self._pool_indices = torch.empty((CATEGORY_COUNT, capacity), dtype=torch.long)
        self._pool_positions = torch.full((CATEGORY_COUNT, capacity), -1, dtype=torch.long)
        self._pool_sizes = [0 for _ in range(CATEGORY_COUNT)]
        self._episode_slots: list[list[tuple[int, int]]] = [[] for _ in range(num_envs)]

    def _allocate(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        fill: float = 0.0,
    ) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype, pin_memory=self._pin_memory)
        return tensor.fill_(fill)

    @property
    def device(self) -> torch.device:
        """Return the required resident replay device."""
        return torch.device('cpu')

    @property
    def size(self) -> int:
        """Return the number of valid transitions."""
        return self._size

    @property
    def pointer(self) -> int:
        """Return the next ring slot."""
        return self._ptr

    @property
    def pool_sizes(self) -> dict[str, int]:
        """Return current normal, boundary, and failure pool sizes."""
        return {
            'normal': self._pool_sizes[NORMAL],
            'boundary': self._pool_sizes[BOUNDARY],
            'failure': self._pool_sizes[FAILURE],
        }

    def _remove_pool_slot(self, slot: int) -> None:
        category = int(self.data['category'][slot].item())
        if category < 0:
            return
        position = int(self._pool_positions[category, slot].item())
        last_position = self._pool_sizes[category] - 1
        last_slot = int(self._pool_indices[category, last_position].item())
        if position != last_position:
            self._pool_indices[category, position] = last_slot
            self._pool_positions[category, last_slot] = position
        self._pool_positions[category, slot] = -1
        self._pool_sizes[category] = last_position

    def _add_pool_slot(self, slot: int, category: int) -> None:
        position = self._pool_sizes[category]
        self._pool_indices[category, position] = slot
        self._pool_positions[category, slot] = position
        self._pool_sizes[category] += 1
        self.data['category'][slot] = category

    @staticmethod
    def _as_cpu_batch(value: torch.Tensor, expected: int) -> torch.Tensor:
        batch = value.detach().to('cpu')
        if batch.ndim == 0:
            batch = batch.unsqueeze(0)
        if batch.shape[0] != expected:
            raise ValueError(f'transition field has batch {batch.shape[0]}, expected {expected}')
        return batch

    def store_batch(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        next_obs: torch.Tensor,
        context: torch.Tensor,
        next_context: torch.Tensor,
        risk_threshold: torch.Tensor,
        next_risk_threshold: torch.Tensor,
        boundary_score: torch.Tensor,
        failure: torch.Tensor,
        source_random: torch.Tensor,
    ) -> None:
        """Store one transition per environment and finalize empirical episode labels."""
        values = {
            key: self._as_cpu_batch(value, self.num_envs)
            for key, value in {
                'obs': obs,
                'action': action,
                'reward': reward,
                'cost': cost,
                'terminated': terminated,
                'truncated': truncated,
                'next_obs': next_obs,
                'context': context,
                'next_context': next_context,
                'risk_threshold': risk_threshold,
                'next_risk_threshold': next_risk_threshold,
                'boundary_score': boundary_score,
                'is_failure': failure,
                'source_random': source_random,
            }.items()
        }
        for name in ('context', 'next_context'):
            if values[name].shape[1:] != (self.context_dim,):
                raise ValueError(
                    f'{name} must have shape [num_envs, {self.context_dim}], '
                    f'got {tuple(values[name].shape)}',
                )
        scores = values['boundary_score'].float()
        if not bool(torch.isfinite(scores).all()) or bool(((scores < 0) | (scores > 1)).any()):
            raise ValueError('boundary scores must be finite and lie in [0, 1]')
        for env_index in range(self.num_envs):
            slot = self._ptr
            if self._size == self.capacity:
                self._remove_pool_slot(slot)
            self.data['generation'][slot] += 1
            generation = int(self.data['generation'][slot].item())
            for key, value in values.items():
                self.data[key][slot].copy_(value[env_index])
            self.data['reward_done'][slot] = values['terminated'][env_index].float()
            self.data['cost_done'][slot] = torch.logical_or(
                values['terminated'][env_index].bool(),
                values['truncated'][env_index].bool(),
            ).float()
            self.data['empirical_cost_to_go'][slot] = float('nan')
            self.data['violation_label'][slot] = float('nan')
            is_failure = bool(values['is_failure'][env_index].item())
            is_boundary = (
                float(values['boundary_score'][env_index].item()) >= self.boundary_score_threshold
            )
            self.data['is_boundary'][slot] = is_boundary
            category = FAILURE if is_failure else BOUNDARY if is_boundary else NORMAL
            self._add_pool_slot(slot, category)
            self._episode_slots[env_index].append((slot, generation))

            episode_done = bool(values['terminated'][env_index].item()) or bool(
                values['truncated'][env_index].item(),
            )
            if episode_done:
                self._finalize_episode(env_index)
            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def _finalize_episode(self, env_index: int) -> None:
        future_cost = 0.0
        for slot, generation in reversed(self._episode_slots[env_index]):
            if int(self.data['generation'][slot].item()) != generation:
                continue
            future_cost += float(self.data['cost'][slot].item())
            self.data['empirical_cost_to_go'][slot] = future_cost
            threshold = float(self.data['risk_threshold'][slot].item())
            self.data['violation_label'][slot] = float(future_cost > threshold)
        self._episode_slots[env_index] = []

    def _desired_counts(self, batch_size: int) -> list[int]:
        failures = round(batch_size * self.failure_fraction)
        boundaries = round(batch_size * self.boundary_fraction)
        normal = batch_size - failures - boundaries
        return [normal, boundaries, failures]

    def sample_batch(  # pylint: disable=too-many-locals
        self,
        device: torch.device | str = 'cpu',
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a fixed global batch and transfer it non-blockingly to the learner device."""
        if self._size == 0:
            raise RuntimeError('cannot sample an empty replay buffer')
        requested = self.batch_size if batch_size is None else batch_size
        if requested <= 0:
            raise ValueError('sample batch size must be positive')
        sampled: list[torch.Tensor] = []
        fallback = 0
        for category, count in enumerate(self._desired_counts(requested)):
            pool_size = self._pool_sizes[category]
            if count == 0:
                continue
            if pool_size == 0:
                fallback += count
                continue
            positions = torch.randint(pool_size, (count,))
            sampled.append(self._pool_indices[category, positions])
        if fallback:
            sampled.append(torch.randint(self._size, (fallback,)))
        indices = torch.cat(sampled)
        indices = indices[torch.randperm(indices.numel())]

        sampled_category = self.data['category'][indices]
        selected_counts = torch.bincount(sampled_category, minlength=CATEGORY_COUNT).float()
        pool_sizes = torch.tensor(self._pool_sizes, dtype=torch.float32).clamp_min(1.0)
        mixture = selected_counts / requested
        probability = mixture[sampled_category] / pool_sizes[sampled_category]
        uniform_probability = 1.0 / self._size
        importance = (uniform_probability / probability.clamp_min(1e-12)).pow(
            self.importance_beta,
        )
        importance = importance / importance.max().clamp_min(1.0)

        target_device = torch.device(device)
        batch = {
            key: value[indices].to(target_device, non_blocking=self._pin_memory)
            for key, value in self.data.items()
        }
        batch['indices'] = indices.to(target_device)
        batch['importance_weights'] = importance.to(target_device, non_blocking=self._pin_memory)
        return batch

    def relabel_boundaries(self, indices: torch.Tensor, boundary_score: torch.Tensor) -> None:
        """Refresh continuous boundary scores and their derived sampling category."""
        cpu_indices = indices.detach().to('cpu', dtype=torch.long)
        cpu_scores = boundary_score.detach().to('cpu', dtype=torch.float32).reshape(-1)
        if cpu_indices.numel() != cpu_scores.numel():
            raise ValueError('boundary indices and scores must have the same size')
        if not bool(torch.isfinite(cpu_scores).all()) or bool(
            ((cpu_scores < 0) | (cpu_scores > 1)).any(),
        ):
            raise ValueError('boundary scores must be finite and lie in [0, 1]')
        for slot, score in zip(cpu_indices.tolist(), cpu_scores.tolist()):
            if not 0 <= slot < self._size:
                raise IndexError(f'replay boundary slot {slot} is not resident')
            self.data['boundary_score'][slot] = score
            is_boundary = score >= self.boundary_score_threshold
            self.data['is_boundary'][slot] = is_boundary
            if bool(self.data['is_failure'][slot].item()):
                continue
            desired = BOUNDARY if is_boundary else NORMAL
            if int(self.data['category'][slot].item()) != desired:
                self._remove_pool_slot(slot)
                self._add_pool_slot(slot, desired)

    @property
    def label_counts(self) -> dict[str, int]:
        """Return finalized resident negative/positive probability labels."""
        valid = self.capacity if self._size == self.capacity else self._size
        labels = self.data['violation_label'][:valid]
        finalized = torch.isfinite(labels)
        positives = int((labels[finalized] > 0.5).sum().item())
        negatives = int(finalized.sum().item()) - positives
        return {
            'negative': negatives,
            'positive': positives,
            'finalized': int(finalized.sum().item()),
        }

    def category_ratios(self) -> dict[str, float]:
        """Return resident replay fractions for training diagnostics."""
        denominator = max(self._size, 1)
        return {key: value / denominator for key, value in self.pool_sizes.items()}

    def reset_pending_episodes(self) -> None:
        """Drop unfinished slot chains when a simulator cannot be restored exactly."""
        self._episode_slots = [[] for _ in range(self.num_envs)]

    def state_dict(self) -> dict[str, Any]:
        """Return a compact replay snapshot suitable for local full checkpoints."""
        valid = self.capacity if self._size == self.capacity else self._size
        return {
            'version': 2,
            'capacity': self.capacity,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'context_dim': self.context_dim,
            'size': self._size,
            'pointer': self._ptr,
            'data': {key: value[:valid].clone() for key, value in self.data.items()},
            'episode_slots': self._episode_slots,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a compact replay snapshot and rebuild category pools."""
        if state.get('version') != 2:
            raise ValueError('unsupported STAR replay checkpoint version')
        for name, expected in (
            ('obs_dim', self.obs_dim),
            ('act_dim', self.act_dim),
            ('context_dim', self.context_dim),
        ):
            if int(state[name]) != expected:
                raise ValueError(f'saved replay {name} does not match configured replay')
        saved_size = int(state['size'])
        saved_data = state['data']
        valid = next(iter(saved_data.values())).shape[0]
        if saved_size > self.capacity or valid > self.capacity:
            raise ValueError('saved replay does not fit the configured capacity')
        if set(saved_data) != set(self.data):
            raise ValueError('saved replay fields do not match configured replay fields')
        for key, value in saved_data.items():
            if value.shape[1:] != self.data[key].shape[1:]:
                raise ValueError(f'saved replay field {key!r} has an incompatible shape')
            self.data[key][:valid].copy_(value)
        self._size = saved_size
        self._ptr = int(state['pointer']) % self.capacity
        self._pool_positions.fill_(-1)
        self._pool_sizes = [0 for _ in range(CATEGORY_COUNT)]
        for slot in range(valid):
            category = int(self.data['category'][slot].item())
            if category >= 0:
                self._add_pool_slot(slot, category)
        self._episode_slots = state['episode_slots']
