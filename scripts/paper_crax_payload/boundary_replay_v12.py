"""Observed-only boundary replay, fourfold odds within each native sample pool.

True terminal violations remain terminal. No fabricated post-contact rollouts,
future labels, environment queries, changed losses or extra gradient updates.
"""
from collections import deque
from copy import deepcopy

import numpy as np
import torch

from scripts.star.arm_safe_entropy.train import Replay


class BoundaryReplay(Replay):
    CONTACT, LOW, PRE_VIOLATION, VIOLATION, RECOVERY = 1, 2, 4, 8, 16
    SETTINGS = dict(priority_odds=4, low_margin_m=-.005, preceding_controls=5,
                    pools='native_half_uniform_half_recent10000', num_envs=2)

    def __init__(self, capacity, obs_dim):
        super().__init__(capacity, obs_dim)
        self.flags = np.zeros(capacity, np.uint8)
        self.generations = np.full(capacity, -1, np.int64)
        self.history = [deque(maxlen=5) for _ in range(2)]
        self.previous_margin = [None, None]
        self.writes = self.sampled = self.priority_sampled = 0
        self.pending = None
    def add(self, **values):
        if self.pending is not None:
            raise ValueError('Observe the previous real transition before adding another')
        slot = self.cursor
        super().add(**values)
        self.flags[slot] = 0
        self.generations[slot] = self.writes
        self.pending = (slot, self.writes)
        self.writes += 1

    def observe_transition(self, env, terminal, info):
        if self.pending is None or env not in (0, 1):
            raise ValueError('Metadata must follow exactly one native transition')
        slot, generation = self.pending
        margin = float(self.data['margin'][slot])
        if not np.isclose(margin, info['local_margin'], rtol=0, atol=1e-7):
            raise ValueError('Priority uses the same observed interval label as the critics')
        low = margin >= self.SETTINGS['low_margin_m']
        flags = self.LOW if low else 0
        if info['collision']:
            flags |= self.CONTACT
        if info['violation']:
            flags |= self.VIOLATION
            # Applied only once the violation has actually been observed.
            for prior_slot, prior_generation in self.history[env]:
                if self.generations[prior_slot] == prior_generation:
                    self.flags[prior_slot] |= self.PRE_VIOLATION
        previous = self.previous_margin[env]
        if previous is not None and previous >= self.SETTINGS['low_margin_m'] and not low:
            flags |= self.RECOVERY
        self.flags[slot] = flags
        self.history[env].append((slot, generation))
        self.previous_margin[env] = margin
        if terminal:
            self.history[env].clear()
            self.previous_margin[env] = None
        self.pending = None

    def _draw(self, count, recent):
        accepted = []
        remaining = count
        while remaining:
            # Rejection sampling gives exactly weights4:1 in either pool,
            # without rebuilding a150K-entry probability array every update.
            size = max(32, remaining * 4)
            index = (self.cursor - 1 - np.random.randint(min(self.count, 10000), size=size)) % self.capacity if recent else np.random.randint(self.count, size=size)
            priority = self.flags[index] != 0
            chosen = index[np.random.random(size) < np.where(priority, 1., .25)][:remaining]
            accepted.append(chosen)
            remaining -= len(chosen)
        return np.concatenate(accepted)

    def sample(self, batch_size, device, recent=False):
        if self.pending is not None or self.count == 0:
            raise ValueError('Sample only completely observed real transitions')
        n = batch_size // 2 if recent else 0
        indices = np.concatenate((self._draw(n, True), self._draw(batch_size-n, False))) if n else self._draw(batch_size, False)
        self.sampled += len(indices)
        self.priority_sampled += int(np.count_nonzero(self.flags[indices]))
        return {key: torch.as_tensor(value[indices], device=device) for key, value in self.data.items()}

    def metrics(self):
        return dict(replay_priority_fraction=float(np.mean(self.flags[:self.count] != 0)) if self.count else 0.,
                    replay_sampled_priority_fraction=self.priority_sampled/max(1, self.sampled),
                    replay_contact_count=int(np.count_nonzero(self.flags[:self.count] & self.CONTACT)),
                    replay_pre_violation_count=int(np.count_nonzero(self.flags[:self.count] & self.PRE_VIOLATION)),
                    replay_recovery_count=int(np.count_nonzero(self.flags[:self.count] & self.RECOVERY)))

    def state_dict(self):
        if self.pending is not None:
            raise ValueError('Checkpoint only complete transition metadata')
        return super().state_dict() | dict(boundary_priority_v12=dict(
            settings=deepcopy(self.SETTINGS), flags=self.flags[:self.count].copy(),
            generations=self.generations[:self.count].copy(), history=[list(v) for v in self.history],
            previous_margin=self.previous_margin.copy(), writes=self.writes,
            sampled=self.sampled, priority_sampled=self.priority_sampled))

    def load_state_dict(self, state):
        meta = state.get('boundary_priority_v12')
        if not meta or meta['settings'] != self.SETTINGS:
            raise ValueError('Require the identical observed-boundary replay state')
        super().load_state_dict(state)
        self.flags[:self.count] = meta['flags']
        self.generations[:self.count] = meta['generations']
        self.history = [deque(v, maxlen=5) for v in meta['history']]
        self.previous_margin = meta['previous_margin'].copy()
        self.writes, self.sampled, self.priority_sampled = (meta[k] for k in ('writes', 'sampled', 'priority_sampled'))
        self.pending = None


class SafetyOnlyBoundaryReplay(BoundaryReplay):
    """Keep native actor/reward sampling; prioritize only the safety batch."""
    def sample(self, batch_size, device, recent=False):
        ordinary = Replay.sample(self, batch_size, device, recent=recent)
        ordinary['safety_batch'] = super().sample(batch_size, device, recent=recent)
        return ordinary

    def state_dict(self):
        return super().state_dict() | dict(safety_only_boundary_replay_v12=True)

    def load_state_dict(self, state):
        if state.get('safety_only_boundary_replay_v12') is not True:
            raise ValueError('Only identical safety-only replay resumes are admitted')
        super().load_state_dict(state)
