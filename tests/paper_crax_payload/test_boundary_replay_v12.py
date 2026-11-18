from copy import deepcopy

import numpy as np
import pytest

from scripts.paper_crax_payload.boundary_replay_v12 import BoundaryReplay


def add(replay, env=0, margin=-.02, terminal=False, violation=False, collision=False):
    replay.add(obs=[replay.writes], next_obs=[replay.writes+1], action=[0, 0, 0],
               reward=0, done=float(terminal), cost=float(violation), margin=margin)
    replay.observe_transition(env, terminal, dict(local_margin=margin, violation=violation, collision=collision))


def test_observed_violation_marks_only_existing_same_episode_history():
    replay = BoundaryReplay(30, 1)
    add(replay, env=0); add(replay, env=1); add(replay, env=0, terminal=True)
    add(replay, env=0); add(replay, env=0)
    assert not replay.flags.any()  # No labels based on a future violation.
    add(replay, env=0, margin=.001, terminal=True, violation=True)
    assert list(np.flatnonzero(replay.flags & replay.PRE_VIOLATION)) == [3, 4]
    add(replay, env=0)
    assert replay.flags[6] == 0  # Never cross the terminal/reset boundary.
    add(replay, env=1, margin=-.001)
    add(replay, env=1, margin=-.02)
    assert replay.flags[8] & replay.RECOVERY


def test_priority_odds_four_and_native_recent_pool_are_preserved():
    np.random.seed(42)
    replay = BoundaryReplay(12000, 1)
    for i in range(12000):
        add(replay, terminal=True, margin=-.001 if i % 5 == 0 else -.02)
    batch = replay.sample(80000, 'cpu', recent=False)
    observed = np.mean(batch['obs'].numpy().ravel() % 5 == 0)
    assert abs(observed-.5) < .012  # .2*4 / (.2*4+.8)
    recent = replay._draw(20000, True)
    assert recent.min() >= 2000
    assert abs(np.mean(recent % 5 == 0)-.5) < .02


def test_ring_generations_and_nonempty_resume_preserve_sampling_exactly():
    replay = BoundaryReplay(3, 1)
    add(replay, env=0)
    for _ in range(3): add(replay, env=1, terminal=True)
    add(replay, env=0, margin=.001, terminal=True, violation=True, collision=True)
    assert not np.any(replay.flags & replay.PRE_VIOLATION)  # Stale slot now holds another env.
    state = deepcopy(replay.state_dict()); restored = BoundaryReplay(3, 1)
    restored.load_state_dict(state)
    rng = np.random.get_state(); first = replay.sample(128, 'cpu', recent=True)
    np.random.set_state(rng); second = restored.sample(128, 'cpu', recent=True)
    assert np.array_equal(first['obs'].numpy(), second['obs'].numpy())
    assert replay.metrics() == restored.metrics()
    with pytest.raises(ValueError, match='identical'):
        restored.load_state_dict({k: v for k, v in state.items() if k != 'boundary_priority_v12'})
