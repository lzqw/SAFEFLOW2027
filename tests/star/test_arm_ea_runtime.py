import random

import numpy as np
import torch

from scripts.star.arm_safe_entropy.stability_runtime import (
    capture_rng,restore_rng,isolated_rng,capture_environment,restore_environment)


def test_evaluation_rng_scope_restores_python_numpy_and_torch():
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    original=capture_rng()
    expected=(random.random(),np.random.rand(),torch.randn(3))
    restore_rng(original)
    with isolated_rng():
        random.seed(123);np.random.seed(123);torch.manual_seed(123)
        random.random();np.random.randn(100);torch.randn(100)
    actual=(random.random(),np.random.rand(),torch.randn(3))
    assert actual[0]==expected[0] and actual[1]==expected[1]
    torch.testing.assert_close(actual[2],expected[2],rtol=0.,atol=0.)


def test_observed_physics_state_restore_preserves_partial_episode_and_next_interval():
    from scripts.star.arm_safe_entropy.environment import TaskConfig,make_environment
    a,b=[make_environment(TaskConfig(version='SafeReach-RelativeLayout-v3')) for _ in range(2)]
    try:
        a.reset(seed=78123)
        a.step(np.zeros(3))
        saved=capture_environment(a)
        restored=restore_environment(b,saved)
        np.testing.assert_allclose(restored,saved['observation'],rtol=0.,atol=2e-6)
        assert b._elapsed==a._elapsed==1
        assert b._previous_distance==a._previous_distance
        action=np.array([.1,-.1,.05])
        left,right=a.step(action),b.step(action)
        np.testing.assert_allclose(left[0],right[0],rtol=0.,atol=2e-5)
        assert abs(left[1]-right[1])<2e-5
        assert left[2:4]==right[2:4]
        assert abs(left[4]['minimum_distance']-right[4]['minimum_distance'])<2e-5
    finally:
        a.close();b.close()
