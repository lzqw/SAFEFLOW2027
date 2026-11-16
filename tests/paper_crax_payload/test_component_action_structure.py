import numpy as np
import pytest

from scripts.paper_crax_payload.component_action_structure import action_statistics


def test_within_state_covariance_not_variance_of_the_mean():
    samples = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
    value = action_statistics(samples)
    np.testing.assert_allclose(value['sample_covariance'], np.diag([2 / 3, 2 / 3, 0]))
    assert value['covariance_trace'] == pytest.approx(4 / 3)
    assert value['final_action_boundary_fraction'] == 1.
    shifted = action_statistics(samples + [20., -30., 40.])
    np.testing.assert_allclose(value['sample_covariance'], shifted['sample_covariance'])
    with pytest.raises(ValueError):
        action_statistics([[np.nan, 0., 0.], [0., 0., 0.]])
