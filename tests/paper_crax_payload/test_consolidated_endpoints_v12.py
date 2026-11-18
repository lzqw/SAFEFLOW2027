import pytest

from scripts.paper_crax_payload.consolidate_endpoints_v12 import measured_count, measured_mean, truth


def test_missing_measurements_are_not_zero_and_quantiles_are_episode_means():
    assert measured_count([{'stall': 'False'}, {'stall': 'True'}], 'stall') == 1
    assert measured_count([{'stall': 'False'}, {'stall': ''}], 'stall') is None
    assert measured_mean([{'p50': '2'}, {'p50': '8'}], 'p50') == 5.
    assert measured_mean([{'p50': '2'}, {}], 'p50') is None
    with pytest.raises(ValueError, match='missing'):
        truth('')
