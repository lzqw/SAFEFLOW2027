from collections import Counter

import pytest

from deployment.piper.protocol import reach90_schedule
from deployment.piper.replay_policy import latency_summary


def test_offline_latency_reports_p99_and_does_not_invent_a_calibrated_period():
    value = latency_summary([.01,.02,.03],.02)
    assert value['p50_s'] == pytest.approx(.02)
    assert value['p99_s'] >= value['p95_s']
    assert value['diagnostic_deadline_misses'] == 1
    assert value['period_physically_calibrated'] is False
    assert latency_summary([.01])['diagnostic_deadline_misses'] is None
    with pytest.raises(ValueError):
        latency_summary([.01],-1.)


def test_default_real_preparation_is_only_the_balanced_unexecuted_reach90():
    rows = reach90_schedule()
    assert len(rows) == 90
    assert set(r['task'] for r in rows) == {'Reach'}
    counts = Counter((r['method'],r['condition']) for r in rows)
    assert len(counts) == 9 and set(counts.values()) == {10}
    assert all(r['status'] == 'planned_not_executed' for r in rows)
