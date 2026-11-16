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


def test_bounded_benchmark_can_include_offline_postprocessing():
    import numpy as np
    import torch
    from deployment.piper.replay_policy import benchmark
    class Policy:
        device='cpu';geometry=None
        manifest={'action_scale_m':.01}
        class actor:
            num_steps=4
            net=torch.nn.Identity()
        def action(self,obs):
            self.actor.net(torch.zeros(1,3))
            return np.zeros((1,3))
    obs=np.zeros(34);obs[12:15]=[.3,0,.3]
    value=benchmark(Policy(),[(0.,obs)],warmup=1,samples=3,include_postprocessing=True)
    assert value['measured_actions']==len(value['raw_action_latencies_s'])==3
    assert value['postprocessing_included'] and value['physical_commands_sent']==0
    assert value['diagnostic_deadline_misses'] is None
