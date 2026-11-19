from types import SimpleNamespace

import pytest
import torch

from scripts.paper_crax_payload.evaluate_piper_deployment_modes_v12 import deployment_mode
from scripts.paper_crax_payload.evaluate_panda_safereach_modes_v12 import normalize_reach, timed_actions


def test_timed_mode_only_changes_deterministic_flag_and_restores_native_method():
    class Learner:
        device = torch.device('cpu')
        def action(self,obs,deterministic=False):
            return obs+(10 if deterministic else 1)
    learner = Learner()
    with deployment_mode(learner,'zero_noise'),timed_actions(learner) as timings:
        assert learner.action(4)==14
    assert 'action' not in learner.__dict__ and learner.action(4)==5
    assert len(timings)==1 and timings[0]>=0
    result = dict(conditions=[dict(records=[dict(length=2,layout_seed=12000000,
        trajectory_tcp_xyz=[[0,0,0],[1,0,0],[2,0,0]],capped_seconds=4)])])
    value = normalize_reach(result,[1.,3.])['conditions'][0]['records'][0]
    assert value['seed']==12000000 and value['inference_p50_ms']==2.
    assert value['time_to_goal_capped_seconds']==4
    with pytest.raises(ValueError,match='Every native action'):
        normalize_reach(result,[1.])
