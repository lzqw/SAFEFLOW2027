import pytest

from scripts.paper_crax_payload.compare_piper_deployment_modes_v12 import comparison, validate_records


def row(seed,success,violation,timeout,stall=False):
    return dict(seed=seed,safe_success=success,violation=violation,safe_timeout=timeout,stall=stall,
        ordinary_terminal_task_success=success,episode_return=1 if success else -1,
        time_to_goal_capped_seconds=1 if success else 4,inference_p50_ms=2,inference_p95_ms=3,
        length=1,trajectory=[[0,0,0],[1,0,0]])


def test_pairwise_safety_gain_does_not_hide_timeout_or_lost_layout():
    old = [row(100,True,False,False),row(101,False,True,False)]
    new = [row(100,False,False,True,True),row(101,True,False,False)]
    validate_records(new,100,2)
    value = comparison(new,old)
    assert value['delta_safe_successes']==0 and value['delta_violations']==-1
    assert value['candidate_only_safe']==1 and value['baseline_only_safe']==1
    assert value['delta_timeouts']==1 and value['delta_stall_count']==1
    with pytest.raises(ValueError,match='registered layout'):
        validate_records(new[:1],100,2)
    new[0]['violation']=True
    with pytest.raises(ValueError,match='terminal class'):
        validate_records(new,100,2)
