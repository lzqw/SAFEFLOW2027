import pytest

from scripts.paper_crax_payload.piper_frozen_frontier import episode_metrics, select_scale, wilson


def test_failures_receive_full_horizon_and_path_includes_first_interval():
    failed = episode_metrics(success=False,length=1,horizon=100,dt=.04,positions=[[0,0,0],[.1,0,0]])
    assert failed['time_to_goal_capped_seconds'] == 4.
    assert failed['time_to_goal_seconds'] is None
    assert failed['path_length_m'] == pytest.approx(.1)
    success = episode_metrics(success=True,length=2,horizon=100,dt=.04,positions=[[0,0,0],[.1,0,0],[.1,.2,0]])
    assert success['time_to_goal_capped_seconds'] == .08
    assert success['path_length_m'] == pytest.approx(.3)


def test_selection_never_calls_nonqualified_or_stagnant_a_task_solution():
    candidates = [dict(action_scale=.6,static_invalid_layouts=0,worst_condition_violation=0.,
        macro_safe_success=0.,macro_capped_seconds=4.),dict(action_scale=1.,static_invalid_layouts=0,
        worst_condition_violation=.2,macro_safe_success=.8,macro_capped_seconds=2.)]
    assert select_scale(candidates,.1)['status'] == 'risk_qualified_but_stagnant'
    assert select_scale(candidates[1:],.1)['status'] == 'nonqualified_tradeoff'


def test_zero_observed_risk_is_not_a_zero_upper_bound():
    low, high = wilson(0,100)
    assert low == pytest.approx(0.)
    assert .03 < high < .04
