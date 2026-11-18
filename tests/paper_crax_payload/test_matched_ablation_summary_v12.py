import pytest

from scripts.paper_crax_payload.summarize_matched_ablation_v12 import training_window


def row(step, violations, success, timeouts, returns):
    return dict(steps=step, episodes=violations+success+timeouts,
        training_violations=violations, safe_successes=success, safe_timeouts=timeouts,
        training_stall_episodes=timeouts, training_stall_windows=2*timeouts,
        training_cumulative_return=returns, ordinary_terminal_task_successes=success)


def test_windows_include_early_failures_and_subtract_actual_boundaries():
    rows = [row(100000,300,2000,400,10000.),row(250000,500,10000,405,90000.)]
    early = training_window(rows,0,100000); late = training_window(rows,100000,250000)
    total = training_window(rows,0,250000)
    assert early['training_violations'] == 300 and early['safe_timeouts'] == 400
    assert late['training_violations'] == 200 and late['safe_timeouts'] == 5
    assert total['training_cumulative_return'] == early['training_cumulative_return']+late['training_cumulative_return']
    with pytest.raises(ValueError,match='Exact'):
        training_window(rows,75000,250000)
    with pytest.raises(ValueError,match='decrease'):
        training_window([rows[0],row(250000,200,10000,405,90000.)],100000,250000)
