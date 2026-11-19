from copy import deepcopy

import pytest

from scripts.paper_crax_payload.close_samehardware_v12 import terminal_state
from scripts.paper_crax_payload.summarize_matched_ablation_v12 import (
    COUNTS, pre_decay_prefix, registered_protocols, candidate_checks, training_curve_comparisons)


def test_registered_controls_and_pre_lr_divergence_are_never_hidden():
    specs = registered_protocols(True)
    assert list(specs) == ['A-full-P5', 'A-Iso-matchedLR', 'A-no-lateLR', 'A-Metric-matchedLR']
    assert specs['A-full-P5']['evaluation'] == specs['A-Metric-matchedLR']['evaluation']
    rows = [dict(arm=arm, steps=step, **{key: 0 for key in COUNTS},
                 actor_loss=0, reward_loss=0, margin_loss=0, cost_loss=0)
            for arm in ('A-full-P5', 'A-no-lateLR') for step in range(1000, 150001, 1000)]
    assert pre_decay_prefix(rows)['exact_recorded_prefix_match']
    changed = deepcopy(rows); changed[152]['reward_loss'] = 1e-8
    result = pre_decay_prefix(changed)
    assert result['first_observed_difference'] == dict(transitions=3000, differing_metrics=['reward_loss'])
    assert not result['strict_lr_causal_claim_permitted']
    assert not terminal_state(dict(status='running', pending=0, live_native_pids=[123],jobs=[]))
    assert terminal_state(dict(status='completed', pending=0, live_native_pids=[],jobs=[]))
    with pytest.raises(RuntimeError, match='never retry'):
        terminal_state(dict(status='completed_with_failures'))


def test_candidate_criteria_preserve_timeout_tradeoffs_and_actual_curve_nodes():
    specs = registered_protocols(direction_taper=True)
    candidate = 'D-direction-only-taper'
    assert len(specs) == 5 and specs[candidate]['evaluation'] == specs['A-full-P5']['evaluation']
    windows = []; nodes = []
    for arm, early, late, timeout in ((candidate,200,100,100),('A-full-P5',300,200,80),
                                      ('A-Iso-matchedLR',250,150,90),('A-Metric-matchedLR',260,100,80)):
        for start,end,bad,good,to in ((0,100000,early,1000,timeout),
                                     (100000,250000,late,2000,0),(0,250000,early+late,3000,timeout)):
            windows.append(dict(arm=arm,start_transitions=start,end_transitions=end,
                training_violations=bad,safe_successes=good,safe_timeouts=to,
                training_stall_episodes=to,training_cumulative_return=10000))
        nodes.append(dict(arm=arm,transitions=250000,condition='nominal',
            safe_successes=29,violations=1,timeouts=0,stall_count=0))
    checks = candidate_checks(windows,nodes)
    def check(name, baseline, scope):
        return next(r['satisfied'] for r in checks if r['criterion']==name and r['baseline']==baseline and r['scope']==scope)
    assert check('historical_first100K_violations','historical_Iso_331.5','0-100K')
    assert check('training_violations_at_least20percent_lower','A-Iso-matchedLR','0-250K')
    assert not check('training_violations_at_least20percent_lower','A-Metric-matchedLR','0-250K')
    assert not check('safe_timeouts_no_higher','A-full-P5','0-250000')
    rows = [dict(arm=arm,steps=step,**{key:value for key in COUNTS}) for arm,value in ((candidate,3),('A-full-P5',5))
            for step in (1000,2000)]
    comparisons = training_curve_comparisons(rows,candidate,['A-full-P5'])
    assert [r['transitions'] for r in comparisons] == [1000,2000]
    assert all(r['delta_training_violations']==-2 for r in comparisons)


def test_endpoint_residual_checks_preserve_new_control_and_reject_task_abandonment():
    specs = registered_protocols(endpoint_residual=True)
    assert len(specs)==6
    candidate,control = 'E-endpoint-residual','E-control-P5'
    assert specs[candidate]['evaluation']==specs['A-Iso-matchedLR']['evaluation']
    windows,nodes = [],[]
    for arm in (candidate,control,'A-full-P5','A-Iso-matchedLR','A-Metric-matchedLR'):
        is_candidate = arm==candidate
        for start,end in ((0,100000),(100000,250000),(0,250000)):
            windows.append(dict(arm=arm,start_transitions=start,end_transitions=end,
                training_violations=100 if is_candidate else 300,
                safe_successes=500 if is_candidate else 1000,
                safe_timeouts=200 if is_candidate else 50,
                training_stall_episodes=200 if is_candidate else 50,
                training_cumulative_return=5000 if is_candidate else 10000))
        nodes.append(dict(arm=arm,transitions=250000,condition='nominal',
            safe_successes=20 if is_candidate else 29,violations=0 if is_candidate else 1,
            timeouts=10 if is_candidate else 0,stall_count=5 if is_candidate else 0))
    checks = candidate_checks(windows,nodes,candidate,(control,'A-full-P5','A-Iso-matchedLR','A-Metric-matchedLR'))
    paired = [r for r in checks if r['baseline']==control]
    assert paired
    assert all(r['satisfied'] for r in paired if r['criterion']=='first100K_violations_no_higher')
    for criterion in ('safe_timeouts_no_higher','training_stall_episodes_no_higher',
                      'training_safe_success_count_no_lower','cumulative_return_drop_at_most1percent',
                      'DEV_safe_successes_no_worse','DEV_timeouts_no_worse','DEV_stall_count_no_worse'):
        assert any(r['criterion']==criterion and not r['satisfied'] for r in paired)
    assert not any(r['independent_confirmation'] for r in checks)


def test_terminal_queue_flag_cannot_override_live_native_handle(monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path,'read_bytes',lambda self:
        b'python\x00-m\x00scripts.paper_crax_payload.train_endpoint_residual_v12\x00--output\x00/data/run')
    queue = dict(status='completed',pending=0,live_native_pids=[],jobs=[dict(pid=123,output='/data/run')])
    assert not terminal_state(queue)
    monkeypatch.setattr(Path,'read_bytes',lambda self:b'unrelated-process')
    assert terminal_state(queue)
