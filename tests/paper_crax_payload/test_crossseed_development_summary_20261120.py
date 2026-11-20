from copy import deepcopy

import pytest

from scripts.paper_crax_payload.close_reach_crossseed_20261120 import summarize_groups


def fixture():
    plan = dict(compiled_training_seed=41028, lines={'piper_reach': dict(
        conditions=[{'name': 'radius_060'}], development_layout_seed_start=12020000)})
    records = [dict(seed=12020000+i, safe_success=i != 0, violation=i == 0,
                    stall=False, episode_return=10. if i else -10., time_to_goal_capped_seconds=1.)
               for i in range(100)]
    value = dict(independent_test_used=False, fixed_node_not_best_checkpoint=True,
        arm='P5-fixed', new_steps=150000, deployment_modes=['sampled', 'zero_noise'],
        conditions=[dict(deployment_mode=mode, condition='radius_060', records=deepcopy(records))
                    for mode in ('sampled', 'zero_noise')])
    return plan, value


def test_keeps_modes_and_failed_episodes_separate():
    plan, value = fixture()
    nodes, episodes = summarize_groups(value, plan, 'P5-fixed', 150000)
    assert len(nodes) == 2 and len(episodes) == 200
    assert {n['deployment_mode'] for n in nodes} == {'sampled', 'zero_noise'}
    assert [n['violations'] for n in nodes] == [1, 1]
    assert sum(e['violation'] for e in episodes) == 2


def test_rejects_omitted_mode_or_failed_layout():
    plan, value = fixture()
    value['conditions'].pop()
    with pytest.raises(ValueError, match='every registered condition'):
        summarize_groups(value, plan, 'P5-fixed', 150000)
    plan, value = fixture()
    value['conditions'][0]['records'].pop(0)
    with pytest.raises(ValueError, match='100 registered layout'):
        summarize_groups(value, plan, 'P5-fixed', 150000)
