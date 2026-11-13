"""Small selection/recording contracts for the offline Reach release."""
from collections import Counter
import json

import pytest

from deployment.piper.replay_policy import read_observations
from scripts.paper_crax_payload.piper_reach_release import choose, reach_schedule


def candidate(step, success, risks, capped=3.):
    return dict(steps=step, score=dict(risk_qualified=all(r <= .1 for r in risks),
        task_progress_observed=success > 0, macro_safe_success=success,
        macro_capped_seconds=capped, per_condition_violation=risks))


def test_common_selection_qualifies_each_stress_and_keeps_stagnant_control():
    unsafe = candidate(60000, 1., [.0, .2, .0])
    safe = candidate(120000, .8, [.02, .1, .04])
    assert choose([unsafe, safe]) == (safe, 'risk_qualified')
    stagnant = candidate(300000, 0., [0., 0., 0.])
    assert choose([stagnant])[1] == 'risk_qualified_but_stagnant'
    assert choose([unsafe])[1] == 'pareto_tradeoff_not_risk_qualified'


def test_reach90_is_balanced_interleaved_and_cannot_enable_motion():
    schedule = reach_schedule()
    assert schedule['episodes'] == len(schedule['rows']) == 90
    counts = Counter((r['condition'], r['method']) for r in schedule['rows'])
    assert len(counts) == 9 and set(counts.values()) == {10}
    assert not schedule['physical_motion_authorized']


def test_observation_reader_requires_explicit_native_schema(tmp_path):
    path = tmp_path/'observations.jsonl'
    header = dict(record_type='metadata', robot_family='Piper', frame='robot_base', observation_dimension=34)
    path.write_text(json.dumps(header)+'\n'+json.dumps(dict(timestamp_s=0., observation=[0.]*34))+'\n')
    _, rows = read_observations(path)
    assert len(rows) == 1 and rows[0][1].shape == (34,)
    header['robot_family'] = 'Panda'
    path.write_text(json.dumps(header)+'\n')
    with pytest.raises(ValueError, match='Piper34'):
        read_observations(path)
