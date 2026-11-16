import gzip
import json

import pytest

from scripts.paper_crax_payload.diagnose_direction_margin import calibration_statistics, terminal_records


def test_reconstruct_pre_action_not_logged_next_observation_for_each_vector_env(tmp_path):
    initial = lambda env, episode, obs: dict(record_type='initial', env=env, episode_id=episode, observation=[obs])
    step = lambda env, episode, n, obs, done: dict(record_type='step', env=env, episode_id=episode,
        transitions=n, observation=[obs], executed_action=[0., 0., 0.], margin=-.01, violation=False, done=done)
    records = [initial(0, 0, 0), initial(1, 0, 10), step(0, 0, 1, 1, False),
        step(1, 0, 2, 11, True), initial(1, 1, 20), step(0, 0, 3, 2, True),
        initial(0, 1, 30), step(1, 1, 4, 21, False), dict(record_type='end')]
    with gzip.open(tmp_path/'trajectories.jsonl.gz', 'wt') as stream:
        for record in records: stream.write(json.dumps(record)+'\n')
    result = terminal_records(tmp_path, start=0, end=4)
    assert [row['obs'][0] for row in result] == [0, 10, 1, 20]
    assert [row['steps'] for row in result] == [1, 2, 3, 4]


def test_small_global_mse_can_coexist_with_false_safe_actual_unsafe_action():
    result = calibration_statistics([[-.002, -.004, .003], [0., -.002, .001]], [.001, -.003, .002])
    assert result['all']['count'] == 3
    assert result['all']['rmse_mm'] == pytest.approx((4./3.)**.5)
    assert result['unsafe_actual']['count'] == 2
    assert result['unsafe_actual']['false_safe_mean_count'] == 1
    assert result['unsafe_actual']['both_heads_predict_safe_count'] == 1
    assert result['unsafe_actual']['mean_plus_one_head_std_predicts_safe_count'] == 1
    assert result['safe_interior_actual']['count'] == 0
