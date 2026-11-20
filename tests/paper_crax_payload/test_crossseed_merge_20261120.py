import pytest

from scripts.paper_crax_payload.merge_reach_crossseed_20261120 import (
    ARMS, CONDITIONS, SEEDS, STEPS, aggregate, comparisons, modes)


def nodes():
    return [dict(training_seed=seed, arm=arm, new_steps=step, condition=condition,
                 deployment_mode=mode, episodes=100, safe_successes=success,
                 violations=100-success, timeouts=0, stall_count=0,
                 mean_return=10., mean_capped_seconds=1., independent_test_used=False)
            for seed, success in zip(SEEDS, (80, 100)) for arm in ARMS
            for step in STEPS for condition in CONDITIONS for mode in modes(step)]


def test_equal_seed_mean_does_not_hide_bad_seed_or_pool_modes():
    source = nodes()
    for row in source:
        if row['arm'] == 'Metric-only':
            row['safe_successes'], row['violations'] = 90, 10
    result = aggregate(source)
    row = next(r for r in result if r['arm'] == 'P5-fixed' and r['new_steps'] == 150000
               and r['condition'] == 'radius_060' and r['deployment_mode'] == 'zero_noise')
    assert row['success_mean'] == .9 and row['success_worst'] == .8
    checks = comparisons(result)
    check = next(r for r in checks if r['arm'] == 'P5-fixed' and r['new_steps'] == 150000
                 and r['condition'] == 'radius_060' and r['deployment_mode'] == 'zero_noise')
    assert check['reach_target_satisfied'] and not check['reach_worst_seed_at_least_85percent']
    assert not check['success_seed_sd_at_most_5pp']
    assert check['success_not_lower_than_metric']
    assert not check['both_seeds_success_not_lower_than_metric']
    assert not check['joint_development_task_checks']
    assert len(result) == 96 and len(checks) == 64


def test_missing_failed_seed_or_duplicated_mode_cannot_be_merged():
    original = nodes()
    with pytest.raises(ValueError, match='both seeds'):
        aggregate([r for r in original if r['training_seed'] == SEEDS[1]])
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate(original+[original[0]])
