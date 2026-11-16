import csv
import json

from scripts.paper_crax_payload.compare_direction_annealing_development import analyze


def test_paired_actual_failures_and_uncollected_branch_not_scored_as_zero(tmp_path):
    config = dict(arms=[dict(name=name) for name in ('A', 'B', 'C')], evaluation=dict(
        milestones=[125000], development_episodes_each=1, development_layout_seed_start=7,
        conditions=[dict(name='execution085')]))
    for arm, failed in [('A', False), ('B', True)]:
        directory = tmp_path/'runs'/arm
        directory.mkdir(parents=True)
        control = dict(step=1, violation=failed, minimum_distance_m=.01 if failed else .03,
            safety_clearance_deficit_m=.005 if failed else 0., closest_protected_link=8,
            raw_norm=1., executed_norm=.85, actual_tcp_displacement_norm_m=.02,
            task_progress_m=.01, collision=False)
        record = dict(layout_seed=7, safe_success=not failed, violation=failed, safe_timeout=False,
            length=1, episode_return=1., minimum_distance_m=control['minimum_distance_m'], capped_seconds=.1,
            layout=dict(initial_minimum_distance_m=.04), control_details=[control])
        (directory/'evaluation_125000.json').write_text(json.dumps(dict(milestone=125000,
            split='paired_fork_development_only', independent_confirmation=False,
            conditions=[dict(condition='execution085', records=[record]) ])))
    output = tmp_path/'analysis'
    result = analyze(config, tmp_path/'runs', output)
    assert result['executed_episode_records'] == 2 and result['violating_controls'] == 1
    assert result['new_simulation_controls'] == result['new_training_transitions'] == 0
    with (output/'paired_development_outcomes.csv').open() as stream:
        row = next(csv.DictReader(stream))
        assert row['C_outcome'] == 'not_collected' and row['C_length'] == ''
    with (output/'paired_development_discordance.csv').open() as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and rows[0]['left_only_success'] == '1'
    with (output/'actual_failure_severity.csv').open() as stream:
        row = next(csv.DictReader(stream))
        assert row['arm'] == 'B' and float(row['clearance_deficit_mm']) == 5.
