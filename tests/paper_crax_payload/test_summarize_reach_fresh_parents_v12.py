from copy import deepcopy
import json
import sys

import pytest

from scripts.paper_crax_payload.summarize_reach_fresh_parents_v12 import PLAN, PROJECT, diagnose, main


def curve():
    return [dict(steps=s, episodes=s//100, training_safe_successes=0,
        training_violation_count=2, training_safe_timeouts=s//100-2,
        learner_updates=s-2000, reward_updates=2*(s-2000), actor_updates=(s-3000)//2,
        cumulative_elapsed_seconds=t, recent_safe_success=0., recent_return=-1.,
        alpha=.0001, safety_multiplier=0., q_policy_mean=-.3, policy_cost=.0002)
        for s,t in ((10000,100.), (20000,1100.), (30000,2100.))]


def test_keeps_zero_success_seed_and_uses_recent_throughput():
    result = diagnose(curve(), 300000)
    assert result['first_success_logged_at_or_before_step'] is None
    assert result['no_success_observed_through_step'] == 30000
    assert result['actor_updates_advancing_in_last_log_interval']
    assert result['recent_window_sps'] == 10.
    assert result['approximate_remaining_hours'] == 7.5
    assert result['winner_confirmation'] is False


def test_first_success_is_a_logging_bound_not_exact_transition():
    rows = curve()
    rows[-1]['training_safe_successes'] = 1
    rows[-1]['training_safe_timeouts'] -= 1
    assert diagnose(rows, 300000)['first_success_logged_at_or_before_step'] == 30000
    assert diagnose(rows[:1], 300000)['approximate_remaining_hours'] is None


@pytest.mark.parametrize('damage', ['retry', 'nan', 'accounting'])
def test_rejects_merged_retries_nan_or_broken_accounting(damage):
    rows = deepcopy(curve())
    if damage == 'retry':
        rows[-1]['steps'] = 10000
    elif damage == 'nan':
        rows[-1]['alpha'] = float('nan')
    else:
        rows[-1]['training_safe_timeouts'] += 1
    with pytest.raises(ValueError):
        diagnose(rows, 300000)


def test_partial_native_two_seed_export_preserves_layout_seed(tmp_path, monkeypatch):
    plan = json.loads(PLAN.read_text())
    base = json.loads((PROJECT/plan['base_config']).read_text())
    for seed in plan['training_seeds']:
        directory = tmp_path/f'Reach-EA-{seed}'
        directory.mkdir()
        config = deepcopy(base)
        config['development_seed'] = seed
        config['evaluation']['development_layout_seed_start'] = plan['development_layout_seed_start']
        config['evaluation']['action_sampling_seed'] = plan['action_sampling_seed']
        manifest = dict(training_seed=seed, initial_steps=0, parent_checkpoint=None,
            initialization='fresh_Piper_no_Panda_transfer', source_commit='22d4966f',
            total_transition_budget=300000, configuration=config)
        (directory/'run_config.json').write_text(json.dumps(manifest))
        (directory/'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in curve()))
        records = [dict(seed=s, safe_success=False, violation=False, episode_return=-1.,
            trajectory=[]) for s in range(plan['development_layout_seed_start'],
                plan['development_layout_seed_start']+30)]
        evaluation = dict(new_training_transitions=0, adaptation_updates=0, training_rng_preserved=True,
            conditions=[dict(condition=c['name'], records=records, safe_success_count=0,
                violation_count=0, safe_timeout_count=30) for c in config['evaluation']['conditions']])
        (directory/'evaluation_30000.json').write_text(json.dumps(evaluation))
    destination = tmp_path/'compact'
    monkeypatch.setattr(sys, 'argv', ['summarize', '--runs', str(tmp_path), '--output', str(destination)])
    main()
    receipt = json.loads((destination/'receipt.json').read_text())
    assert receipt['status'] == 'partial_native_parent_learning_diagnostic'
    assert receipt['completed_parents'] == 0
    assert receipt['observed_training_steps'] == 60000
    assert len(receipt['parents']) == 2 and receipt['omitted_registered_seeds'] == []
    assert (destination/'development_episodes.csv').read_text().splitlines()[1].startswith('41028,30000,nominal,11990000,')
    monkeypatch.setattr(sys,'argv',['summarize','--runs',str(tmp_path),'--output',str(tmp_path/'rejected'),
        '--require-complete'])
    with pytest.raises(FileNotFoundError):
        main()
    assert not (tmp_path/'rejected').exists()

    # Build a complete receipt/node fixture while retaining the zero-success seed.
    for seed in plan['training_seeds']:
        directory = tmp_path/f'Reach-EA-{seed}'
        rows = []
        for step in range(1000,300001,1000):
            row = deepcopy(curve()[0])
            row.update(steps=step,episodes=step//100,training_safe_timeouts=step//100-2,
                learner_updates=max(0,step-2000),reward_updates=2*max(0,step-2000),
                actor_updates=max(0,(step-3000)//2),cumulative_elapsed_seconds=step/10)
            rows.append(row)
        (directory/'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        last = rows[-1]
        common = dict(training_seed=seed,training_steps=300000,
            **{k:last[k] for k in ('training_safe_successes','training_violation_count','training_safe_timeouts')})
        (directory/'training_receipt_300000.json').write_text(json.dumps(dict(common,status='completed_native_checkpoint')))
        (directory/'receipt.json').write_text(json.dumps(dict(common,status='completed',
            newly_collected_transitions=300000,initial_steps=0,engineering=False,training_source_commit='22d4966f',
            training_episodes=last['episodes'],learner_updates=last['learner_updates'],
            reward_updates=last['reward_updates'],actor_updates=last['actor_updates'])))
        (directory/'full_300000.pt').write_bytes(b'fixture_only_not_a_loadable_scientific_checkpoint')
        evaluation = json.loads((directory/'evaluation_30000.json').read_text())
        (directory/'evaluation_30000.json').unlink()
        for step in (60000,120000,180000,240000,300000):
            (directory/f'evaluation_{step}.json').write_text(json.dumps(evaluation))
    monkeypatch.setattr(sys,'argv',['summarize','--runs',str(tmp_path),'--output',str(tmp_path/'complete'),
        '--require-complete'])
    main()
    receipt = json.loads((tmp_path/'complete/receipt.json').read_text())
    assert receipt['complete_native_campaign_verified'] and receipt['completed_parents']==2
    assert receipt['observed_training_steps']==600000
    assert receipt['winner_confirmation'] is False and receipt['downstream_training_started'] is False
    import csv
    aggregates = list(csv.DictReader((tmp_path/'complete/equal_seed_development.csv').read_text().splitlines()))
    assert len(aggregates)==15
    assert all(float(r['success_worst'])==0 for r in aggregates)
