"""One synthetic complete campaign; never launches RL or native simulation."""
import json
from pathlib import Path
from statistics import mean

import pytest

from scripts.paper_crax_payload.report_cvpo_campaign import (
    MILESTONES, SEEDS, canonical_digest, prepare, report, sha256)


ROOT = Path(__file__).resolve().parents[2]


def test_complete_report_retains_originals_and_rejects_tampered_or_partial_records(tmp_path):
    runs = tmp_path / 'runs'
    closure = tmp_path / 'closure'
    closure.mkdir()
    config = json.loads((ROOT / 'configs/paper_crax_payload/cvpo_formal_v4.json').read_text())
    receipts, candidates = [], []

    def write(path, value):
        path.write_text(json.dumps(value))

    def evaluation(seed, count, goals, cost):
        records = [dict(native_return=goals, episode_cost=cost, goals=goals) for _ in range(count)]
        return dict(seed=seed, episodes=count, records=records, mean_native_return=goals,
            mean_goals_per_episode=goals, mean_episode_cost=cost, budget_compliant_fraction=1.)

    for seed in SEEDS:
        directory = runs / f'cvpo-{seed}-10m'
        directory.mkdir(parents=True)
        receipt = dict(status='completed', steps=10000128, training_seed=seed,
            evaluated_milestones=MILESTONES, training_cost=1000., elapsed_seconds=100.,
            complete_train_sps=100001.28, source_commit='fixture',
            official_source_commit=config['official_source_commit'])
        write(directory / 'receipt.json', receipt)
        write(directory / 'run_config.json', dict(configuration=config, training_seed=seed,
            source_commit='fixture', official_source_commit=config['official_source_commit']))
        receipts.append(receipt)
        for index, step in enumerate(MILESTONES):
            results = [evaluation(s, 50, index + 1., 20.) for s in range(61027, 61032)]
            path = directory / f'validation_{step}.json'
            write(path, dict(steps=step, training_seed=seed, split='common_validation',
                independent_test_used=False, policy_sha256='fixture-policy', results=results))
            candidates.append(dict(method='cvpo', run=directory.name, training_seed=seed,
                checkpoint_steps=step, policy_sha256='fixture-policy', validation_sha256=sha256(path),
                mean_native_return=index + 1., mean_goals_per_episode=index + 1.,
                mean_episode_cost=20., budget_compliant_fraction=1., mean_capped_first_goal_seconds=1.))
    winner = candidates[4]
    selection = dict(source_commit='fixture', candidates=candidates, selected=winner,
                     independent_test_used=False, status='risk_qualified')
    selection['freeze_digest'] = canonical_digest(selection)
    write(closure / 'selection.json', selection)
    write(closure / 'receipt.json', dict(status='completed', independent_episodes=500,
        training_transitions=0, existing_three_method_results_not_changed=True,
        selection_freeze_digest=selection['freeze_digest'], source_commit='fixture'))
    write(closure / 'fixed_10m.json', dict(training_seed_count=3,
        by_training_seed=[r for r in candidates if r['checkpoint_steps'] == 10000128], receipts=receipts))
    independent = dict(selected=winner, episodes=500, independent_test_used_for_selection=False,
        selection_freeze_digest=selection['freeze_digest'], mean_native_return=7.,
        mean_goals_per_episode=7., mean_episode_cost=20., budget_compliant_fraction=1.,
        per_seed=[evaluation(s, 100, 7., 20.) for s in range(71027, 71032)])
    write(closure / 'independent.json', independent)
    historical = ROOT / 'reports/robot_gap_v2/paper'
    value = prepare(runs, closure, historical)
    assert len(value['curves']) == 60 and len(value['endpoints']) == 12
    assert len(value['selected']) == 4 and len(value['summary']) == 4
    assert value['cvpo_actual_training_transitions'] == 30000384
    final = next(r for r in value['summary'] if r['method'] == 'cvpo')
    assert final['mean_goals_per_episode_mean'] == 5.
    assert final['mean_goals_per_episode_sd'] == 0.
    assert next(r for r in value['selected'] if r['method'] == 'cvpo')['mean_goals_per_episode'] == 7.
    # Original values/identities remain byte-serialized CSV values, not reselected candidates.
    assert next(r for r in value['selected'] if r['method'] == 'ea')['checkpoint_steps'] == '8000000'
    output = tmp_path / 'report'
    result = report(runs, closure, historical, output)
    assert result['selection_repeated'] is False and result['new_simulation_controls'] == 0
    assert (output / 'four_method_learning.png').exists()
    independent['per_seed'][0]['records'].pop()
    write(closure / 'independent.json', independent)
    missing = tmp_path / 'must-not-create'
    with pytest.raises(ValueError, match='denominator'):
        report(runs, closure, historical, missing)
    assert not missing.exists()
