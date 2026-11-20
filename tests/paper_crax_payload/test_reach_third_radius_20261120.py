from copy import deepcopy
import json

import pytest

from scripts.paper_crax_payload.train_reach_third_radius_20261120 import (
    FORMAT, PARENT_PLAN, PARENT_SOURCE, parent_configuration, validate_parent)


def inputs():
    config = parent_configuration(json.loads(PARENT_PLAN.read_text()))
    saved = dict(format=FORMAT, robot_family='Piper', task='Reach', method='EA',
        training_seed=41030, steps=300000, source_commit=PARENT_SOURCE, engineering=False,
        configuration=config, geometry_adaptation=None, task_transfer=None,
        violations=700, safe_successes=6000, safe_timeouts=846, episodes=7546)
    receipt = dict(status='completed', training_seed=41030, training_steps=300000,
        initial_steps=0, newly_collected_transitions=300000, training_source_commit=PARENT_SOURCE,
        engineering=False, training_violation_count=700, training_safe_successes=6000,
        training_safe_timeouts=846, training_episodes=7546)
    manifest = dict(source_commit=PARENT_SOURCE, training_seed=41030, initial_steps=0,
        total_transition_budget=300000, configuration=config, parent_checkpoint=None)
    return saved, receipt, manifest


def test_third_radius_changes_only_the_registered_geometry():
    saved, receipt, manifest = inputs()
    config = validate_parent(saved, receipt, manifest)
    assert config.pop('training_radius_values') == [.045,.05,.055]
    assert config == saved['configuration']
    assert saved['safe_timeouts'] == 846


def test_reject_best_node_other_seed_and_erased_failure_counts():
    saved, receipt, manifest = inputs()
    for field, value in [('steps',240000), ('training_seed',41029), ('safe_timeouts',0)]:
        changed = deepcopy(saved)
        changed[field] = value
        with pytest.raises(ValueError):
            validate_parent(changed, receipt, manifest)
