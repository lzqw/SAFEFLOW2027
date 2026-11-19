from copy import deepcopy

import pytest

from scripts.paper_crax_payload.close_samehardware_v12 import terminal_state
from scripts.paper_crax_payload.summarize_matched_ablation_v12 import COUNTS, pre_decay_prefix, registered_protocols


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
    assert not terminal_state(dict(status='running', pending=0, live_native_pids=[123]))
    assert terminal_state(dict(status='completed', pending=0, live_native_pids=[]))
    with pytest.raises(RuntimeError, match='never retry'):
        terminal_state(dict(status='completed_with_failures'))
