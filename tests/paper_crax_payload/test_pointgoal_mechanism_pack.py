"""Synthetic contract fixtures only; never reported as measured mechanism data."""
import copy

import pytest

from scripts.paper_crax_payload.pointgoal_mechanism_pack import METHODS, SAMPLERS, prepare


def fixture():
    plan = dict(analysis_sha256='source', planned_total_controls=16704, frozen_final_selection_digest='freeze')
    receipt = dict(status='completed', physical_controls=16704, states=64, draws_per_state=32,
                   training_transitions=0, optimizer_steps=0, no_pool_reselection_after_cost_labels=True,
                   analysis_sha256='source')
    pool = dict(records=[dict(physical_state_sha256=f'fixture-{i}') for i in range(64)])
    selection = dict(freeze_digest='freeze', independent_test_used=False,
                     methods={m: dict(selected=dict(checkpoint_steps=8000000)) for m in METHODS})
    rows = [dict(method=m, sampler=s, checkpoint_steps=8000000, states=64, draws=32,
                 endpoint_vn=.02, endpoint_vt=.04) for m in METHODS for s in SAMPLERS]
    groups = [dict(method=m, sampler=s, states=[dict(state_index=i, native_cost_positive_fraction=float(i < 4))
                                               for i in range(64)]) for m in METHODS for s in SAMPLERS]
    return dict(plan=plan, receipt=receipt, results=dict(summaries=rows, per_state=groups),
                pool=pool, selection=selection)


def test_mixed_pool_does_not_imply_sampler_event_discrimination():
    data = fixture()
    rows, diagnostic = prepare(**data)
    assert len(rows) == 8 and rows[0]['endpoint_vn_over_vt'] == .5
    assert diagnostic['ea']['states_with_different_cost_event_rates'] == 0
    assert diagnostic['ea']['states_all_zero_every_sampler'] == 60
    changed = copy.deepcopy(data)
    changed['results']['per_state'][-1]['states'][0]['native_cost_positive_fraction'] = .5
    assert prepare(**changed)[1]['ea']['states_with_different_cost_event_rates'] == 1


def test_partial_historical_or_reselected_inputs_are_rejected():
    for key, value in (('physical_controls', 2208), ('states', 32), ('status', 'running')):
        data = fixture()
        data['receipt'][key] = value
        with pytest.raises(ValueError):
            prepare(**data)
    data = fixture()
    data['pool']['records'][1] = data['pool']['records'][0]
    with pytest.raises(ValueError):
        prepare(**data)
