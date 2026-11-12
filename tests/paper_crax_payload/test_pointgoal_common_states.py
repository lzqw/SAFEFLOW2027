import numpy as np
import pytest

from scripts.paper_crax_payload.pointgoal_common_states import endpoint_projection,select_strata


def test_stratification_keeps_unsafe_and_interior_without_duplicate_padding():
    margins=np.array([.01,.03]+[-.05]*10+[-.8]*20)
    fingerprints=[f'state{i}' for i in range(len(margins))]
    selected=select_strata(margins,fingerprints,set(),123)
    assert len(selected)==len(set(selected))==16
    assert 0 in selected and 1 in selected
    assert sum(margins[i]<-.15 for i in selected)==8
    assert select_strata(margins,fingerprints,set(),123)==selected
    with pytest.raises(ValueError):select_strata(margins,fingerprints,set(fingerprints),123)


def test_action_covariance_uses_actual_action_dimension_and_keeps_undefined_axes():
    actions=np.array([[1.,0.],[-1.,0.],[0.,2.],[0.,-2.]])
    metric=endpoint_projection(actions,np.array([1.,0.]))
    assert metric['vn']==pytest.approx(2/3)
    assert metric['vt']==pytest.approx(8/3)
    undefined=endpoint_projection(actions,np.zeros(2))
    assert undefined['trace']==pytest.approx(10/3)
    assert undefined['vn'] is None and undefined['vt'] is None


def test_duplicate_physical_fingerprints_are_not_used_as_distinct_states():
    margins=np.array([.01]+[-.05]*15+[-.8]*20)
    fingerprints=['same','same']+[f'state{i}' for i in range(34)]
    selected=select_strata(margins,fingerprints,set(),123)
    assert len(selected)==16
    assert len({fingerprints[i] for i in selected})==16
