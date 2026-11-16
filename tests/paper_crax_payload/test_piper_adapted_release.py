from scripts.paper_crax_payload.piper_adapted_release import freeze_record


def test_fixed_adapted_endpoint_never_becomes_hardware_authorization():
    score=dict(risk_qualified=False,task_progress_observed=True,
        macro_safe_success=.9,per_condition_violation=[0.,.3,0.])
    record=freeze_record('EA',400000,score)
    assert record['selected_checkpoint_steps']==400000
    assert record['candidate_count']==1
    assert record['status']=='pareto_tradeoff_not_risk_qualified'
    assert not record['independent_test_used_for_selection']
    assert not record['physical_deployment_eligible']


def test_qualified_but_stagnant_is_not_reported_as_useful_success():
    record=freeze_record('Iso',400000,dict(risk_qualified=True,task_progress_observed=False))
    assert record['status']=='risk_qualified_but_stagnant'
    assert not record['physical_deployment_eligible']
