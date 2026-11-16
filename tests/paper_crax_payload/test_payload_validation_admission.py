import json

import pytest

from scripts.paper_crax_payload.payload_repair_validation import training_state


def test_only_full_equal_new_receipt_counts_as_completed(tmp_path):
    run=tmp_path/'runs'/'example';run.mkdir(parents=True)
    job=dict(run='example',label='example',method='s_ea',initial_steps=240000)
    plan=dict(new_training_target_each=100000)
    value=dict(status='completed',method='s_ea',initial_steps=240000,newly_collected_transitions=99000)
    (run/'receipt.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):
        training_state(tmp_path,plan,job)
    value['newly_collected_transitions']=100000
    (run/'receipt.json').write_text(json.dumps(value))
    assert training_state(tmp_path,plan,job)=='completed'


def test_interrupted_worker_is_not_silently_restarted_or_accepted(tmp_path):
    run=tmp_path/'runs'/'example';run.mkdir(parents=True)
    (run/'status.json').write_text(json.dumps(dict(state='engineering_interruption')))
    with pytest.raises(RuntimeError,match='interrupted'):
        training_state(tmp_path,{},dict(run='example',label='example'))


def test_wait_requires_a_real_queue_handle_not_only_a_stale_file(tmp_path,monkeypatch):
    (tmp_path/'queue.json').write_text(json.dumps(dict(pid=123)))
    plan=dict(queue_state='queue.json',queue_expected_command='finite_queue')
    job=dict(run='example',label='example')
    monkeypatch.setattr('scripts.paper_crax_payload.payload_repair_validation.native_live',lambda *a:True)
    assert training_state(tmp_path,plan,job)=='waiting_for_finite_queue_admission'
    monkeypatch.setattr('scripts.paper_crax_payload.payload_repair_validation.native_live',lambda *a:False)
    with pytest.raises(RuntimeError,match='handle is absent'):
        training_state(tmp_path,plan,job)
