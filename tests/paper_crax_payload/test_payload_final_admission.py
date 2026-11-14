from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.paper_crax_payload.payload_repair_independent import LABELS, admit_review


def fixture():
    plan=json.loads(Path('configs/paper_crax_payload/payload_repair_v3_memory_validation.json').read_text())
    candidates=[dict(label=label,steps=steps,full_checkpoint_sha256=str(i)*64)
                for i,label in enumerate(LABELS,1) for steps in plan['new_candidate_steps']]
    selection=dict(independent_test_used=False,candidates=candidates,
        selected={label:dict(candidate=next(c for c in candidates if c['label']==label),candidate_count=5)
                  for label in LABELS})
    hashes=dict(validation_plan_sha256='a'*64,validation_receipt_sha256='b'*64,
                validation_selection_sha256='c'*64)
    receipt=dict(status='stage1_validation_completed_pending_review',validation_candidates=20,
        validation_episodes=4000,stage1_training_receipts_each=100000,total_E01_training_including_failure=400325,
        conditional_second_stage_launched=False,independent_test_executed=False,
        plan_sha256=hashes['validation_plan_sha256'],selection_sha256=hashes['validation_selection_sha256'])
    review=dict(review_completed=True,decision='finalize_after_stage1',conditional_continuation=False,
        independent_test_labels_seen=False,reason='Validation-only scientific review recorded before any final test.',
        **hashes,selected_checkpoint_sha256={label:selection['selected'][label]['candidate']['full_checkpoint_sha256']
                                           for label in LABELS})
    return review,selection,receipt,plan,hashes


def test_prepared_pending_template_cannot_open_test():
    args=list(fixture())
    args[0]=json.loads(Path('configs/paper_crax_payload/payload_repair_v3_review_PENDING.json').read_text())
    with pytest.raises(ValueError,match='review'):
        admit_review(*args)


def test_prefix_or_changed_identity_cannot_open_test():
    args=list(fixture())
    args[2]['validation_episodes']=3999
    with pytest.raises(ValueError,match='complete'):
        admit_review(*args)
    args=list(fixture())
    args[0]['selected_checkpoint_sha256']['EA-repair']='f'*64
    with pytest.raises(ValueError,match='identity'):
        admit_review(*args)


def test_complete_metadata_passes_but_continuation_or_test_exposure_does_not():
    args=fixture()
    admit_review(*args)
    for key in ['conditional_continuation','independent_test_labels_seen']:
        changed=deepcopy(args)
        changed[0][key]=True
        with pytest.raises(ValueError,match='review'):
            admit_review(*changed)
