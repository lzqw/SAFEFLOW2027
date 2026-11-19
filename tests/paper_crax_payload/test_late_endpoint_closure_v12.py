from copy import deepcopy
import json

import pytest

from scripts.paper_crax_payload.close_late_endpoint_v12 import check_suffix
from scripts.paper_crax_payload.train_late_endpoint_v12 import FORMAT, PLAN


def test_suffix_accounting_rejects_double_counting_and_changed_prefix():
    config=json.loads(PLAN.read_text());arm=config['arms'][1]
    parent=dict(training_violations=295,safe_successes=2606,safe_timeouts=356)
    provenance=dict(source_checkpoint_sha256=config['source_checkpoint_sha256'],prefix_steps=100000,
        prefix_not_new_training=True,prefix_counts=parent,prefix_evaluation_controls=1000)
    manifest=dict(configuration=config,arm=arm,host='447',format=FORMAT,
        source_commit='7eda9c8f',fork_provenance=provenance)
    receipt=dict(status='completed',engineering_only=False,steps=250000,initial_steps=100000,
        new_training_transitions=150000,training_seed=2034,arm=arm,source_commit='7eda9c8f',
        independent_endpoint_test_executed=False,evaluations=config['evaluation']['milestones'],
        fork_provenance=provenance,evaluation_controls=3000,new_evaluation_controls=2000,
        **{k:v+10 for k,v in parent.items()},**{'new_'+k:10 for k in parent})
    check_suffix(receipt,manifest,config,arm,parent)
    for key,value in (('new_training_transitions',250000),('new_safe_successes',2616),
                      ('new_evaluation_controls',3000),('independent_endpoint_test_executed',True)):
        bad=deepcopy(receipt);bad[key]=value
        with pytest.raises(ValueError):check_suffix(bad,manifest,config,arm,parent)
    bad=deepcopy(receipt);bad['fork_provenance']['prefix_counts']['training_violations']=294
    with pytest.raises(ValueError):check_suffix(bad,manifest,config,arm,parent)
