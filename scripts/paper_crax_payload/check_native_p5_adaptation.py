"""One targeted CPU check on an exact parent; zero environment interactions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from scripts.paper_crax_payload.native_p5_adaptation import (
    attach_adaptation, build_native, scheduled_lr, validate_parent, actor_digest, target_deployment)
from scripts.paper_crax_payload.train_native_p5_adaptation import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--line', required=True)
    p.add_argument('--parent', type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text()); line = plan['lines'][args.line]
    parent = torch.load(args.parent or line['parent_path'], map_location='cpu', weights_only=False)
    validate_parent(parent, plan, args.line, sha256(args.parent or line['parent_path']))
    digests = []
    for arm in plan['arms']:
        learner = build_native(parent['configuration'], line, 'cpu')
        learner.load_state_dict(parent['learner'])
        before = (actor_digest(learner.actor), actor_digest(learner.target_actor))
        attach_adaptation(learner, plan, args.line, arm)
        assert before == (actor_digest(learner.actor), actor_digest(learner.target_actor))
        digests.append(before)
        assert learner.actor_opt.param_groups[0]['lr'] == plan['actor_lr']['initial']
        online = learner.actor
        try:
            with target_deployment(learner):
                assert learner.actor is learner.target_actor
                raise RuntimeError('deliberate context restoration test')
        except RuntimeError:
            pass
        assert learner.actor is online
        state = deepcopy(learner.state_dict())
        other = build_native(parent['configuration'], line, 'cpu')
        other.load_state_dict(parent['learner']); attach_adaptation(other, plan, args.line, arm)
        other.load_state_dict(state)
        wrong = deepcopy(state); wrong['native_p5_adaptation_v11']['arm'] = 'not_this_arm'
        try:
            other.load_state_dict(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError('strict arm guard missing')
        batch = {k:torch.as_tensor(v[:8]) for k,v in parent['replay']['data'].items()}
        # Parent has mature critic/actor counters. One discarded CPU update
        # checks the real native34D/45D backward path, not a synthetic learner.
        learner.update(batch)
        assert all(torch.isfinite(v).all() for v in learner.actor.parameters())
        learner.set_transitions(line['parent_steps']+80000)
        assert abs(learner.actor_opt.param_groups[0]['lr']-3e-5)<1e-12
    assert digests[0] == digests[1]
    assert scheduled_lr(0, plan['actor_lr']) == 1e-4
    print(json.dumps(dict(status='passed', line=args.line, checks=[
        'exact_native_parent', 'paired_network_identity', 'native_backward_both_arms',
        'new_step_lr_schedule', 'strict_resume_arm', 'target_context_restoration'],
        new_training_environment_steps=0, physical_robot_interactions=0)))


if __name__=='__main__':
    main()
