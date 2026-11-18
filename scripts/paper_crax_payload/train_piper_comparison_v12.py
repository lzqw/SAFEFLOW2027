"""Native Reach combination and explicit same-parent Metric-only comparison."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.paper_crax_payload.component_models import ComponentActor
from scripts.paper_crax_payload.train_native_p5_adaptation import train
from scripts.paper_crax_payload.train_piper_refinement_v12 import (
    attach as attach_refinement, PiperRefinementLearner)
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import evaluate, TrainingTracker

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_piper_round3.json'
FORMAT = 'native_piper_explicit_metric_comparison_v12_full'


class ComparedPiperLearner(PiperRefinementLearner):
    def actor_regularizer(self, trace):
        if self.comparison['component'] == 'metric_only':
            return 0.
        return super().actor_regularizer(trace)

    def metrics(self):
        value = super().metrics()
        if self.comparison['component'] == 'metric_only':
            value['direction_weight_effective'] = 0.
        return value

    def state_dict(self):
        return super().state_dict() | dict(native_comparison_v12=deepcopy(self.comparison))

    def load_state_dict(self, state):
        if state.get('native_comparison_v12') != self.comparison:
            raise ValueError('Retain explicit native sampling/energy comparison identity')
        super().load_state_dict(state)


def attach(learner, plan, line, arm):
    cfg = deepcopy(plan)
    cfg['candidates'] = {k: {n: v[n] for n in ('ramp_steps', 'initial_fraction', 'activation_offset_m')}
                         for k, v in plan['candidates'].items()}
    attach_refinement(learner, cfg, line, arm)
    learner.__class__ = ComparedPiperLearner
    learner.comparison = dict(arm=arm, component=plan['candidates'][arm]['component'],
                              metric_definition='isotropic_sampling_directional_energy_no_outward_loss')
    if learner.comparison['component'] == 'metric_only':
        for actor in (learner.actor, learner.target_actor):
            actor.__class__ = ComponentActor
            actor.sampling_directional, actor.energy_directional = False, True
    return learner


def validate(plan, args):
    if (plan != json.loads(PLAN.read_text()) or args.line != 'piper_reach'
            or args.arm not in plan['arms'] or args.resume):
        raise ValueError('Use the exact registered native round3 comparison')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'parent', 'assets-receipt', 'output', 'data-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', required=True); p.add_argument('--line', default='piper_reach')
    p.add_argument('--source-commit', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--cuda-memory-fraction', type=float, default=.05)
    args = p.parse_args(); args.resume = None
    train(args, protocol_validator=validate, adaptation_attacher=attach, format_name=FORMAT,
          piper_evaluator=evaluate, tracker_factory=TrainingTracker)


if __name__ == '__main__':
    main()
