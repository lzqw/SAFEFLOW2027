"""Prepared one-factor5mm geometry-temperature forks; no implicit launch."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.paper_crax_payload.train_native_p5_adaptation import train
from scripts.paper_crax_payload.train_piper_comparison_v12 import attach as attach_comparison, ComparedPiperLearner
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import evaluate, TrainingTracker

PLANS = {name: Path(__file__).resolve().parents[2]/
         ('configs/paper_crax_payload/simulation_v12_localized_'+name+'.json')
         for name in ('piper_reach', 'piper_payload')}
FORMAT = 'native_piper_localized_geometry_v12_full'


class LocalizedGeometryLearner(ComparedPiperLearner):
    def state_dict(self):
        return super().state_dict() | dict(localized_geometry_v12=deepcopy(self.localization))

    def load_state_dict(self, state):
        if state.get('localized_geometry_v12') != self.localization:
            raise ValueError('Only identical localized geometry temperature may resume')
        super().load_state_dict(state)

    def metrics(self):
        return super().metrics() | dict(geometry_temperature_m=self.margin_geometry.temperature)


def attach(learner, plan, line, arm):
    cfg = deepcopy(plan)
    cfg['candidates'] = {k: {n: v[n] for n in ('component', 'ramp_steps', 'initial_fraction', 'activation_offset_m')}
                         for k, v in plan['candidates'].items()}
    attach_comparison(learner, cfg, line, arm)
    if learner.margin_geometry.temperature != .02 or learner.geometry.margin is not learner.margin_geometry:
        raise ValueError('Require the registered shared20mm EMA geometry reference')
    temperature = plan['candidates'][arm]['geometry_temperature_m']
    if temperature not in (.005, .02):
        raise ValueError('Only original20mm or the single prepared5mm candidate is admitted')
    learner.__class__ = LocalizedGeometryLearner
    learner.localization = dict(geometry_temperature_m=temperature,
        online_supervision_changed=False, true_safety_distance_changed=False)
    learner.margin_geometry.temperature = temperature
    return learner


def validate(plan, args):
    if (args.line not in PLANS or plan != json.loads(PLANS[args.line].read_text()) or args.resume
            or args.arm not in plan['arms']):
        raise ValueError('Use only the exact prepared localized-geometry task plan')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'parent', 'assets-receipt', 'output', 'data-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--line', required=True); p.add_argument('--arm', required=True)
    p.add_argument('--source-commit', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--cuda-memory-fraction', type=float, default=.05)
    args = p.parse_args(); args.resume = None
    train(args, protocol_validator=validate, adaptation_attacher=attach, format_name=FORMAT,
          piper_evaluator=evaluate, tracker_factory=TrainingTracker)
