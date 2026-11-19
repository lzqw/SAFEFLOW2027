"""Paired native Piper direction-loss-only taper; full parents, original geometry."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.paper_crax_payload.direction_development import outward_loss
from scripts.paper_crax_payload.direction_taper_v12 import DirectionTaperActor, SPEC
from scripts.paper_crax_payload.train_native_p5_adaptation import train
from scripts.paper_crax_payload.train_piper_comparison_v12 import attach as attach_comparison, ComparedPiperLearner
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import evaluate, TrainingTracker

PLANS = {name: Path(__file__).resolve().parents[2]/('configs/paper_crax_payload/simulation_v12_piper_direction_taper_'+name+'.json')
         for name in ('reach','payload')}
FORMAT = 'native_piper_direction_loss_taper_v12_full'


class PiperDirectionTaperLearner(ComparedPiperLearner):
    def actor_regularizer(self, trace):
        original = super().actor_regularizer(trace)
        if not self.piper_direction_taper['enabled']:
            return original
        loss = outward_loss(trace['velocities'],trace['normals'],trace['direction_weights'],1./self.actor.num_steps)
        self._last_metrics.update(direction_loss=loss.detach(),untapered_direction_loss=original.detach(),
            direction_taper_supported_fraction=(trace['direction_margin_m']>-.01).float().mean(),
            mean_direction_loss_weight=trace['direction_weights'].mean())
        return loss

    def state_dict(self):
        return super().state_dict() | dict(native_piper_direction_taper_v12=deepcopy(self.piper_direction_taper))

    def load_state_dict(self, state):
        if state.get('native_piper_direction_taper_v12') != self.piper_direction_taper:
            raise ValueError('Resume only the same native Piper taper task/control')
        super().load_state_dict(state)


def attach(learner, plan, line, arm):
    attach_comparison(learner,plan,line,arm)
    learner.__class__ = PiperDirectionTaperLearner
    learner.piper_direction_taper = dict(enabled=plan['taper_enabled'][arm], specification=deepcopy(SPEC))
    if learner.piper_direction_taper['enabled']:
        for actor in (learner.actor,learner.target_actor):
            actor.__class__ = DirectionTaperActor
    return learner


def validate(plan, args):
    key = args.line.removeprefix('piper_')
    if (key not in PLANS or plan != json.loads(PLANS[key].read_text()) or args.resume
            or args.arm not in plan['arms'] or plan['direction_taper'] != SPEC
            or any(c['ramp_steps'] != 0 for c in plan['candidates'].values())):
        raise ValueError('Use exact registered fixed-direction native taper pair')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','parent','assets-receipt','output','data-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--line',required=True); p.add_argument('--arm',required=True)
    p.add_argument('--source-commit',required=True); p.add_argument('--device',default='cuda:0')
    p.add_argument('--cuda-memory-fraction',type=float,default=.025)
    args = p.parse_args(); args.resume = None
    train(args,protocol_validator=validate,adaptation_attacher=attach,format_name=FORMAT,
        piper_evaluator=evaluate,tracker_factory=TrainingTracker)
