"""One-factor native Piper boundary activation or quarter-strength warm-up."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble
from scripts.paper_crax_payload.train_native_p5_adaptation import train
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import (
    attach as attach_warmup, PiperWarmupLearner, evaluate, TrainingTracker)

FORMAT='native_piper_p5_refinement_v12_full'
PLAN=Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_piper_round2.json'


class ActivationMargin(LocalMarginEnsemble):
    def forward(self, obs, action):
        return super().forward(obs,action)+self.activation_offset_m


class PiperRefinementLearner(PiperWarmupLearner):
    def metrics(self):
        return super().metrics()|dict(activation_offset_m=self.refinement['activation_offset_m'])

    def state_dict(self):
        return super().state_dict()|dict(native_piper_refinement_v12=deepcopy(self.refinement))

    def load_state_dict(self,state):
        if state.get('native_piper_refinement_v12')!=self.refinement:
            raise ValueError('Only identical activation offset and ramp may resume')
        super().load_state_dict(state)


def attach(learner,plan,line,arm):
    cfg=deepcopy(plan)
    cfg['candidates']={k:{n:v[n] for n in ('ramp_steps','initial_fraction')} for k,v in plan['candidates'].items()}
    attach_warmup(learner,cfg,line,arm)
    learner.__class__=PiperRefinementLearner
    learner.refinement=deepcopy(plan['candidates'][arm])
    if learner.refinement['activation_offset_m']:
        if type(learner.margin_geometry) is not LocalMarginEnsemble or learner.geometry.margin is not learner.margin_geometry:
            raise ValueError('Expected native shared EMA geometry ensemble')
        learner.margin_geometry.__class__=ActivationMargin
        learner.margin_geometry.activation_offset_m=learner.refinement['activation_offset_m']
    return learner


def validate(plan,args):
    if plan!=json.loads(PLAN.read_text()) or args.line!='piper_reach' or args.arm not in plan['arms'] or args.resume:
        raise ValueError('Use complete registered fresh round2 native100K forks')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('plan','parent','assets-receipt','output','data-root'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--line',default='piper_reach')
    p.add_argument('--source-commit',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--cuda-memory-fraction',type=float,default=.05)
    a=p.parse_args();a.resume=None
    train(a,protocol_validator=validate,adaptation_attacher=attach,format_name=FORMAT,
          piper_evaluator=evaluate,tracker_factory=TrainingTracker)


if __name__=='__main__':main()
