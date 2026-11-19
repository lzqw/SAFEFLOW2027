"""Native three-arm150K entry point;requires separately pinned execution admission.

No queue creation, auto parent selection, independent test or physical hardware.
Imports the original comparison learner;no change to its update mathematics.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform

from scripts.paper_crax_payload.native_p5_adaptation import actor_digest
from scripts.paper_crax_payload.prepare_reach_crossseed_v12 import validate_resolved,sha256
from scripts.paper_crax_payload.train_native_p5_adaptation import train
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import evaluate as evaluate_native,TrainingTracker
from scripts.paper_crax_payload.train_piper_comparison_v12 import attach as attach_original,ComparedPiperLearner
from scripts.paper_crax_payload.evaluate_piper_deployment_modes_v12 import deployment_mode


FORMAT='native_reach_crossseed_three_arm150K_v12_full'


class CrossseedPiperLearner(ComparedPiperLearner):
    def state_dict(self):
        return super().state_dict()|dict(reach_crossseed_v12=deepcopy(self.crossseed_identity))

    def load_state_dict(self,state):
        if state.get('reach_crossseed_v12')!=self.crossseed_identity:
            raise ValueError('Retain the exact native seed,parent,150K recipe and deployment comparison')
        super().load_state_dict(state)


def attach(learner,plan,line,arm):
    validate_resolved(plan)
    attach_original(learner,plan,line,arm)
    learner.__class__=CrossseedPiperLearner
    learner.crossseed_identity=dict(seed=plan['compiled_training_seed'],template_sha256=plan['template_sha256'],
        parent_sha256=plan['lines'][line]['parent_sha256'],new_training_budget=150000,
        modes=deepcopy(plan['same_mode_final_deployment_review']),independent_test_used=False)
    return learner


def validate(plan,args):
    line=validate_resolved(plan)
    admission=json.loads(args.execution_registration.read_text())
    jobs={(j['seed'],j['arm']) for j in admission['jobs']}
    expected={(seed,arm) for seed in (41028,41029) for arm in plan['arms']}
    root=Path(args.data_root).resolve(strict=True)
    if (args.resume or args.line!='piper_reach' or args.arm not in plan['arms']
            or args.parent.resolve()!=Path(line['parent_path']).resolve()
            or admission['status']!='admitted_fixed_crossseed900K_development'
            or admission['template_sha256']!=plan['template_sha256']
            or admission['runner_source_commit']!=args.source_commit
            or admission['runner_sha256']!=sha256(Path(__file__))
            or admission['new_training_steps']!=900000 or jobs!=expected or len(admission['jobs'])!=6
            or admission['independent_test_used'] is not False
            or admission['scientific_early_stopping'] is not False
            or admission['automatic_retry'] is not False
            or str(root)!=admission['data_roots_by_host'].get(platform.node())
            or sha256(args.plan)!=admission['plan_sha256_by_seed'][str(line['seed'])]):
        raise ValueError('Require separate exact six-job resource/source/recipe admission;preparation alone cannot launch')
    # Parent loading stays in the original strict driver:full SHA/native dimensions,
    # inherited optimizers/replay/dual, task and CUDA generator0 must all match.


def evaluate(learner,config,assets,task,episodes,seed_start):
    age=learner.transitions-learner.adaptation['parent_steps']
    modes=('sampled','zero_noise') if age in (100000,150000) else ('sampled',)
    before=(learner.transitions,learner.updates,learner.actor_updates,learner.safety_multiplier,
        actor_digest(learner.actor),actor_digest(learner.target_actor))
    results=[]
    for mode in modes:
        with deployment_mode(learner,mode):
            result=evaluate_native(learner,config,assets,task,episodes,seed_start)
        for group in result['conditions']:
            group['deployment_mode']=mode
        results.append(result)
    after=(learner.transitions,learner.updates,learner.actor_updates,learner.safety_multiplier,
        actor_digest(learner.actor),actor_digest(learner.target_actor))
    if before!=after:
        raise ValueError('Frozen development mode evaluation mutated native models,counters or dual')
    combined=deepcopy(results[0]);combined['conditions']=[g for r in results for g in r['conditions']]
    combined.update(protocol='v12_crossseed_same_mode_development',deployment_modes=list(modes),
        new_adaptation_steps=age,compiled_training_seed=learner.crossseed_identity['seed'],
        fixed_node_not_best_checkpoint=True,independent_test_used=False,
        training_risk_changed_by_evaluation=False,
        condition_identity='condition+deployment_mode;all arms receive the same mode schedule',
        new_simulation_controls=sum(row['length'] for g in combined['conditions'] for row in g['records']))
    return combined


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','parent','assets-receipt','output','data-root','execution-registration'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--line',default='piper_reach')
    p.add_argument('--source-commit',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--cuda-memory-fraction',type=float,default=.025)
    args=p.parse_args();args.resume=None
    train(args,protocol_validator=validate,adaptation_attacher=attach,format_name=FORMAT,
        piper_evaluator=evaluate,tracker_factory=TrainingTracker)


if __name__=='__main__':
    main()
