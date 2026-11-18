"""Fresh matched100K early-risk screen; no Piper checkpoint or old test reuse."""
import json
from pathlib import Path

from scripts.paper_crax_payload.direction_development import DirectionDevelopmentLearner
from scripts.paper_crax_payload.direction_warmup_v12 import DirectionWarmup, SafeReachWarmupLearner
from scripts.paper_crax_payload.evaluate_direction_annealing import evaluate
from scripts.paper_crax_payload.evaluate_direction_polyak import target_deployment
from scripts.paper_crax_payload.train_direction_development import main as run
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import TrainingTracker

FORMAT='safereach_fresh_early100k_screen_v12_full'
PLAN=Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_safereach_early.json'


class ScreenLearner(SafeReachWarmupLearner):
    @property
    def current_direction_weight(self):
        return self.warmup.weight(self.transitions)


class SafeReachTracker(TrainingTracker):
    def __init__(self,n):
        super().__init__(n,success_distance=.035)


def validate(config):
    if config != json.loads(PLAN.read_text()):
        raise ValueError('Use the registered fresh100K screen, not a historical resume or implicit retuning')


def build(config, arm, seed, device):
    if seed not in config['development_training_seeds'] or arm not in config['arms']:
        raise ValueError('Use registered matched fresh seed/arm')
    s=config['shared_learner']
    common=dict(device=device,actor_lr=s['actor_lr'],critic_lr=s['critic_lr'],
        critic_only_updates=arm.get('critic_only_updates',s['critic_only_updates']),
        actor_interval=s['actor_update_interval'],margin_ema_tau=config['geometry']['ema_tau'])
    if arm['component'] in ('iso','metric_only'):
        learner=DirectionDevelopmentLearner(component=arm['component'],rho=.25,
            normal_shrink=False,direction_weight=0.,**common)
        learner.current_direction_weight=0.
        return learner
    return ScreenLearner(DirectionWarmup(arm['ramp_steps'],arm['initial_fraction']),**common)


def deployed(learner,config,steps):
    with target_deployment(learner):
        result=evaluate(learner,config,steps)
    for group in result['conditions']:
        for record in group['records']:
            details=record['control_details']
            distances=[details[0]['distance_to_goal_before_m']]+[x['distance_to_goal_after_m'] for x in details]
            record['stall']=any(distances[i]-distances[i+10]<.001 for i in range(len(distances)-10))
            record['ordinary_terminal_task_success']=distances[-1]<.035
        group['stall_count']=sum(r['stall'] for r in group['records'])
    result.update(split='v12_fresh100k_screen_development_only',independent_test_used=False,
        deployment='sole_existing_target_actor_polyak_0.001',
        contact_response_gate='waived_by_user_not_evaluated',inference_timing='separate_frozen_policy_measurement')
    return result


if __name__=='__main__':
    run(validator=validate,builder=build,format_name=FORMAT,evaluator=deployed,tracker_factory=SafeReachTracker)
