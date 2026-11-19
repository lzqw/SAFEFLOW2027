"""One fixed post100K residual curriculum;strict real prefix,shared matched fork."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

from scripts.paper_crax_payload.endpoint_residual_v12 import EndpointRecordedActor,endpoint_residual
from scripts.paper_crax_payload.train_matched_ablation_v12 import AblationLearner,deployed
from scripts.paper_crax_payload.train_direction_development import main as run
from scripts.paper_crax_payload.train_safereach_early_v12 import SafeReachTracker


PROJECT=Path(__file__).resolve().parents[2]
PLAN=PROJECT/'configs/paper_crax_payload/simulation_v12_late_endpoint.json'
ROOT=Path('/home/carla/4T/SAFEFLOW2027_SIMULATION_V12')
FORMAT='safereach_shared100K_prefix_late_endpoint_v12_full'
SPEC=dict(start_steps=100000,full_steps=150000,coefficient=1.,scale_m=.015)


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def ramp(steps):
    return min(1.,max(0.,(steps-100000)/50000))


class LateEndpointLearner(AblationLearner):
    def actor_regularizer(self,trace):
        original=super().actor_regularizer(trace)
        weight=ramp(self.transitions) if self.late_endpoint['enabled'] else 0.
        if weight==0.:
            return original
        margin=self.margin_geometry(trace['endpoint_observation'],trace['endpoint_action']).mean(0)
        loss=endpoint_residual(margin)
        self._last_metrics.update(late_endpoint_coefficient=torch.as_tensor(weight,device=self.device),
            endpoint_residual_loss=loss.detach(),endpoint_residual_effective=(weight*loss).detach())
        return original+weight*loss

    def state_dict(self):
        return super().state_dict()|dict(late_endpoint_v12=deepcopy(self.late_endpoint))

    def load_state_dict(self,state):
        if state.get('late_endpoint_v12')!=self.late_endpoint:
            raise ValueError('Resume only identical late curriculum/control and genuine prefix')
        super().load_state_dict(state)


def attach(learner,enabled,parent_sha):
    # No constructor,RNG draw,new parameter,optimizer,or reinterpretation of old load.
    learner.__class__=LateEndpointLearner
    learner.late_endpoint=dict(enabled=enabled,specification=deepcopy(SPEC),parent_sha256=parent_sha)
    for actor in (learner.actor,learner.target_actor):actor.__class__=EndpointRecordedActor
    return learner


def validate(config):
    if config!=json.loads(PLAN.read_text()) or config['late_endpoint']!=SPEC:
        raise ValueError('One fixed curriculum,not a coefficient/ramp scan')
    source=json.loads((PROJECT/config['source_plan']).read_text())
    for key in ('task_version','num_envs','shared_learner','geometry','deployment','evaluation'):
        if source[key]!=config[key]:raise ValueError('Keep the entire original native scientific recipe')


def build(config,arm,seed,device):
    if seed!=2034 or arm not in config['arms']:raise ValueError('Registered original training seed and two suffix arms only')
    source=json.loads((PROJECT/config['source_plan']).read_text())
    original=next(a for a in source['arms'] if a['name']==config['source_arm'])
    s=config['shared_learner']
    learner=AblationLearner(original,device=device,actor_lr=s['actor_lr'],critic_lr=s['critic_lr'],
        critic_only_updates=s['critic_only_updates'],actor_interval=s['actor_update_interval'],margin_ema_tau=.005)
    return attach(learner,arm['late_endpoint'],config['source_checkpoint_sha256'])


def initialize_fork(args,config,arm,learner):
    validate(config)
    path=Path(config['source_checkpoint'])
    if (args.fork_from.resolve()!=path.resolve() or sha256(path)!=config['source_checkpoint_sha256']
            or not args.output.resolve().is_relative_to(ROOT.resolve(strict=True))
            or args.seed!=2034 or torch.device(args.device).type!='cuda'
            or torch.cuda.device_count()!=1):
        raise ValueError('Exact original100K checkpoint and one shared4080 GPU;new data-disk output only')
    saved=torch.load(path,map_location=learner.device,weights_only=False)
    source=json.loads((PROJECT/config['source_plan']).read_text())
    original=next(a for a in source['arms'] if a['name']==config['source_arm'])
    manifest=json.loads((path.parent/'run_config.json').read_text())
    if (saved['format']!=config['source_format'] or saved['configuration']!=source or saved['arm']!=original
            or saved['steps']!=100000 or saved['seed']!=2034 or saved['source_commit']!=config['source_commit']
            or manifest['engineering_only'] or manifest['initial_network_sha256']!='6eed16a4650593305c7ecb157ffe6d61c990b7c2fa1dad307a1a8025b6f31e6a'
            or len(saved['rng']['cuda'])!=1 or saved['replay']['data']['obs'].shape[-1]!=25
            or saved['violations']!=295 or saved['successes']!=2606 or saved['timeouts']!=356):
        raise ValueError('Do not invent,retune,rename or replace the genuine observed first100K prefix')
    # Strict original native load FIRST;only then attach the explicit new factor.
    enabled=arm['late_endpoint'];learner.__class__=AblationLearner
    learner.load_state_dict(saved['learner'])
    attach(learner,enabled,config['source_checkpoint_sha256'])
    prefix=[json.loads(s) for s in (path.parent/'progress.jsonl').read_text().splitlines()
        if json.loads(s)['steps']<=100000]
    if [r['steps'] for r in prefix]!=list(range(1000,100001,1000)):
        raise ValueError('Every actual1K prefix node required')
    (args.output/'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prefix))
    evaluations=[];controls=0;hashes=[]
    for step in (25000,50000,75000,100000):
        original_path=path.parent/f'evaluation_{step}.json';raw=json.loads(original_path.read_text())
        if raw['milestone']!=step or raw['independent_test_used'] is not False:
            raise ValueError('Preserve all four actual original DEV prefix nodes')
        shutil.copy2(original_path,args.output/original_path.name)
        controls+=raw['native_controls'];evaluations.append(step)
        hashes.append(dict(step=step,sha256=sha256(original_path)))
    provenance=dict(prefix_steps=100000,source_checkpoint=str(path),source_checkpoint_sha256=sha256(path),
        source_commit=config['source_commit'],source_format=config['source_format'],
        prefix_counts={k:prefix[-1][k] for k in ('training_violations','safe_successes','safe_timeouts')},
        prefix_evaluation_controls=controls,prefix_evaluations=hashes,
        prefix_training_elapsed_seconds=saved['elapsed_seconds'],
        original_trajectory_file=str(path.parent/'trajectories.jsonl.gz'),
        prefix_not_new_training=True,same_training_seed_not_independent_confirmation=True,
        physical_restore_scope='observed native state;tolerance2e-6;warmstart caches not guaranteed bitwise historical')
    saved['evaluations']=evaluations;saved['evaluation_controls']=controls;saved['elapsed_seconds']=0.
    return saved,provenance


def evaluate(learner,config,steps):
    result=deployed(learner,config,steps)
    result.update(split='v12_shared_real100K_prefix_late_curriculum_development',
        late_endpoint=deepcopy(learner.late_endpoint),new_postprefix_steps=steps-100000)
    return result


if __name__=='__main__':
    if not any(arg in ('--fork-from','--resume') or arg.startswith(('--fork-from=','--resume=')) for arg in sys.argv[1:]):
        raise ValueError('This experiment requires the real100K full fork;fresh relabeled training is forbidden')
    run(validator=validate,builder=build,format_name=FORMAT,evaluator=evaluate,
        tracker_factory=SafeReachTracker,fork_initializer=initialize_fork)
