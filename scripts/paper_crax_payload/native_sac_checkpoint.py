"""Pinned official SAC-Lag with HOST-boundary full-state instrumentation.

No official file is edited and no learner loss/optimizer or physics is replaced.
Only trusted checkpoints created by this program may be unpickled.  The older
policy/normalizer-only reference is intentionally NOT resumable by this runner.
"""
import argparse
import ast
from collections import defaultdict, deque
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np


CRAX_COMMIT='63504b540705be98024abd675e646ff5e046c6c2'
TRAIN_SHA256='e4d73d7d16e06ef6d50cd907b75ceba2622f108594a7b2cc483d845c1910baa4'
CHECKPOINT_FORMAT='SAFEFlow-native-SAC-Lag-full-host-epoch-v1'
CORE_FUNCTIONS=('sgd_step','get_experience','training_step','prefill_replay_buffer',
                'training_epoch','training_epoch_with_timing')


def _statements(source):
    return ast.parse(source).body


def instrument_source(source):
    """Return a train AST with narrowly enumerated HOST-only modifications."""
    if hashlib.sha256(source.encode()).hexdigest()!=TRAIN_SHA256:
        raise ValueError('Official SAC-Lag source changed; refusing a silent unpinned patch')
    original=ast.parse(source)
    train=deepcopy(next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='train'))
    before={n.name:ast.dump(n,include_attributes=False) for n in train.body
            if isinstance(n,ast.FunctionDef) and n.name in CORE_FUNCTIONS}
    if set(before)!=set(CORE_FUNCTIONS):
        raise ValueError('Unexpected native inner-function layout')
    # Preserve original epoch granularity on resume, including extensions.
    plan_index=next(i for i,n in enumerate(train.body) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='num_training_steps_per_epoch' for t in n.targets))
    train.body[plan_index+1:plan_index+1]=_statements('''
num_training_steps_per_epoch, num_evals_after_init = _paper_checkpoint.epoch_plan(
    num_training_steps_per_epoch,num_evals_after_init,num_timesteps,
    num_prefill_env_steps,env_steps_per_actor_step,local_devices_to_use,jax.process_count())
''')
    # Restore after native objects are instantiated, before any initial eval or
    # replay prefill.  Fresh initializations consume no actual training steps.
    metrics_index=next(i for i,n in enumerate(train.body) if isinstance(n,ast.Assign)
                       and any(isinstance(t,ast.Name) and t.id=='metrics' for t in n.targets))
    train.body[metrics_index+1:metrics_index+1]=_statements('''
(training_state,env_state,buffer_state,local_key,training_walltime,
 _paper_start_epoch,_paper_start_step,metrics) = _paper_checkpoint.restore(
    training_state,env_state,buffer_state,local_key,evaluator,metrics_aggregator,replay_buffer)
''')
    initial_eval=next(n for n in train.body if isinstance(n,ast.If)
                      and 'num_evals > 1' in ast.unparse(n.test))
    initial_eval.test=ast.BoolOp(op=ast.And(),values=[initial_eval.test,
        ast.UnaryOp(op=ast.Not(),operand=ast.Attribute(value=ast.Name(id='_paper_checkpoint',ctx=ast.Load()),
                                                     attr='resume_available',ctx=ast.Load()))])
    prefill_end=next(i for i,n in enumerate(train.body) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='training_walltime' for t in n.targets))
    prefill_start=max(i for i,n in enumerate(train.body[:prefill_end]) if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='t' for t in n.targets))
    train.body[prefill_start:prefill_end+1]=[ast.If(test=ast.UnaryOp(op=ast.Not(),
        operand=ast.Attribute(value=ast.Name(id='_paper_checkpoint',ctx=ast.Load()),attr='resume_available',ctx=ast.Load())),
        body=train.body[prefill_start:prefill_end+1],orelse=[])]
    step_assignment=next(n for n in train.body if isinstance(n,ast.Assign)
                         and any(isinstance(t,ast.Name) and t.id=='current_step' for t in n.targets))
    step_assignment.value=ast.Name(id='_paper_start_step',ctx=ast.Load())
    epoch_loop=next(n for n in train.body if isinstance(n,ast.For)
                    and isinstance(n.iter,ast.Call) and ast.unparse(n.iter)=='range(num_evals_after_init)')
    epoch_loop.iter.args.insert(0,ast.Name(id='_paper_start_epoch',ctx=ast.Load()))
    # Evaluation remains the official evaluator, only at five declared eligible
    # checkpoints.  Additional diagnostic epochs in an extension cannot create
    # extra implicit candidate opportunities.
    epoch_eval=next(n for n in epoch_loop.body if isinstance(n,ast.If)
                    and 'num_evals > 0' in ast.unparse(n.test))
    epoch_eval.test=ast.BoolOp(op=ast.And(),values=[epoch_eval.test,
        ast.Call(func=ast.Attribute(value=ast.Name(id='_paper_checkpoint',ctx=ast.Load()),attr='is_candidate_epoch',ctx=ast.Load()),
                 args=[ast.BinOp(left=ast.Name(id='_',ctx=ast.Load()),op=ast.Add(),right=ast.Constant(1))],keywords=[])])
    epoch_loop.body.extend(_statements('''
_paper_checkpoint.save_epoch(training_state,env_state,buffer_state,local_key,
    evaluator,metrics_aggregator,replay_buffer,training_walltime,_+1,current_step,metrics)
'''))
    # Count actual native evaluation calls and preserve its private RNG state.
    class EvaluationCalls(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and node.func.attr=='run_evaluation'
                    and isinstance(node.func.value,ast.Name) and node.func.value.id=='evaluator'):
                node.func=ast.Attribute(value=ast.Name(id='_paper_checkpoint',ctx=ast.Load()),attr='evaluate',ctx=ast.Load())
                node.args.insert(0,ast.Name(id='evaluator',ctx=ast.Load()))
            return node
    train=EvaluationCalls().visit(train)
    after={n.name:ast.dump(n,include_attributes=False) for n in train.body
           if isinstance(n,ast.FunctionDef) and n.name in CORE_FUNCTIONS}
    if before!=after:
        raise AssertionError('Host checkpoint instrumentation altered a scientific inner function')
    return ast.fix_missing_locations(ast.Module(body=[train],type_ignores=[]))


def build_native_train(manager):
    module=importlib.import_module('crax.training.agents.sac_lag.train')
    source=Path(module.__file__).read_text()
    syntax=instrument_source(source)
    namespace=dict(module.__dict__,_paper_checkpoint=manager)
    exec(compile(syntax,str(module.__file__)+'[host-checkpoint-only]','exec'),namespace)
    return namespace['train']


def _atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def _atomic_pickle(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


class ExplicitCheckpointStop(RuntimeError):
    """Requested engineering boundary, not a scientific early-stop decision."""


class FullCheckpointManager:
    def __init__(self,output,settings,*,resume=None,allow_target_extension=False,
                 stop_after_epochs=0,candidate_count=5,backend=None):
        if not 1<=candidate_count<=5:
            raise ValueError('At most five predeclared eligible checkpoints')
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.settings=deepcopy(settings);self.candidate_count=candidate_count
        self.allow_target_extension=allow_target_extension
        self.stop_after_epochs=stop_after_epochs
        self.backend=backend;self.payload=None;self.resume_available=resume is not None
        self.epochs_this_invocation=0;self.evaluation_calls=0;self.candidates=[]
        self.initial_steps=0;self.last_steps=0;self.plan=None;self.last_metrics={}
        self.checkpoint_seconds=0.
        if resume is not None:
            # Trust boundary: never load a downloaded/user-supplied pickle.
            with Path(resume).open('rb') as stream:self.payload=pickle.load(stream)
            if not isinstance(self.payload,dict) or self.payload.get('format')!=CHECKPOINT_FORMAT:
                raise ValueError('Not a full native SAC checkpoint; policy-only reference cannot resume')
            if self.payload['native_train_sha256']!=TRAIN_SHA256 or self.payload['crax_commit']!=CRAX_COMMIT:
                raise ValueError('Pinned scientific source mismatch')
            if self.payload['settings']!=self.settings:
                raise ValueError('Resume requires identical learner, task, device and dependency configuration')
            self.initial_steps=self.last_steps=int(self.payload['current_step'])
            self.evaluation_calls=int(self.payload['evaluation_calls'])
            self.candidates=deepcopy(self.payload['candidate_checkpoints'])
            self.last_metrics=deepcopy(self.payload.get('metrics',{}))

    @property
    def jax(self):
        if self.backend is None:self.backend=importlib.import_module('jax')
        return self.backend

    def epoch_plan(self,stride,epochs,target,prefill,actor_steps,devices,processes):
        if devices!=1 or processes!=1:
            raise ValueError('This explicit checkpoint path supports one native device/process only')
        if self.payload:
            old=self.payload['epoch_plan']
            if old['target_steps']==target and old['candidate_count']!=self.candidate_count:
                raise ValueError('Cannot change checkpoint selection opportunity during exact continuation')
            if old['actor_steps']!=actor_steps or old['prefill_steps']!=prefill:
                raise ValueError('Native vector/prefill geometry changed')
            if target<old['target_steps']:
                raise ValueError('Cannot shorten the declared endpoint during continuation')
            if target!=old['target_steps'] and not self.allow_target_extension:
                raise ValueError('Target extension must be explicit; old stride and all actual steps remain charged')
            stride=old['actor_epoch_steps']
            epochs=math.ceil((target-prefill)/(stride*actor_steps))
        self.plan=dict(target_steps=int(target),prefill_steps=int(prefill),actor_steps=int(actor_steps),
            actor_epoch_steps=int(stride),total_epochs=int(epochs),
            candidate_count=self.candidate_count,
            actual_rounded_final_steps=int(prefill+epochs*stride*actor_steps))
        # Pure budget/time schedule, never selected by observed return or risk.
        self.plan['candidate_epochs']=[math.ceil((target*i/self.candidate_count-prefill)/(stride*actor_steps))
                                       for i in range(1,self.candidate_count+1)]
        self.plan['candidate_epochs']=[max(1,e) for e in self.plan['candidate_epochs']]
        if len(set(self.plan['candidate_epochs']))!=self.candidate_count:
            raise ValueError('Host granularity cannot provide the declared distinct candidate checkpoints')
        if self.payload:
            completed=self.payload['completed_epochs']
            for epoch in self.plan['candidate_epochs']:
                if epoch<=completed and not any(c['completed_epoch']==epoch and Path(c['path']).is_file() for c in self.candidates):
                    raise ValueError('Extension lacks an earlier required candidate artifact; use a fresh fairly budgeted run')
        _atomic_json(self.output/'checkpoint_plan.json',self.plan|dict(format=CHECKPOINT_FORMAT,
            resume_available=self.resume_available,initial_steps=self.initial_steps,
            fixed_native_train_sha256=TRAIN_SHA256,selection='exactly declared candidate_epochs;other host epochs ineligible'))
        return stride,epochs

    def is_candidate_epoch(self,epoch):
        return epoch in self.plan['candidate_epochs']

    def restore(self,training_state,env_state,buffer_state,local_key,evaluator,logger,replay):
        if not self.payload:
            return training_state,env_state,buffer_state,local_key,0.,0,0,{}
        p=self.payload
        training_state,env_state,buffer_state,local_key=self.jax.device_put(
            (p['training_state'],p['env_state'],p['buffer_state'],p['local_key']))
        evaluator._key=self.jax.device_put(p['evaluator_key'])
        evaluator._eval_walltime=p['evaluator_walltime']
        replay._size=p['replay_python_size']
        state=p['logger_state']
        logger._metrics_buffer=defaultdict(lambda:deque(maxlen=logger._buffer_size),
            {k:deque(v,maxlen=logger._buffer_size) for k,v in state['metrics_buffer'].items()})
        for name,value in state['attributes'].items():setattr(logger,name,deepcopy(value))
        # Offline/process-initialization time is not steady training time.
        logger._sps_last_time=None
        random.setstate(p['python_rng']);np.random.set_state(p['numpy_rng'])
        return (training_state,env_state,buffer_state,local_key,p['training_walltime'],
                p['completed_epochs'],p['current_step'],deepcopy(p.get('metrics',{})))

    def evaluate(self,evaluator,*args,**kwargs):
        result=evaluator.run_evaluation(*args,**kwargs)
        self.evaluation_calls+=1
        return result

    def save_epoch(self,training_state,env_state,buffer_state,local_key,evaluator,logger,replay,
                   training_walltime,completed_epoch,current_step,metrics):
        started=time.monotonic();self.jax.effects_barrier()
        self.last_steps=int(current_step)
        host=self.jax.device_get(dict(training_state=training_state,env_state=env_state,
            buffer_state=buffer_state,local_key=local_key,evaluator_key=evaluator._key))
        if self.is_candidate_epoch(completed_epoch):
            path=self.output/f'policy_normalizer_{current_step}.pkl'
            _atomic_pickle(path,dict(params=(host['training_state'].normalizer_params,
                host['training_state'].policy_params),steps=int(current_step),completed_epoch=completed_epoch,
                source='official_SAC_Lag_full_state',normalizer_saved=True,
                validation_status='pending_offline_common_validation;native_eval_is_diagnostic'))
            self.candidates=[c for c in self.candidates if c['completed_epoch']!=completed_epoch]
            self.candidates.append(dict(completed_epoch=completed_epoch,steps=int(current_step),path=str(path.resolve())))
        logger_attributes={name:deepcopy(getattr(logger,name)) for name in
            ('_num_steps','_last_log_steps','_log_count','_episodic_metrics_updated','_sps_last_steps')}
        self.last_metrics={k:float(np.asarray(v).mean()) for k,v in metrics.items()}
        payload=dict(format=CHECKPOINT_FORMAT,crax_commit=CRAX_COMMIT,native_train_sha256=TRAIN_SHA256,
            settings=self.settings,epoch_plan=self.plan,completed_epochs=completed_epoch,current_step=int(current_step),
            **host,evaluator_walltime=evaluator._eval_walltime,training_walltime=training_walltime,
            replay_python_size=getattr(replay,'_size',0),logger_state=dict(
                attributes=logger_attributes,metrics_buffer={k:list(v) for k,v in logger._metrics_buffer.items()}),
            candidate_checkpoints=self.candidates,evaluation_calls=self.evaluation_calls,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),metrics=self.last_metrics)
        _atomic_pickle(self.output/'full_latest.pkl',payload)
        eligible=[c for c in self.candidates if c['completed_epoch'] in self.plan['candidate_epochs']]
        raw_cost=np.asarray(host['env_state'].info['paper_total_cost']).sum()
        raw_goals=np.asarray(host['env_state'].info['paper_total_goals']).sum()
        self.checkpoint_seconds+=time.monotonic()-started
        _atomic_json(self.output/'checkpoint_status.json',dict(status='saved_host_boundary',
            format=CHECKPOINT_FORMAT,steps=int(current_step),initial_steps=self.initial_steps,
            new_training_transitions=int(current_step)-self.initial_steps,
            gradient_steps=int(np.asarray(host['training_state'].gradient_steps.lo).reshape(-1)[0])
                +(int(np.asarray(host['training_state'].gradient_steps.hi).reshape(-1)[0])<<32),
            training_cumulative_cost=float(raw_cost),training_cumulative_goals=float(raw_goals),
            evaluation_calls=self.evaluation_calls,candidate_checkpoints=eligible,
            selection_opportunity_limit=self.candidate_count,epoch_plan=self.plan,
            full_state_fields=list(host),checkpoint_seconds_this_invocation=self.checkpoint_seconds))
        self.epochs_this_invocation+=1
        if self.stop_after_epochs and self.epochs_this_invocation>=self.stop_after_epochs:
            raise ExplicitCheckpointStop('Explicit engineering stop after a fully saved host epoch')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--steps',type=int,default=1000000)
    parser.add_argument('--seed',type=int,default=41027)
    parser.add_argument('--num-envs',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--updates',type=int,default=4)
    parser.add_argument('--lagrangian-lr',type=float,default=.01,
        help='Explicit native base LR;effective per-SGD LR divides by native horizon')
    parser.add_argument('--normalize-observations',action=argparse.BooleanOptionalAction,default=False,
                        help='Explicit one-factor baseline setting; default remains the raw-state reference')
    parser.add_argument('--min-replay-size',type=int,default=20000)
    parser.add_argument('--max-replay-size',type=int,default=1000000)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--allow-target-extension',action='store_true')
    parser.add_argument('--stop-after-epochs',type=int,default=0)
    parser.add_argument('--engineering-check',action='store_true')
    parser.add_argument('--skip-native-evaluation',action='store_true')
    parser.add_argument('--epoch-actor-steps',type=int,default=0,
                        help='Engineering-only fixed host epoch size for one small actual resume check')
    args=parser.parse_args()
    if args.steps<=args.min_replay_size or args.steps<=0:parser.error('A finite target beyond prefill is required')
    if args.lagrangian_lr<=0:parser.error('Positive explicit Lagrangian base LR required')
    if (args.skip_native_evaluation or args.epoch_actor_steps) and not args.engineering_check:
        parser.error('Engineering scheduling flags are not a scientific training configuration')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new output directory for each explicit invocation; no overwrite/restart')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(args.output.parent/'jax-compilation-cache'))
    import jax
    from crax import envs
    from crax.training import acting,logger as native_logger
    from scripts.paper_crax_payload.native_sac_reference import CompleteReset
    # Prospective metadata/reproducibility fix after the dc197f3 recovery check:
    # official algorithm streams already use explicit JAX keys; these host
    # streams were unused but should also start reproducibly in future runs.
    started=time.monotonic();random.seed(args.seed);np.random.seed(args.seed)
    environment=envs.get_environment('safe_goal_point',level=1)
    native=environment.unwrapped
    settings=dict(crax_commit=CRAX_COMMIT,native_train_sha256=TRAIN_SHA256,seed=args.seed,
        num_envs=args.num_envs,batch_size=args.batch_size,grad_updates_per_step=args.updates,
        min_replay_size=args.min_replay_size,max_replay_size=args.max_replay_size,
        learning_rate=3e-4,discount=.99,reward_scaling=1.,cost_scaling=1.,normalize_observations=args.normalize_observations,
        safety_bound=25.,lagrangian_lr=args.lagrangian_lr,lambda_max=2.,episode_length=native.episode_length,
        action_repeat=1,env_id='safe_goal_point',difficulty=1,physics_timestep=float(native.sys.opt.timestep),
        frame_skip=native._n_frames,action_dim=native.action_size,obs_dim=native.observation_size,
        jax_version=jax.__version__,jax_devices=[d.device_kind for d in jax.local_devices()],
        engineering_check=args.engineering_check,engineering_epoch_actor_steps=args.epoch_actor_steps,
        skip_native_evaluation=args.skip_native_evaluation,host_rng_initialization='Python_NumPy_seed_from_training_seed',
        api_repairs=['fresh_complete_autoreset','pre_reset_terminal_observation','cumulative_raw_cost_accounting'])
    settings['dependency_versions']={}
    for package in ('jax','jaxlib','flax','optax','mujoco','mujoco-mjx','numpy','crax'):
        try:settings['dependency_versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:settings['dependency_versions'][package]=None
    settings['native_module_sha256']={}
    for name in ('crax.training.agents.sac_lag.train','crax.training.agents.sac_lag.losses',
                 'crax.training.agents.sac_lag.networks','crax.training.types',
                 'crax.envs.safe_goal','crax.envs.hazards'):
        module=importlib.import_module(name)
        settings['native_module_sha256'][name]=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    wrapper_module=importlib.import_module('scripts.paper_crax_payload.native_sac_reference')
    settings['paper_reset_wrapper_file_sha256']=hashlib.sha256(Path(wrapper_module.__file__).read_bytes()).hexdigest()
    manager=FullCheckpointManager(args.output,settings,resume=args.resume,
        allow_target_extension=args.allow_target_extension,stop_after_epochs=args.stop_after_epochs)
    if args.epoch_actor_steps:
        original_plan=manager.epoch_plan
        def engineering_plan(stride,epochs,target,prefill,actor_steps,devices,processes):
            return original_plan(args.epoch_actor_steps,math.ceil((target-prefill)/(args.epoch_actor_steps*actor_steps)),
                                 target,prefill,actor_steps,devices,processes)
        manager.epoch_plan=engineering_plan
    _atomic_json(args.output/'configuration.json',dict(settings=settings,requested_steps=args.steps,
        source_commit=args.source_commit,checkpoint_kind=CHECKPOINT_FORMAT,
        actual_native_source_guard=True,common_validation='offline_on_exactly_five_saved_candidates',
        resume_from=str(args.resume) if args.resume else None))
    def progress(step,metrics):
        row=dict(steps=int(step),elapsed_seconds=time.monotonic()-started,
                 metrics={k:float(np.asarray(v).mean()) for k,v in metrics.items()})
        with (args.output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    original_actor_step=acting.actor_step
    def corrected_actor_step(*args,**kwargs):
        state,transition=original_actor_step(*args,**kwargs)
        return state,transition._replace(next_observation=state.info['terminal_observation'])
    acting.actor_step=corrected_actor_step
    try:
        native_train=build_native_train(manager)
        stop='completed'
        try:
            native_train(environment=environment,num_timesteps=args.steps,episode_length=native.episode_length,
                num_envs=args.num_envs,num_eval_envs=50,learning_rate=3e-4,discounting=.99,seed=args.seed,
                batch_size=args.batch_size,num_evals=0 if args.skip_native_evaluation else 6,
                normalize_observations=args.normalize_observations,max_devices_per_host=1,reward_scaling=1.,
                min_replay_size=args.min_replay_size,max_replay_size=args.max_replay_size,
                grad_updates_per_step=args.updates,deterministic_eval=False,progress_fn=progress,
                training_metrics_steps=50000,safety_bound=25.,lagrangian_lr=args.lagrangian_lr,initial_lambda=0.,
                lambda_max=2.,wrap_env_fn=CompleteReset)
        except ExplicitCheckpointStop:
            stop='explicit_engineering_checkpoint_stop'
        jax.effects_barrier()
        status=json.loads((args.output/'checkpoint_status.json').read_text())
        elapsed=time.monotonic()-started
        _atomic_json(args.output/'receipt.json',dict(status=stop,steps=manager.last_steps,
            initial_steps=manager.initial_steps,new_training_transitions=manager.last_steps-manager.initial_steps,
            requested_steps=args.steps,source_commit=args.source_commit,elapsed_seconds=elapsed,
            training_cumulative_cost=status['training_cumulative_cost'],
            training_cumulative_goals=status['training_cumulative_goals'],
            native_evaluation_calls_total=manager.evaluation_calls,
            native_evaluation_calls_this_invocation=manager.evaluation_calls-(manager.payload['evaluation_calls'] if manager.payload else 0),
            native_evaluation_controls_this_invocation=(manager.evaluation_calls-(manager.payload['evaluation_calls'] if manager.payload else 0))*50*native.episode_length,
            common_validation_controls=0,common_validation_status='pending_offline_equal_candidate_rule',
            full_training_including_compile_eval_save_sps=(manager.last_steps-manager.initial_steps)/elapsed,
            full_state_resume_saved=True,five_candidate_schedule=manager.plan['candidate_epochs'],
            eligible_candidates=status['candidate_checkpoints'],native_train_sha256=TRAIN_SHA256,
            inner_scientific_functions_unchanged=list(CORE_FUNCTIONS),new_real_robot_interactions=0))
    finally:
        acting.actor_step=original_actor_step


if __name__=='__main__':main()
