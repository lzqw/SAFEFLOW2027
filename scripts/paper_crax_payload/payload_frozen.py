"""New standalone Payload validation/test adapter; active training stays unchanged."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scripts.paper_crax_payload.payload_select import candidates, digest, verify_test_authorization


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def frozen_fingerprint(learner):
    """One periodic host copy, not a hot-loop loss synchronization."""
    result = hashlib.sha256()
    for name in ('actor', 'target_actor', 'q', 'qt', 'cost_q', 'cost_qt', 'margin_online', 'margin_geometry'):
        module = getattr(learner, name, None)
        if module is None:
            continue
        result.update(name.encode())
        for key, value in sorted(module.state_dict().items()):
            result.update(key.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    result.update(learner.log_alpha.detach().cpu().numpy().tobytes())
    result.update(json.dumps(dict(safety_multiplier=learner.safety_multiplier,
        transitions=learner.transitions, updates=learner.updates, actor_updates=learner.actor_updates,
        normalizer=dict(kind='identity_after_explicit_native_units', dimension=45, frozen=True)), sort_keys=True).encode())
    return result.hexdigest()


def load_candidate(plan, cohort, identifier, run_root, config, device):
    import torch
    from scripts.paper_crax_payload.payload_models import build_payload_learner
    registered = {c['candidate_id']: c for c in candidates(plan, cohort)}
    if identifier not in registered:
        raise ValueError('Not one of the five registered checkpoints for an allowed seed/method')
    candidate = registered[identifier]
    directory = Path(run_root)/candidate['run_id']
    manifest = json.loads((directory/'run_config.json').read_text())
    if (manifest.get('task') != plan['task'] or manifest['configuration'] != config or
            manifest['method'] != candidate['method'] or manifest['training_seed'] != candidate['training_seed'] or
            manifest.get('phase') != cohort):
        raise ValueError('Native training provenance/configuration mismatch; no renamed/engineering policy')
    policy = directory/candidate['policy_filename']
    # This immutable full snapshot is atomically finalized AFTER the native
    # policy file, so an actively written policy cannot be read halfway through.
    if not (directory/candidate['full_filename']).is_file():
        raise ValueError('Checkpoint publication is incomplete or missing; do not omit it silently')
    state = torch.load(policy, map_location=device, weights_only=False)
    native = state.get('payload_sac') if candidate['method']=='sac_lag' else state.get('ea_stability_v2')
    if native is None or native['transitions'] != candidate['checkpoint_steps']:
        raise ValueError('Native policy carries a different real transition endpoint')
    learner = build_payload_learner(config, candidate['method'], device)
    learner.load_state_dict(state)
    for name in ('actor', 'target_actor', 'q', 'qt', 'cost_q', 'cost_qt', 'margin_online', 'margin_geometry'):
        module = getattr(learner, name, None)
        if module is not None:
            module.eval().requires_grad_(False)
    learner.log_alpha.requires_grad_(False)
    return candidate, manifest, policy, learner


def execute(plan, cohort, identifier, run_root, config, split, device='cpu', selection=None,
            trace_directory=None, trace_episodes=0):
    import torch
    from scripts.paper_crax_payload.payload_rollouts import evaluate_frozen_rollouts
    torch.set_num_threads(1)
    candidate, manifest, policy, learner = load_candidate(plan, cohort, identifier, run_root, config, device)
    policy_hash = file_sha256(policy)
    status = 'validation_candidate'
    if split in ('independent_test', 'route_calibration'):
        if selection is None:
            raise ValueError('Independent test is locked until a complete common-rule selection is frozen')
        status = verify_test_authorization(plan, cohort, candidate, policy_hash, selection)
    elif split != 'validation':
        raise ValueError('Development scores cannot be relabelled as common validation')
    if split == 'route_calibration':
        protocol = plan['route_qualification']
        settings = dict(episodes_per_condition=protocol['calibration_layout_count'],
            layout_seed_start=protocol['calibration_layout_seed_start'],
            action_sampling_seed=protocol['action_sampling_seed'],
            conditions=[plan['validation']['conditions'][0]])
        trace_episodes = settings['episodes_per_condition']
        if trace_directory is None:
            raise ValueError('Route calibration needs private initial-state and executed-action witnesses')
    else:
        settings = plan[split]
    copied = deepcopy(config)
    copied['evaluation']['action_sampling_seed'] = settings['action_sampling_seed']
    before = frozen_fingerprint(learner)
    trace_location=Path(trace_directory)/split/identifier.replace(':','_') if trace_directory is not None else None
    result = evaluate_frozen_rollouts(learner, copied, settings['episodes_per_condition'],
        settings['layout_seed_start'], settings['conditions'], trace_location, trace_episodes,
        provenance=dict(candidate=candidate, policy_sha256=policy_hash, plan_digest=digest(plan),
            selection_freeze_digest=selection['freeze_digest'] if selection else None))
    if frozen_fingerprint(learner) != before or file_sha256(policy) != policy_hash:
        raise RuntimeError('Frozen parameters/normalizer/dual or source policy changed during evaluation')
    return dict(status='completed', schema='payload_frozen_evaluation_v1', split=split,
        cohort=cohort, candidate_id=identifier, candidate=candidate, plan_digest=digest(plan),
        training_config_digest=digest(config), policy_sha256=policy_hash,
        training_source_commit=manifest['source_commit'], normalizer=dict(kind='identity', dimension=45, frozen=True),
        frozen_unchanged=True, frozen_fields=['Actor', 'online_margin', 'EMA_margin', 'normalizer', 'dual', 'critics', 'targets'],
        geometry_input_gradients_enabled=True, adaptation_updates=0, new_training_transitions=0,
        policy_selection_status=status,
        selection_freeze_digest=selection['freeze_digest'] if split in ('independent_test','route_calibration') else None,
        route_claim='offset_geometry_only_until_separate_truth_replay_qualification', result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--candidate-id', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'independent_test', 'route_calibration'], required=True)
    parser.add_argument('--trace-directory', type=Path)
    parser.add_argument('--trace-episodes', type=int, default=0)
    parser.add_argument('--source-commit')
    parser.add_argument('--selection-freeze', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve prior frozen evidence; never overwrite an independent test')
    from scripts.paper_crax_payload.payload_rollouts import FrozenRolloutInterrupted
    try:
        result = execute(json.loads(args.plan.read_text()), args.cohort, args.candidate_id, args.run_root,
            json.loads(args.training_config.read_text()), args.split, args.device,
            json.loads(args.selection_freeze.read_text()) if args.selection_freeze else None,
            args.trace_directory,args.trace_episodes)
    except FrozenRolloutInterrupted as error:
        result=error.receipt|dict(candidate_id=args.candidate_id,cohort=args.cohort,split=args.split,
            plan_digest=digest(json.loads(args.plan.read_text())),evaluation_source_commit=args.source_commit)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        from scripts.star.arm_safe_entropy.train import atomic_json
        atomic_json(args.output,result)
        raise
    result['evaluation_source_commit']=args.source_commit
    result['evaluation_source_files_sha256']={name:file_sha256(Path(__file__).with_name(name)) for name in
        ('payload_frozen.py','payload_select.py','payload_rollouts.py','payload_route_qualification.py')}
    import torch
    import numpy as np
    from importlib.metadata import version
    result['evaluation_software']=dict(torch=str(torch.__version__),numpy=np.__version__,device=args.device,
        **{name:version(name) for name in ('pybullet','gymnasium','panda-gym')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from scripts.star.arm_safe_entropy.train import atomic_json
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('result', 'frozen_fields')}))


if __name__ == '__main__':
    main()
