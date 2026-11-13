"""Versioned Piper policy export, with native provenance and integrity checks.

Exporter only packages genuinely trained Piper state supplied by the training
runner. It does not generate a trained policy, transfer Panda weights, or
authorize physical deployment. Normalization is explicitly frozen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_provenance(metadata: dict):
    if metadata.get('robot_family') != 'Piper':
        raise ValueError('Panda/other robot policy cannot be relabelled as Piper')
    if metadata.get('task') not in ('Reach', 'Payload'):
        raise ValueError('Unrecognised Piper task')
    if metadata.get('method') not in ('Iso', 'Iso-Low', 'EA'):
        raise ValueError('Unrecognised deployment method')
    if metadata.get('observation_dim') != 34 or metadata.get('action_dim') != 3:
        raise ValueError('This simulator has an explicit Piper34/action3 contract')
    if metadata.get('training_robot_family') != 'Piper' or metadata.get('training_steps', 0) <= 0:
        raise ValueError('A native Piper training receipt is required; no renamed weights')
    if not metadata.get('model_commit') or not metadata.get('training_source_commit'):
        raise ValueError('Pinned model and training source revisions are required')
    if metadata.get('normalizer_frozen') is not True:
        raise ValueError('Deployment normalizer must be frozen')
    if metadata.get('action_frame') != 'robot_base' or metadata.get('action_unit') != 'normalized_delta_xyz':
        raise ValueError('Action scaling/frame is ambiguous')
    if metadata.get('physical_motion_enabled') is not False:
        raise ValueError('Preparation exports may not enable physical motion')


def export_bundle(output: Path, learner_state: dict, metadata: dict,
                  normalizer: dict, training_receipt: Path):
    """Call only after selecting a native Piper policy with the common rule."""
    import torch
    validate_provenance(metadata)
    training_receipt = Path(training_receipt)
    receipt = json.loads(training_receipt.read_text())
    if receipt.get('robot_family') != 'Piper' or receipt.get('task') != metadata['task']:
        raise ValueError('Native training receipt does not match robot/task')
    if receipt.get('training_steps') != metadata['training_steps']:
        raise ValueError('Training step count mismatch')
    if receipt.get('model_commit') != metadata['model_commit']:
        raise ValueError('Training model revision mismatch')
    if normalizer.get('dimension') != metadata['observation_dim']:
        raise ValueError('Normalizer dimension mismatch')
    if normalizer.get('frozen') is not True:
        raise ValueError('Normalizer must be frozen at export')
    if 'actor' not in learner_state:
        raise ValueError('Missing Actor')
    stable = learner_state.get('ea_stability_v2', {})
    margin = stable.get('margin_geometry')
    if metadata['method'] == 'EA' and margin is None:
        raise ValueError('EA export must contain its frozen EMA geometry margin, not its online head')
    expected_variant = 's_ea' if metadata['method'] == 'EA' else 's_iso'
    if stable.get('variant') != expected_variant:
        raise ValueError('Native learner variant does not match deployment method')
    variance = learner_state.get('reference_variance')
    if variance is None or metadata.get('reference_variance') != variance:
        raise ValueError('Explicit native reference variance must match; never rename Iso as Iso-Low')
    if metadata['method'] == 'Iso-Low' and not 0 < variance < metadata.get('iso_reference_variance', .04):
        raise ValueError('Iso-Low must really use lower variance and preserve that native provenance')
    for field in ['hidden_width', 'flow_steps', 'action_scale_m']:
        if metadata.get(field, 0) <= 0:
            raise ValueError(f'Explicit positive {field} required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # No replay or optimizer state is needed by inference. Raw training receipt
    # remains linked/digested; this file is not a recovery checkpoint.
    torch.save({'actor': learner_state['actor'], 'margin': margin,
                'sampler': {key: learner_state.get(key) for key in
                            ['reference_variance', 'normal_variance_ratio']},
                'geometry_transitions': stable.get('transitions', metadata['training_steps'])}, output/'policy.pt')
    (output/'normalizer.json').write_text(json.dumps(normalizer,indent=2,allow_nan=False)+'\n')
    manifest=dict(metadata, schema_version=1,
                  policy_sha256=sha256(output/'policy.pt'),
                  normalizer_sha256=sha256(output/'normalizer.json'),
                  training_receipt_sha256=sha256(training_receipt),
                  hardware_transport_available=False, recovery_checkpoint=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class FrozenPiperPolicy:
    """Inference adapter for an exported native bundle; no transport/motion.

    Actor/margin/normalizer stay frozen, but geometry input autograd stays
    enabled. Timing/deadline handling belongs to OfflineController and caller.
    """
    def __init__(self, directory: Path, device='cpu'):
        import numpy as np
        import torch
        from scripts.star.arm_safe_entropy.endpoint_geometry import EndpointGeometry, StableFlowActor
        from scripts.star.arm_safe_entropy.geometry import LocalMarginEnsemble
        directory=Path(directory)
        self.manifest=validate_bundle(directory)
        state=torch.load(directory/'policy.pt',map_location=device,weights_only=True)
        sampler=state['sampler']
        self.device=torch.device(device)
        self.actor=StableFlowActor(34,width=self.manifest['hidden_width'],
            num_steps=self.manifest['flow_steps'],variance=sampler['reference_variance'],
            rho=sampler['normal_variance_ratio']).to(self.device)
        self.actor.load_state_dict(state['actor'])
        self.actor.eval().requires_grad_(False)
        self.geometry=None
        if self.manifest['method']=='EA':
            margin=LocalMarginEnsemble(34).to(self.device)
            margin.load_state_dict(state['margin'])
            margin.eval().requires_grad_(False)
            self.geometry=EndpointGeometry(margin,'endpoint_pullback',state['geometry_transitions'])
        normalizer=json.loads((directory/'normalizer.json').read_text())
        if normalizer.get('kind')=='identity':
            self.mean=np.zeros(34,dtype=np.float32)
            self.scale=np.ones(34,dtype=np.float32)
        elif normalizer.get('kind')=='frozen_affine':
            self.mean=np.asarray(normalizer['mean'],dtype=np.float32)
            self.scale=np.asarray(normalizer['scale'],dtype=np.float32)
            if (self.mean.shape!=(34,) or self.scale.shape!=(34,)
                    or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.scale))
                    or not np.all(self.scale>0)):
                raise ValueError('Invalid frozen normalizer')
        else:
            raise ValueError('Unknown normalizer; never silently substitute identity')

    def action(self, observation, deterministic=False):
        import numpy as np
        import torch
        observation=np.asarray(observation,dtype=np.float32)
        if observation.shape[-1]!=34 or not np.all(np.isfinite(observation)):
            raise ValueError('Invalid Piper observation')
        x=torch.as_tensor((observation-self.mean)/self.scale,device=self.device)
        # Do not use torch.inference_mode: endpoint pullback needs input grads.
        with torch.no_grad():
            action,_,_=self.actor(x,deterministic=deterministic,geometry=self.geometry)
        return action.cpu().numpy()


def validate_bundle(directory: Path):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    validate_provenance(manifest)
    if sha256(directory/'policy.pt') != manifest['policy_sha256']:
        raise ValueError('Policy integrity mismatch')
    if sha256(directory/'normalizer.json') != manifest['normalizer_sha256']:
        raise ValueError('Normalizer integrity mismatch')
    return manifest
