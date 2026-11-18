"""Unlaunched v12 SafeReach warm-up candidate; historical P5 loaders unchanged.

Only the outward-loss coefficient changes. Covariance, margin geometry, true
task thresholds and critic updates retain their native P5 paths. This module
does not approve a protocol, allocate a budget, open physics or run training.
Piper's native adaptation requires a separately registered native-task fork.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from scripts.paper_crax_payload.direction_development import DirectionDevelopmentLearner


@dataclass(frozen=True)
class DirectionWarmup:
    ramp_steps: int
    initial_fraction: float = 0.

    def __post_init__(self):
        if type(self.ramp_steps) is not int or self.ramp_steps not in (0, 50000, 75000):
            raise ValueError('Use fixed P5 or a separately registered 50K/75K ramp')
        if self.initial_fraction not in (0., .25):
            raise ValueError('Use the proposed zero or quarter-strength initial coefficient')
        if self.ramp_steps == 0 and self.initial_fraction != 0.:
            raise ValueError('Fixed P5 has no warm-up initial fraction')

    def weight(self, transitions):
        if type(transitions) is not int or transitions < 0:
            raise ValueError('Use the actual nonnegative integer training transition count')
        if self.ramp_steps == 0:
            return 1.
        return self.initial_fraction + (1.-self.initial_fraction)*min(1., transitions/self.ramp_steps)


def p5_actor_lr(transitions, decay=True):
    if type(transitions) is not int or transitions < 0 or type(decay) is not bool:
        raise ValueError('Invalid absolute SafeReach transition count or LR-decay switch')
    if not decay:
        return .0001
    fraction = min(1., max(0., (transitions-150000)/50000))
    return .00003 + (.0001-.00003)*.5*(1.+math.cos(math.pi*fraction))


class SafeReachWarmupLearner(DirectionDevelopmentLearner):
    """Fresh SafeReach P5/one-factor ramp, with strict own-format resume.

    No arbitrary historical prefix import: warm-up starts with fresh training,
    which is necessary to measure genuine first100K exploration risk. The
    baseline's existing late Actor LR and target deployment remain separate.
    """
    def __init__(self, warmup, *, late_actor_lr_decay=True, **kwargs):
        if not isinstance(warmup, DirectionWarmup) or type(late_actor_lr_decay) is not bool:
            raise ValueError('Explicit immutable warm-up settings required')
        if 'actor_lr' in kwargs and kwargs['actor_lr'] != .0001:
            raise ValueError('Preserve baseline initial Actor LR')
        super().__init__(component='full', rho=.25, normal_shrink=True, direction_weight=1., **kwargs)
        self.warmup = warmup
        self.late_actor_lr_decay = late_actor_lr_decay
        self.set_transitions(0)

    def set_transitions(self, transitions):
        # The native base rejects rewinding real training steps.
        if type(transitions) is not int or transitions < 0:
            raise ValueError('Use an actual nonnegative integer training transition count')
        super().set_transitions(transitions)
        if hasattr(self, 'warmup'):
            for group in self.actor_opt.param_groups:
                group['lr'] = p5_actor_lr(transitions, self.late_actor_lr_decay)

    def actor_regularizer(self, trace):
        return self.warmup.weight(self.transitions)*super().actor_regularizer(trace)

    def metrics(self):
        return super().metrics() | dict(
            direction_weight_effective=self.warmup.weight(self.transitions),
            actor_lr_effective=self.actor_opt.param_groups[0]['lr'])

    def _settings(self):
        return dict(task='SafeReach-RelativeLayout-v3', time_basis='fresh_training_absolute_transitions',
            warmup=asdict(self.warmup), late_actor_lr_decay=self.late_actor_lr_decay)

    def state_dict(self):
        return super().state_dict() | dict(direction_warmup_v12=dict(
            settings=self._settings(), transitions=self.transitions,
            direction_weight=self.warmup.weight(self.transitions),
            actor_lr=p5_actor_lr(self.transitions, self.late_actor_lr_decay)))

    def load_state_dict(self, state):
        meta = state.get('direction_warmup_v12')
        if not meta or self.transitions != 0 or meta.get('settings') != self._settings():
            raise ValueError('Resume only the identical v12 schedule into a fresh learner')
        step = meta.get('transitions')
        if (type(step) is not int or step < 0
                or meta.get('direction_weight') != self.warmup.weight(step)
                or meta.get('actor_lr') != p5_actor_lr(step, self.late_actor_lr_decay)
                or state['ea_stability_v2']['transitions'] != step
                or any(g['lr'] != meta['actor_lr'] for g in state['actor_opt']['param_groups'])):
            raise ValueError('Saved v12 transition, coefficient or optimizer LR differs')
        # Native network/optimizer/margin/EMA and component guards still apply.
        super().load_state_dict(deepcopy(state))
