"""Piper Payload fixed-P5/activation/Metric native120K full-parent forks."""
import argparse
import json
from pathlib import Path

from scripts.paper_crax_payload.train_native_p5_adaptation import train
from scripts.paper_crax_payload.train_piper_comparison_v12 import attach
from scripts.paper_crax_payload.train_native_p5_warmup_v12 import evaluate, TrainingTracker

PLAN = Path(__file__).resolve().parents[2]/'configs/paper_crax_payload/simulation_v12_payload_round1.json'
FORMAT = 'native_piper_payload_activation_metric_v12_full'


def validate(plan, args):
    if (plan != json.loads(PLAN.read_text()) or args.line != 'piper_payload'
            or args.arm not in plan['arms'] or args.resume):
        raise ValueError('Use the exact registered Payload native100K comparison')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('plan', 'parent', 'assets-receipt', 'output', 'data-root'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--arm', required=True); p.add_argument('--line', default='piper_payload')
    p.add_argument('--source-commit', required=True); p.add_argument('--device', default='cuda:0')
    p.add_argument('--cuda-memory-fraction', type=float, default=.05)
    args = p.parse_args(); args.resume = None
    train(args, protocol_validator=validate, adaptation_attacher=attach, format_name=FORMAT,
          piper_evaluator=evaluate, tracker_factory=TrainingTracker)
