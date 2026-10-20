#!/usr/bin/env python3
"""Run unchanged training source with CUDA synchronization on exit/signals.

MPS clients must not be force-killed with outstanding device work. This entrypoint
does not alter the learner, configuration, sampling, budget, or checkpoint format.
"""
import runpy
import signal
import sys


def requested_exit(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    assert len(sys.argv) > 1 and sys.argv[1] == 'scripts/star/train_star.py'
    signal.signal(signal.SIGTERM, requested_exit)
    signal.signal(signal.SIGINT, requested_exit)
    sys.argv = sys.argv[1:]
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        torch = sys.modules.get('torch')
        cuda = getattr(torch, 'cuda', None)
        if cuda is not None and getattr(cuda, 'is_initialized', lambda: False)():
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            cuda.synchronize()


if __name__ == '__main__':
    main()
