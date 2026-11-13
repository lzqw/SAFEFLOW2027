"""Independent-process Piper simulation; every native physics query is retained."""
import multiprocessing as mp
import traceback

from deployment.piper.simulation import PiperSimConfig
from scripts.paper_crax_payload.piper_helpers import (
    PiperTrainingEnvironment, capture_piper_environment, restore_piper_environment)

COUNTERS = ('total_control_intervals', 'total_physics_substeps', 'total_distance_queries')


def _worker(connection, assets, settings):
    env = None
    try:
        env = PiperTrainingEnvironment(assets, PiperSimConfig(**settings))
        while True:
            operation, payload = connection.recv()
            if operation == 'close':
                break
            if operation == 'capture':
                result = capture_piper_environment(env)
            elif operation == 'restore':
                result = restore_piper_environment(env, payload)
            elif operation == 'definition':
                result = env.definition()
            elif operation == 'reset':
                result = env.reset(**payload)
            elif operation == 'step':
                result = env.step(payload)
            else:
                raise ValueError(f'Unknown environment operation: {operation}')
            connection.send((True, result, {k: getattr(env, k) for k in COUNTERS}))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        connection.send((False, traceback.format_exc(), {}))
    finally:
        if env is not None:
            env.close()
        connection.close()


class PiperProcessEnvironment:
    """One worker-owned Bullet client; dispatch all steps before receiving any."""
    def __init__(self, assets, settings):
        context = mp.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, assets, settings), daemon=True)
        self.pending = False
        for key in COUNTERS:
            setattr(self, key, 0)
        self.process.start()
        child.close()

    def _receive(self):
        if not self.connection.poll(180):
            raise TimeoutError(f'Piper environment worker {self.process.pid} timed out; no silent reset')
        okay, result, counters = self.connection.recv()
        self.pending = False
        if not okay:
            raise RuntimeError(result)
        for key, value in counters.items():
            setattr(self, key, value)
        return result

    def _call(self, operation, payload=None):
        if self.pending:
            raise RuntimeError('Cannot interleave operations with a pending physical step')
        self.connection.send((operation, payload))
        self.pending = True
        return self._receive()

    def begin_step(self, action):
        if self.pending:
            raise RuntimeError('Previous step is still pending')
        self.connection.send(('step', action))
        self.pending = True

    def step(self, action):
        if not self.pending:
            self.begin_step(action)
        return self._receive()

    def reset(self, **kwargs):
        return self._call('reset', kwargs)

    def definition(self):
        return self._call('definition')

    def capture(self):
        return self._call('capture')

    def restore(self, state):
        return self._call('restore', state)

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(('close', None))
            except (OSError, BrokenPipeError):
                pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
        self.connection.close()
