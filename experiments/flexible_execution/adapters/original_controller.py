"""Run the hash-pinned upstream controller with the shared metered transport."""
import subprocess
import os
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

AGENT_BLOB = "15c07cd1b55cb5e26d8d6872a603fb3a0b2daf1c"
AGENT_SHA256 = "6fb095847e927b712f9a8e5bdb1a25ee7119f2cc7d49e72a5aba042a6d781066"


def upstream():
    root = Path(__file__).resolve().parents[3]
    stored = subprocess.run(["git", "cat-file", "blob", AGENT_BLOB], cwd=root,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if stored.returncode == 0:
        source = stored.stdout
    else:
        upstream = Path(os.environ.get("VLA_LEGACY_ROOT", Path.home() / "uav/Embodied-intelligent-behavior"))
        source = (upstream / "experiments/task_attack/agent.py").read_bytes()
    if sha256(source).hexdigest() != AGENT_SHA256:
        raise ValueError("Original controller source fingerprint differs")
    module = ModuleType("experiments.task_attack.upstream_agent")
    module.__package__ = "experiments.task_attack"
    exec(compile(source, "upstream/agent.py", "exec"), module.__dict__)
    return module


def run(backend, task, directory, event, timeout, *, model, trace):
    module = upstream()
    module.ModelClient = lambda path: model
    # The upstream loop emits task decisions, but has no interval clock.
    def timed(call, role):
        def measured(*args, **kwargs):
            with trace.measure(role, model.calls + 1, purpose=call.__name__):
                return call(*args, **kwargs)
        return measured
    with ExitStack() as stack:
        for obj, name, role in [(model, "request", "model_request"), (backend, "observe", "observation"),
                                (backend, "move", "action")]:
            stack.enter_context(patch.object(obj, name, timed(getattr(obj, name), role)))
        return module.run_agent(backend, task, directory, event, timeout)
