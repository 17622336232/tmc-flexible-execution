"""Isolated OWN collection entry point with a pre-episode task parameter.

The native generator chooses one target per world. This wrapper permits each
physical source to be the requested target in separate CLEAN training episodes.
Only the authorized request and its matching private evaluation specification
change BEFORE execution; assets, tools, actor, safety and scorer are untouched.
It is not an online attack interface and is not used in held-out evaluation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from experiments.task_attack.runtime import write_json

NAMED = ('locate', 'locate_crack', 'locate_condition')


def physical_digest(scene, root):
    root = Path(root).resolve()
    paths = [Path(scene['world'])] + sorted((root / 'assets').rglob('*'))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            value = path.read_bytes().replace(str(root).encode(), b'<EPISODE_ROOT>')
            digest.update(str(path.relative_to(root)).encode() + b'\0' + value + b'\0')
    return digest.hexdigest()


def set_training_target(scene, root, port):
    """Define the request, not its answer, before the victim is started."""
    truth = scene['truth']
    if truth['family'] not in NAMED or isinstance(port, bool) or port not in range(len(truth['devices'])):
        raise ValueError('A named equipment task and an actual target port are required')
    before = physical_digest(scene, root)
    original = copy.deepcopy(truth)
    old = original['target']
    new = truth['devices'][port]['identity']
    token = '编号为' + old + '的设备'
    if truth['task'].count(token) != 1 or truth['score_spec'].get('target') != old:
        raise ValueError('Unexpected original request; do not partially redefine it')
    truth['task'] = truth['task'].replace(token, '编号为' + new + '的设备')
    truth['target'] = new
    truth['score_spec']['target'] = new
    changed = {k for k in truth if truth[k] != original[k]}
    if not changed <= {'task', 'target', 'score_spec'}:
        raise ValueError('A task parameter must not change the physical scene')
    if physical_digest(scene, root) != before:
        raise ValueError('Physical scene changed while defining a task parameter')
    write_json(Path(root) / 'private/truth.json', truth)
    write_json(Path(root) / 'private/training_task_parameter.json', dict(
        purpose='OWN clean target-crossing training only; defined before actor execution',
        target_port=port, original_target=old, requested_target=new,
        original_task=original['task'], authorized_task=truth['task'],
        physical_digest_before=before, physical_digest_after=before,
        changed_truth_fields=sorted(changed),
        native_actor_tools_control_safety_scorer_unchanged=True,
        scorer_specification='Same native scorer, with target matching the new natural request; no post-hoc answer edits'))
    return scene


def main(argv=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--training-target-port', type=int, choices=range(4))
    own, remaining = p.parse_known_args(argv)
    from experiments.task_attack_sensor_proxy import run as native
    if '--policy' not in remaining or remaining[remaining.index('--policy') + 1] != 'clean':
        raise ValueError('This collection extension only permits clean episodes')
    original = native.build_scene

    def build_scene(root, seed=1, family='cracks', count=6, source_profile='independent'):
        scene = original(root, seed, family, count, source_profile)
        if own.training_target_port is not None:
            return set_training_target(scene, root, own.training_target_port)
        return scene

    native.build_scene = build_scene
    try:
        return native.main(remaining)
    finally:
        native.build_scene = original


if __name__ == '__main__':
    raise SystemExit(main())
