"""Collect a bounded, fully target-crossed OWN training cohort.

Three new worlds x18 natural requests. These are CLEAN training executions,
not attack trials or a new success-rate evaluation. Existing results and source
remain frozen. The extension only parameterizes the pre-episode user request.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import threading
import time

from experiments.task_attack import MODEL, EFFORT
from experiments.task_attack.runtime import write_json
from experiments.task_attack_sensor_proxy.audit import implementation_hash, snapshot_implementation
from experiments.task_attack_realtime.batch import account_failure
from experiments.task_attack_value.perception import prepare
from run_target_parameter_training import NAMED, physical_digest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STOP = threading.Event()
LOCK = threading.Lock()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    used = set()
    for path in (ROOT / 'artifacts/task_attack').glob('*/manifest.json'):
        try:
            value = json.loads(path.read_text())
            used.update(r['seed'] for r in value.get('rows', []) if 'seed' in r)
        except (ValueError, OSError, TypeError):
            pass
    base = json.loads((ROOT / 'models/task_attack/integrity_v19/bundle.json').read_text())
    used.update(base['all_development_scene_seeds'])
    rng = random.Random(20261117031)
    worlds, rows = [], []
    for i, profile in enumerate(('single', 'independent', 'shared')):
        seed = rng.randrange(1, 2**31)
        while seed in used:
            seed = rng.randrange(1, 2**31)
        used.add(seed)
        world = 'world_%02d' % (i + 1)
        worlds.append(dict(world=world, seed=seed, source_profile=profile))
        requests = [(f, p) for f in NAMED for p in range(4)]
        requests += [(f, None) for f in ('cracks', 'temperature', 'crack_and_heat',
                                      'safe_equipment', 'thermal_minimum', 'equipment_labels')]
        rng.shuffle(requests)
        for family, port in requests:
            label = family if port is None else family + '_target_%d' % port
            rows.append(dict(world=world, family=family, target_port=port, seed=seed,
                source_profile=profile, policy='clean', directory=world + '/' + label + '/clean'))
    return dict(status='OWN clean target-crossing TRAINING; not held-out ASR',
        worlds=worlds, rows=rows, planned=len(rows), independent_worlds=3,
        task_types=9, task_parameter_configurations=18, equipment_count=4, workers=3,
        actor=dict(model=MODEL, reasoning_effort=EFFORT), implementation_hash=implementation_hash(),
        extension_sha256={p.name: sha(p) for p in (Path(__file__), HERE / 'run_target_parameter_training.py')},
        baseline_bundle='models/task_attack/baseline_bundle/bundle.json',
        budgets=dict(episode_seconds=900, request_seconds=120),
        task_extension='Each named family is executed for each of4 actual identities in the same physical world. Only the pre-episode natural request and corresponding private score target change; original scorer code unchanged.',
        autonomy='Original victim selects all observations, routes and reports using original common tools; no teacher route or answer.',
        information_boundary='Target, private task and truth are offline labels only. Unknown attack runtime cannot read them.',
        split='TRAIN-only fresh worlds. Every execution of a world stays in one fold. Old calibration/evaluation data remain untouched.',
        comparisons='Every method may use the same new OWN data. No attack budget, deadline, flight safety or scoring relaxation.',
        stop_rule='Stop dispatch after account/auth failure, native sensor timeout latch, source change or missing result. Keep every attempt and incomplete outcome.',
        attack_trials=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    mp = out / 'manifest.json'
    if mp.exists():
        manifest = json.loads(mp.read_text())
    else:
        manifest = plan()
        write_json(mp, manifest)
        snapshot_implementation(out / 'native_source_snapshot')
        for name in manifest['extension_sha256']:
            shutil.copy2(HERE / name, out / name)
    if manifest['actor'] != dict(model='gpt-6.1-terra', reasoning_effort='max'):
        raise ValueError('Retain the authorized Terra Max victim')

    def verify_sources():
        if implementation_hash() != manifest['implementation_hash'] or any(
                sha(HERE / name) != digest for name, digest in manifest['extension_sha256'].items()):
            raise RuntimeError('Frozen collection source changed')

    verify_sources()
    if not a.execute:
        print('Frozen TRAIN-only plan:54 clean executions; not dispatched', flush=True)
        return
    if (out / 'execution_stop.json').exists():
        raise RuntimeError('A recorded stop requires inspection; do not bypass it')
    queue = manifest['worlds'][:]

    def checkpoint():
        with LOCK:
            write_json(out / 'progress.json', dict(planned=manifest['planned'],
                completed=sum((out / r['directory'] / 'result.json').exists() for r in manifest['rows']),
                stop_requested=STOP.is_set(), attack_trials=0, wall=time.time()))

    def invoke(args, directory, extended=False):
        directory.mkdir(parents=True, exist_ok=True)
        if extended:
            script = '\n'.join([
                'source /opt/ros/noetic/setup.bash',
                'source ' + shlex.quote(str(ROOT / 'sim_ws/devel/setup.bash')),
                'export PYTHONPATH=' + shlex.quote(str(ROOT / 'bridge') + ':' + str(ROOT)) + ':"${PYTHONPATH:-}"',
                'exec ' + ' '.join(shlex.quote(v) for v in args)])
            args = ['bash', '-e', '-c', script]
        with (directory / 'runner.log').open('w') as log:
            return subprocess.call(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    def worker(index):
        while not STOP.is_set():
            with LOCK:
                if not queue:
                    return
                world = queue.pop(0)
            try:
                verify_sources()
                setup = out / 'setup' / world['world']
                capture, static = setup / 'capture', setup / 'shared_static.json'
                common = ['--backend', 'cpu', '--headless', '--equipment-count', '4',
                    '--seed', str(world['seed']), '--source-profile', world['source_profile'],
                    '--ros-port', str(11831 + index), '--gazebo-port', str(11931 + index),
                    '--baseline-bundle', manifest['baseline_bundle']]
                if not static.exists():
                    if not (capture / 'result.json').exists():
                        code = invoke(['bash', 'scripts/run_attack_sensor_proxy.sh', '--sensor-smoke',
                            '--family', 'equipment_labels', '--output', str(capture)] + common, capture)
                        if code:
                            raise RuntimeError('Native sensor preparation failed')
                    prepare(capture, static)
                expected_physical = None
                for row in [r for r in manifest['rows'] if r['world'] == world['world']]:
                    if STOP.is_set():
                        return
                    verify_sources()
                    directory = out / row['directory']
                    if not (directory / 'result.json').exists():
                        print('TRAIN START', world['world'], row['family'], row['target_port'], flush=True)
                        args = ['python3', str(HERE / 'run_target_parameter_training.py'),
                            '--policy', 'clean', '--family', row['family'], '--episode-timeout', '900',
                            '--output', str(directory)] + common
                        if row['target_port'] is not None:
                            args += ['--training-target-port', str(row['target_port'])]
                        code = invoke(args, directory, extended=True)
                        if not (directory / 'result.json').exists():
                            raise RuntimeError('Training process ended without result; exit=%d' % code)
                    result = json.loads((directory / 'result.json').read_text())
                    if result.get('robot_model') != manifest['actor']:
                        raise RuntimeError('Victim model differs from frozen plan')
                    physical = physical_digest(dict(world=str(directory / 'inspection.world')), directory)
                    if expected_physical is None:
                        expected_physical = physical
                    if physical != expected_physical:
                        raise RuntimeError('World/assets differ across natural requests')
                    failure = account_failure(directory)
                    if result.get('error', {}).get('code') == 'native_sensor_timeout_latched':
                        failure = dict(reason='Original native sensor timeout latch')
                    if failure:
                        raise RuntimeError('Native/account stop: ' + str(failure)[:250])
                    print('TRAIN DONE', world['world'], row['family'], row['target_port'],
                          result.get('score', {}).get('outcome'), flush=True)
                    checkpoint()
            except Exception as error:
                STOP.set()
                write_json(out / ('execution_stop_worker_%d.json' % index), dict(
                    world=world['world'], type=type(error).__name__, message=str(error)[:350], wall=time.time()))
                write_json(out / 'execution_stop.json', dict(
                    world=world['world'], type=type(error).__name__, message=str(error)[:350], wall=time.time()))
                checkpoint()
                raise

    checkpoint()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(worker, i) for i in range(3)]:
            future.result()
    checkpoint()
    print('TRAIN collection complete; audit all54 before fitting; no attacks dispatched', flush=True)


if __name__ == '__main__':
    main()
