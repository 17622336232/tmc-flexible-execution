"""Gazebo display synchronization and cleanup, outside the frozen experiment.

Only the GUI pose topic receives copied, measured static-model poses. The task,
physics, RGB/LiDAR topics and original native experiment are not intercepted.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def build_sync():
    source = Path(__file__).with_name('gui_pose_sync.cc')
    flags = shlex.split(subprocess.check_output(['pkg-config', '--cflags', '--libs', 'gazebo'], text=True))
    version = subprocess.check_output(['pkg-config', '--modversion', 'gazebo'], text=True).strip()
    digest = hashlib.sha256(source.read_bytes()+repr((flags, version)).encode()).hexdigest()[:20]
    cache = ROOT/'.runtime/task_attack_showcase'
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache/('gui_pose_sync_'+digest)
    if not binary.exists():
        handle, temporary = tempfile.mkstemp(prefix='gui_pose_sync_', dir=str(cache))
        os.close(handle)
        try:
            subprocess.run(['g++', '-std=c++17', '-O2', str(source), '-o', temporary]+flags, check=True)
            os.chmod(temporary, 0o755)
            os.replace(temporary, str(binary))
        finally:
            Path(temporary).unlink(missing_ok=True)
    return binary


def static_models(world_file, destination):
    world = ET.parse(str(world_file)).getroot().find('world')
    if world is None:
        raise ValueError('Gazebo场景缺少world')
    names = [model.get('name') for model in world.findall('model')
             if model.findtext('static', '').strip().lower() in ('true', '1')]
    if not names or any(not name or '\n' in name for name in names):
        raise ValueError('Gazebo场景静态模型清单无效')
    destination.write_text('\n'.join(names)+'\n')
    return world.get('name')


def owned_processes(directory, auxiliaries=False):
    """Match this unique output path and executable, never all ROS/Gazebo jobs."""
    found = {}
    marker = str(directory.resolve())+'/'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc/'cmdline').read_bytes().split(b'\0')
            if not raw or not raw[0]:
                continue
            executable = Path(os.fsdecode(raw[0])).name
            command = [os.fsdecode(part) for part in raw if part]
            allowed = {'gzclient', 'rviz'}
            if auxiliaries:
                allowed.update(('gzserver', 'rosmaster', 'rosout'))
                if executable in ('python3', 'python', 'sh') and len(command) > 1:
                    executable = Path(command[1]).name
            if executable not in allowed or not any(marker in part for part in command[1:]):
                continue
            # PID reuse must not let cleanup signal an unrelated later process.
            start = (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]
            found[int(proc.name)] = start
        except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
            continue
    return found


def owned_gui_processes(directory):
    return owned_processes(directory)


def cleanup_gui(directory, tracked, auxiliaries=False):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        current = owned_processes(directory, auxiliaries=auxiliaries)
        for pid, start in tracked.items():
            if current.get(pid) == start:
                try:
                    os.kill(pid, signum)
                except ProcessLookupError:
                    pass
        if signum == signal.SIGTERM and any(pid in current for pid in tracked):
            time.sleep(.5)


def run(command, env, directory, world_file, gazebo_port, binary):
    display = directory/'display'
    display.mkdir(exist_ok=True)
    task = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    helper = None
    helper_log = None
    tracked = {}
    started = time.monotonic()
    warned = False
    def interrupted(*_):
        raise KeyboardInterrupt()
    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        while task.poll() is None:
            tracked.update(owned_processes(directory, auxiliaries=True))
            if helper is None and world_file.exists() and owned_gui_processes(directory):
                world = static_models(world_file, display/'static_models.txt')
                helper_env = dict(env, GAZEBO_MASTER_URI='http://127.0.0.1:%d'%gazebo_port)
                helper_log = (display/'pose_sync.jsonl').open('w')
                helper = subprocess.Popen([str(binary), world, str(display/'static_models.txt')],
                    env=helper_env, stdout=helper_log, stderr=subprocess.STDOUT, start_new_session=True)
                (display/'manifest.json').write_text(json.dumps(dict(
                    mode='copy_actual_static_poses_to_gui_topic',
                    source_topic='~/pose/local/info', destination_topic='~/pose/info',
                    physics_or_sensor_writes=False, task_or_attack_decisions_changed=False,
                    binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()), indent=2))
                print('[Gazebo显示] 已启用静态物体位置同步，主窗口跟随实际场景修改。', flush=True)
            if helper is not None and helper.poll() is not None and not warned:
                print('[Gazebo显示] 位置同步助手已退出，详见本次 display/pose_sync.jsonl。', flush=True)
                warned = True
            time.sleep(.2)
        return task.returncode
    except KeyboardInterrupt:
        print('正在结束本次展示并关闭本次窗口。', flush=True)
        try:
            os.killpg(task.pid, signal.SIGINT)
            task.wait(timeout=35)
        except subprocess.TimeoutExpired:
            os.killpg(task.pid, signal.SIGTERM)
            try:
                task.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(task.pid, signal.SIGKILL)
                task.wait()
        except ProcessLookupError:
            pass
        return 130
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        tracked.update(owned_processes(directory, auxiliaries=True))
        if helper is not None and helper.poll() is None:
            helper.terminate()
            try:
                helper.wait(timeout=3)
            except subprocess.TimeoutExpired:
                helper.kill()
                helper.wait()
        if helper_log is not None:
            helper_log.close()
        cleanup_gui(directory, tracked, auxiliaries=True)
        (display/'cleanup.json').write_text(json.dumps(dict(
            tracked_display_and_auxiliary_pids=sorted(tracked),
            remaining_owned_display_and_auxiliary_pids=sorted(owned_processes(directory, auxiliaries=True)),
            elapsed_seconds=round(time.monotonic()-started, 2)), indent=2))
