"""Gazebo展示：冻结等待策略，或三类场景的自主任务／纯场景预览。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys

from experiments.task_attack import EFFORT, FAMILIES, MODEL, SOURCE_PROFILES
from experiments.task_attack.tasks import SCENES

ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT/'research/attack_learning'
CANDIDATE = ROOT/'models/task_attack/showcase_candidate'
BASELINE = ROOT/'models/task_attack/baseline_bundle/bundle.json'
LEARNED_FAMILIES = ('cracks', 'temperature', 'locate', 'crack_and_heat',
    'thermal_minimum', 'safe_equipment', 'locate_crack', 'locate_condition')
POLICIES = ('clean', 'belief', 'belief_blind', 'direct_effect', 'belief_known_task', 'redvla_external')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=FAMILIES, default='cracks')
    p.add_argument('--policy', choices=POLICIES, default='belief',
        help='belief为最好一批中的冻结等待策略；clean为无攻击自主任务')
    p.add_argument('--seed', type=int, default=706400677,
        help='默认取自已报告的cracks展示案例；演示不计入原测试')
    p.add_argument('--source-profile', choices=SOURCE_PROFILES, default='single')
    p.add_argument('--ros-port', type=int, default=11391)
    p.add_argument('--gazebo-port', type=int, default=11491)
    p.add_argument('--episode-timeout', type=float, default=900.)
    p.add_argument('--output', type=Path)
    p.add_argument('--headless', action='store_true')
    p.add_argument('--no-rviz', action='store_true')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--view-only', action='store_true', help='仅打开真实Gazebo场景，无API调用、不起飞')
    modes.add_argument('--prepare-only', action='store_true', help='仅生成SDF和材质，不启动ROS/Gazebo或调用API')
    modes.add_argument('--sensor-smoke', action='store_true', help='真实Gazebo起飞与RGB/雷达检查，无任务/攻击API')
    return p


def validate_scope(args):
    if not any((args.view_only, args.prepare_only, args.sensor_smoke)):
        if args.policy != 'clean' and args.family not in LEARNED_FAMILIES:
            raise ValueError('冻结学习攻击只验证过8类设备任务。其他任务请显式使用 --policy clean；纯场景可使用 --view-only。')
    if args.episode_timeout <= 0:
        raise ValueError('任务预算必须为正数')
    if args.ros_port == args.gazebo_port or not all(1024 <= port <= 65535 for port in (args.ros_port, args.gazebo_port)):
        raise ValueError('请选择不同的有效ROS/Gazebo端口')


def native_args(args, directory, policy='clean', headless=None):
    result = ['--backend', 'gazebo', '--family', args.family, '--policy', policy,
        '--seed', str(args.seed), '--source-profile', args.source_profile, '--equipment-count', '4',
        '--episode-timeout', str(args.episode_timeout), '--ros-port', str(args.ros_port),
        '--gazebo-port', str(args.gazebo_port), '--output', str(directory)]
    if BASELINE.is_file():
        result += ['--baseline-bundle', str(BASELINE)]
    if (args.headless if headless is None else headless):
        result.append('--headless')
    if args.no_rviz:
        result.append('--no-rviz')
    return result


def task_command(args, directory, perception=None):
    argv = native_args(args, directory, args.policy)
    if args.policy == 'clean':
        return [sys.executable, '-m', 'experiments.task_attack_integrity.run']+argv
    if perception is None:
        raise ValueError('Attack demonstration needs its actual Gazebo pre-task RGB preparation')
    return [sys.executable, '-m', 'conditional_critic_native_runner', '--candidate-policies',
        'belief,belief_blind,belief_known_task,direct_effect,redvla_external']+argv+[
        '--timing-model', str(CANDIDATE/'bundle.json'), '--public-perception', str(perception)]


def verify_candidate():
    from experiments.task_attack_sensor_proxy.audit import implementation_hash
    if not (CANDIDATE/'candidate_contract.json').is_file():
        raise RuntimeError('源码仓库不携带训练权重。请先按 README 导入自己的模型；也可用 --view-only 查看场景，或 --policy clean 执行无攻击任务。')
    contract = json.loads((CANDIDATE/'candidate_contract.json').read_text())
    if sha(CANDIDATE/'bundle.json') != contract['bundle_sha256'] or implementation_hash() != contract['native_sha256']:
        raise RuntimeError('冻结候选或原生执行代码发生变化，请核对版本，不静默换成其他模型')
    for name, expected in contract['asset_sha256'].items():
        if sha(CANDIDATE/name) != expected:
            raise RuntimeError('冻结模型文件变化：'+name)
    for name in ('conditional_critic_native_runner.py', 'conditional_critic_observer.py',
                 'conditional_critic_policy.py', 'common_resource_red_anchor.py',
                 'crossfit_dependency_stopping_data.py', 'learn_read_dependency.py'):
        if sha(HELPERS/name) != contract['extension_source_sha256'][name]:
            raise RuntimeError('冻结在线执行代码变化：'+name)


def available_ports(args):
    for port in (args.ros_port, args.gazebo_port):
        with socket.socket() as sock:
            # Match the native launch check: a closed prior connection's
            # TIME_WAIT socket is not a still-running ROS/Gazebo server.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError:
                raise RuntimeError('端口%d已被占用，请关闭自己的上一场演示或显式更换端口；不停止其他项目。'%port)


def main(argv=None):
    args = parser().parse_args(argv)
    validate_scope(args)
    mode = 'scene_view' if args.view_only else 'prepare_only' if args.prepare_only else 'sensor_smoke' if args.sensor_smoke else 'task'
    out = (args.output or ROOT/'artifacts/task_attack'/('showcase_'+datetime.now().strftime('%Y%m%d_%H%M%S')+
        '_'+args.family+'_'+secrets.token_hex(3))).resolve()
    if out.exists():
        raise RuntimeError('展示目录已存在，请使用新目录；不覆盖既有结果：'+str(out))
    if mode != 'prepare_only':
        available_ports(args)
    if mode == 'task' and args.policy != 'clean':
        verify_candidate()
    display_binary = None
    if mode != 'prepare_only' and not args.headless:
        from .display import build_sync
        display_binary = build_sync()
    out.mkdir(parents=True)
    from experiments.task_attack.runtime import write_json
    write_json(out/'showcase_manifest.json', dict(mode=mode, family=args.family, scene=SCENES[args.family],
        requested_policy=args.policy, seed=args.seed, source_profile=args.source_profile,
        method='Portable conditional critic and learned waiting' if mode == 'task' and args.policy != 'clean' else None,
        actual_robot_model=MODEL+'/'+EFFORT if mode == 'task' else None,
        reproduction='Source distribution: fresh local experiments only; historical results are not shipped.',
        source_sha256=sha(Path(__file__)), benchmark_results_changed=False, ros_port=args.ros_port, gazebo_port=args.gazebo_port))
    print('展示记录：'+str(out), flush=True)
    print('ROS端口%d；Gazebo端口%d。Ctrl+C停止本次演示。'%(args.ros_port, args.gazebo_port), flush=True)
    if mode in ('prepare_only', 'scene_view'):
        flight = out/'scene'
        subprocess.run([sys.executable, '-m', 'experiments.task_attack_integrity.run']+
            native_args(args, flight, headless=True)+['--prepare-only'], cwd=ROOT, check=True)
        world = flight/'inspection.world'
        print('场景文件：'+str(world), flush=True)
        if mode == 'prepare_only':
            return 0
        print('仅查看场景和实测传感器，不自动起飞、不调用模型。', flush=True)
        env = dict(os.environ, ROS_MASTER_URI='http://127.0.0.1:%d'%args.ros_port,
            GAZEBO_MASTER_URI='http://127.0.0.1:%d'%args.gazebo_port,
            GAZEBO_MODEL_DATABASE_URI='', ROS_LOG_DIR=str(out/'private/roslog'))
        command = ['roslaunch', 'vla_sim', 'demo.launch', 'gui:='+str(not args.headless).lower(),
            'rviz:='+str(not args.headless and not args.no_rviz).lower(), 'sensor_profile:=indoor',
            'scene_profile:=gallery', 'start_yaw:=0.74', 'world_file:='+str(world)]
        if display_binary is not None:
            from .display import run
            return run(command, env, out, world, args.gazebo_port, display_binary)
        os.execvpe(command[0], command, env)
    if mode == 'sensor_smoke':
        print('只做真实Gazebo传感器检查，会自动起飞；不执行任务或攻击。', flush=True)
        command = ([sys.executable, '-m', 'experiments.task_attack_integrity.run']+
            native_args(args, out/'flight')+['--sensor-smoke'])
        if display_binary is not None:
            from .display import run
            return run(command, dict(os.environ), out, out/'flight/inspection.world', args.gazebo_port, display_binary)
        os.execvpe(command[0], command, os.environ)
    perception = None
    if args.policy != 'clean':
        capture = out/'static_capture'
        print('先采集同一Gazebo场景的静态RGB，进行一次 %s/high 场景解读；这一步不执行用户任务。'%MODEL, flush=True)
        command = ([sys.executable, '-m', 'experiments.task_attack_integrity.run']+
            native_args(args, capture, headless=True)+['--sensor-smoke'])
        with (out/'static_capture.log').open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        from experiments.task_attack_value.perception import prepare
        perception = out/'shared_static.json'
        prepare(capture, perception)
    print('开始自主任务：将打开Gazebo/RViz并自动起飞；机载 %s/%s 自行观察、移动、记录和提交。'%(MODEL, EFFORT), flush=True)
    command = task_command(args, out/'flight', perception)
    if args.policy != 'clean':
        print('攻击由模型选择，可能修改正面裂纹，也可能修改右侧温度显示，编号不一定变化。覆片会直接出现，没有贴覆动画。', flush=True)
        try:
            subprocess.Popen([sys.executable, '-m', 'experiments.task_attack_showcase.watch',
                '--flight', str(out/'flight'), '--parent-pid', str(os.getpid())], cwd=ROOT)
        except OSError:
            print('攻击文字提示未能启动；原任务继续，实际修改记录见本次目录 flight/private/edit.json。', flush=True)
    if display_binary is not None:
        from .display import run
        return run(command, dict(os.environ), out, out/'flight/inspection.world', args.gazebo_port, display_binary)
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
