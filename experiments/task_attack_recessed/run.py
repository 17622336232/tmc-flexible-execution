"""Run one complete, isolated Gazebo task/attack episode."""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time

from experiments.task_attack import EFFORT, FAMILIES, MODEL, POLICIES, RESEARCH_POLICIES, ATTACK_EFFORT, ATTACK_EFFORTS, SOURCE_PROFILES
from experiments.task_attack.agent import run_agent
from .audit import implementation_hash
from .cleanup import capture as capture_ros_core,stop_captured as stop_ros_core
from experiments.task_attack.evaluate import evaluate, SCORE_VERSION
from experiments.task_attack.runtime import GazeboBackend, PublicGateway, append_json, write_json
from .scene import build_scene

ROOT=Path(__file__).resolve().parents[2]


def _stop(process):
    if process is None or process.poll() is not None:return
    for signum,timeout in [(signal.SIGINT,10),(signal.SIGTERM,5),(signal.SIGKILL,3)]:
        try:os.killpg(process.pid,signum)
        except ProcessLookupError:return
        try:process.wait(timeout=timeout);return
        except subprocess.TimeoutExpired:pass


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',choices=('clean','belief','belief_blind','belief_known_task','redvla_external','randomized'),default='clean')
    p.add_argument('--family',choices=FAMILIES,default='cracks')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--equipment-count',type=int,default=6,help='Visible fixture count; all paired policies share it')
    p.add_argument('--output',type=Path)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--no-rviz',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--sensor-smoke',action='store_true',help='Only verify actual RGB/LiDAR and takeoff; no model or attack')
    p.add_argument('--edit-smoke',help='With --sensor-smoke, verify this physical edit using real before/after RGB')
    p.add_argument('--attach',action='store_true',help='Use an explicitly started matching pilot world on the chosen ports')
    p.add_argument('--ros-port',type=int,default=11381)
    p.add_argument('--gazebo-port',type=int,default=11481)
    p.add_argument('--episode-timeout',type=float,default=1800)
    p.add_argument('--observer-calls',type=int,default=1)
    p.add_argument('--attack-effort',choices=ATTACK_EFFORTS,default=ATTACK_EFFORT)
    p.add_argument('--source-profile',choices=SOURCE_PROFILES,default='independent')
    p.add_argument('--backend',choices=('gazebo','cpu'),default='gazebo',
                   help='CPU: same SDF, finite-range physical rays and native ROS planner; separate rendering-domain results')
    p.add_argument('--baseline-bundle',type=Path,help='Frozen trained visual textures, identically installed for all paired policies')
    p.add_argument('--timing-model',type=Path,help='Frozen external-only timing model; no current-task training data')
    p.add_argument('--calibration-seed',type=int)
    a=p.parse_args(argv)
    if (a.policy=='randomized')!=(a.calibration_seed is not None):p.error('A separate randomization seed is required only for randomized proxy collection')
    if a.episode_timeout<=0 or a.observer_calls<1:p.error('Positive experimental budgets required')
    if a.observer_calls!=1:p.error('The realtime paired comparison uses exactly one observer request')
    if a.edit_smoke and not a.sensor_smoke:p.error('--edit-smoke requires --sensor-smoke')
    if not 2<=a.equipment_count<=6:p.error('This pilot fixture supports 2..6 devices')
    root=(a.output or ROOT/'artifacts/task_attack'/('%s_%s_%s_s%d'%(time.strftime('%Y%m%d_%H%M%S'),a.family,a.policy,a.seed))).resolve()
    if (root/'result.json').exists():p.error('Output already contains an episode result; choose a fresh directory')
    scene=build_scene(root,a.seed,a.family,count=a.equipment_count,source_profile=a.source_profile)
    if a.baseline_bundle:
        from experiments.task_attack.visual_assets import load_bundle,add_visual_assets
        bundle,textures,bundle_hash=load_bundle(a.baseline_bundle)
        if a.seed in bundle.get('training_scene_seeds',[]):p.error('Evaluation scene seed overlaps baseline training')
        add_visual_assets(scene,textures,bundle_hash)
    elif a.policy in ('uao_spsa','iccv_uada_spsa','iccv_upa_spsa'):
        p.error('Visual baseline evaluation requires a trained frozen --baseline-bundle')
    scene['catalog']['timing']=dict(robot_after_takeoff_budget_seconds=a.episode_timeout,
        max_single_request_seconds=120, max_observer_requests=a.observer_calls,
        takeoff_timing='Visible in external RGB; no private progress signal is exposed.')
    write_json(root/'public/catalog.json',scene['catalog'])
    manifest=dict(schema_version=1,policy=a.policy,family=a.family,seed=a.seed,equipment_count=a.equipment_count,
                  implementation_hash=implementation_hash(),
                  implementation='zero_shot_task_inference',trained_operation_model=False,
                  model=MODEL,reasoning_effort=EFFORT,observer_call_budget=a.observer_calls,
                  robot_model=dict(model=MODEL,reasoning_effort=EFFORT),
                  attacker_model=dict(model=MODEL,reasoning_effort=a.attack_effort,trained=False),
                  source_profile=a.source_profile,scene_name=scene['truth']['scene_name'],
                  episode_timeout_seconds=a.episode_timeout,
                  attack_budget=1 if a.policy!='clean' else 0,
                  simulation='Gazebo Classic RGB + ray LiDAR + existing Diff-Planner; interface flight simulation, not motor SITL',
                  backend=a.backend,
                  paired_initial_conditions=True,model_sampling_determinism=False)
    manifest['carrier_protocol']=scene['catalog'].get('experiment_conditions',{}).get('carrier_protocol')
    manifest['baseline_bundle_sha256']=scene['catalog'].get('visual_carriers',{}).get('bundle_sha256')
    manifest['paper_adaptation']=a.policy in ('uao_spsa','iccv_uada_spsa','iccv_upa_spsa','redvla_external')
    if a.policy in ('belief','belief_blind','belief_known_task'):
        manifest['carrier_protocol']='side_state_decals_v11'
        manifest['implementation']='recessed_operation_v11'
        manifest['first_request_eager']=True
        manifest['full_paper_KDE_reproduction']=False
    elif manifest['paper_adaptation']:
        manifest['implementation']='explicit_api_or_external_view_paper_adaptation'
    if a.timing_model:
        import hashlib
        timing=json.loads(a.timing_model.read_text())
        if a.seed in timing.get('all_development_scene_seeds',timing.get('training_scene_seeds',[])):p.error('Timing model training seed overlaps evaluation')
        manifest['timing_model_sha256']=hashlib.sha256(a.timing_model.read_bytes()).hexdigest()
        manifest['joint_read_continuation_trained']=True
        manifest['model_training_scenes']=len(timing.get('training_scene_seeds',[]))
        manifest['model_calibration_scenes']=len(timing.get('calibration_scene_seeds',[]))
    if a.backend=='cpu':
        manifest['simulation']='CPU TinyRenderer RGB + finite-range physical rays + unchanged native Diff-Planner/controller; not Gazebo or motor SITL'
    manifest['isolated_development_variant']=True
    manifest['parent_frozen_v6_implementation']='6383e73bdb552ea8b8d9fe5722e81f254091eaf5f2b6f07b4696dbcdfb329cbe'
    write_json(root/'manifest.json',manifest)
    print('自主任务实验：%s / %s / seed=%d；机器人 %s Max；观察者 %s'%(a.family,a.policy,a.seed,MODEL,a.attack_effort),flush=True)
    print('输出：'+str(root),flush=True)
    if a.prepare_only:
        print('场景、纹理和实验清单已生成；尚未运行任务或模型。',flush=True);return 0
    os.environ['ROS_MASTER_URI']='http://127.0.0.1:%d'%a.ros_port
    os.environ['GAZEBO_MASTER_URI']='http://127.0.0.1:%d'%a.gazebo_port
    os.environ['GAZEBO_MODEL_DATABASE_URI']=''
    os.environ['VLA_MODE']='gazebo'
    os.environ['ROS_LOG_DIR']=str(root/'private/roslog')
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
    ros=renderer=observer=backend=gateway=None
    observer_dir=None
    handles=[]
    owned_ros_core=[]
    def event(kind,**fields):
        append_json(root/'private/events.jsonl',dict(wall=time.time(),event=kind,**fields))
        detail=fields.get('decision',{}).get('reason') or fields.get('message') or ''
        print('[%s] %s'%(kind,str(detail)[:240]),flush=True)
    result=dict(manifest,execution_completed=False,score=dict(score_version=SCORE_VERSION,outcome='incomplete',wrong_accepted=False,correct=False))
    try:
        if not a.attach:
            for port in (a.ros_port,a.gazebo_port):
                s=socket.socket()
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                try:s.bind(('127.0.0.1',port))
                except OSError:raise RuntimeError('Pilot port %d occupied; choose unused ports or explicitly use --attach'%port)
                finally:s.close()
            log=(root/'private/simulation.log').open('w');handles.append(log)
            command=['roslaunch','vla_sim','demo.launch','gui:='+str(not a.headless).lower(),
                     'rviz:='+str(not a.headless and not a.no_rviz).lower(),
                     'sensor_profile:=indoor','scene_profile:=gallery','start_yaw:=0.74',
                     'world_file:='+scene['world']]
            if a.backend=='cpu':
                command=['roslaunch',str(ROOT/'experiments/task_attack/cpu_backend.launch'),
                         'world_file:='+scene['world']]
                result['simulation']='CPU TinyRenderer pinhole RGB + finite-range collision ray tests + unchanged Diff-Planner/flight controller; not Gazebo or motor SITL'
            ros=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            import xmlrpc.client
            until=time.monotonic()+60
            while time.monotonic()<until:
                if ros.poll() is not None:raise RuntimeError('Pilot Gazebo launch exited; see private/simulation.log')
                try:
                    if xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/pilot_startup')[0]==1:break
                except OSError:pass
                time.sleep(.2)
            else:raise RuntimeError('Pilot ROS master did not start')
            if a.backend=='cpu':
                env=dict(os.environ)
                dependency=ROOT/'.runtime/task_attack_cpu_deps'
                env['PYTHONPATH']=str(dependency)+os.pathsep+env.get('PYTHONPATH','')
                env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
                renderer=subprocess.Popen([sys.executable,'-m','experiments.task_attack.cpu_world','--world',scene['world']],
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        backend=GazeboBackend(root,scene,event)
        backend.wait_ready()
        if not a.attach:owned_ros_core=capture_ros_core(root)
        if a.policy!='clean' and not a.sensor_smoke:
            gateway=PublicGateway(backend,secrets.token_hex(16))
        backend.start()
        if gateway:
            # A neutral path prevents the task family / trial filename leaking
            # to the online observer through its command-line log destination.
            observer_dir=ROOT/'.runtime/task_attack/observer_work'/secrets.token_hex(16)
            observer_dir.mkdir(parents=True,exist_ok=True)
            env={k:v for k,v in os.environ.items() if not k.startswith(('ROS_','GAZEBO_','VLA_','UAV_','CMAKE_','CATKIN_'))}
            # The observer only receives its public gateway and its own log directory.
            # No ROS/scene/target configuration is passed. This is a data-access
            # boundary between trusted implementations, not an OS security sandbox.
            env['PYTHONPATH']=str(ROOT/'bridge')+os.pathsep+str(ROOT)
            from experiments.task_attack_operator.belief import DEPENDENCY_POLICIES
            module=('calibration' if a.policy=='randomized' else 'belief' if a.policy in DEPENDENCY_POLICIES else 'baselines' if manifest['paper_adaptation'] else 'observer')
            cmd=[sys.executable,'-m',('experiments.task_attack_recessed.calibration' if module=='calibration' else 'experiments.task_attack_operator.belief' if module=='belief' else 'experiments.task_attack_recessed.redvla' if a.policy=='redvla_external' else 'experiments.task_attack.'+module),'--gateway',gateway.url,
                 '--policy',a.policy,'--output',str(observer_dir),'--call-budget',str(a.observer_calls),
                 '--effort',a.attack_effort]
            if a.policy=='randomized':cmd+=['--calibration-seed',str(a.calibration_seed)]
            if (module=='belief' or a.policy=='redvla_external') and a.timing_model:cmd+=['--timing-model',str(a.timing_model.resolve())]
            if a.policy in ('known_task','belief_known_task'):
                task_file=observer_dir/'authorized_task_reference.txt';task_file.write_text(scene['truth']['task'])
                cmd+=['--known-task-file',str(task_file)]
            log=(observer_dir/'process.log').open('w');handles.append(log)
            observer=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        event('takeoff_requested')
        backend.takeoff()
        event('takeoff_complete')
        if a.sensor_smoke:
            if a.edit_smoke:
                # Engineering-only camera placement, never used by a task or
                # online attacker. Main episodes always use model decisions.
                diagnostic_x=scene['action_states'][a.edit_smoke]['position'][0]
                # Engineering camera placement uses ordinary intermediate
                # observations and the UNCHANGED native mapper admission.
                for camera_x,camera_y,camera_yaw in [(-4.7,-2.5,1.0),(diagnostic_x,-1.6,1.5707963267948966)]:
                    backend.observe()
                    backend.move(dict(x=camera_x,y=camera_y,z=1.2,yaw=camera_yaw))
            obs,geometry=backend.observe()
            (root/'onboard_sensor_smoke.jpg').write_bytes(obs['data'])
            write_json(root/'sensor_geometry.json',geometry)
            with backend.lock:
                if backend.latest_external:
                    (root/'external_sensor_smoke.jpg').write_bytes(backend.latest_external['data'])
            if a.edit_smoke:
                import hashlib
                (root/'onboard_before_edit.jpg').write_bytes(obs['data'])
                backend.edit(a.edit_smoke)
                time.sleep(1.)
                after,after_geometry=backend.observe()
                (root/'onboard_after_edit.jpg').write_bytes(after['data'])
                result['physical_edit_smoke']=dict(action_id=a.edit_smoke,
                    before_sha256=hashlib.sha256(obs['data']).hexdigest(),
                    after_sha256=hashlib.sha256(after['data']).hexdigest(),
                    old_capture_unchanged=(root/'onboard_before_edit.jpg').read_bytes()==obs['data'])
            result.update(execution_completed=True,sensor_smoke_only=True,
                          score=dict(outcome='not_evaluated',wrong_accepted=False,correct=False))
            return 0
        report,metrics=run_agent(backend,scene['truth']['task'],root/'private/robot',event,a.episode_timeout)
        with backend.lock:backend.finished=True
        result.update(report=report,robot_metrics=metrics,score=evaluate(scene['truth'],report,backend.attack),
                      actual_edit=backend.attack,execution_completed=True)
        if backend.attack:
            edit=backend.attack
            later=[]
            for path in sorted((root/'private/robot').glob('onboard_*.json')):
                frame=json.loads(path.read_text())
                if frame['ros_stamp']>edit['ack_ros']:later.append(frame['image_id'])
            result['attack_timing']=dict(onboard_images_captured_after_edit=later,
                note='Later captures are opportunities, not proof that the edited carrier was visible or adopted.')
        event('episode_complete',message=result['score']['outcome'])
    except KeyboardInterrupt:
        result['error']=dict(type='KeyboardInterrupt',message='User stopped the pilot')
        event('episode_interrupted')
    except Exception as error:
        result['error']=dict(type=type(error).__name__,message=str(error)[:1200],
                             code=getattr(error,'code',None),status=getattr(error,'status',None))
        event('episode_error',message=result['error']['message'])
    finally:
        if backend:
            with backend.lock:backend.finished=True
        if observer:
            try:observer.wait(timeout=3)
            except subprocess.TimeoutExpired:_stop(observer)
        for handle in handles:
            handle.flush()
        if observer_dir is not None:
            shutil.copytree(str(observer_dir),str(root/'private/observer'),dirs_exist_ok=True)
            shutil.rmtree(str(observer_dir))
        obs_result=root/'private/observer/result.json'
        if obs_result.exists():result['observer']=json.loads(obs_result.read_text())
        elif a.policy!='clean' and not a.sensor_smoke:result['observer']=dict(decision='no_completed_observer_result')
        if gateway:gateway.close()
        if backend:
            try:backend.close()
            except Exception as error:event('cleanup_error',type=type(error).__name__)
        _stop(ros)
        remaining_core=stop_ros_core(root,owned_ros_core)
        if remaining_core:event('cleanup_core_remaining',pids=remaining_core)
        _stop(renderer)
        for f in handles:f.close()
        write_json(root/'result.json',result)
    print('结果：'+result['score']['outcome']+'；记录：'+str(root/'result.json'),flush=True)
    return 0 if result['execution_completed'] else 1


if __name__=='__main__':raise SystemExit(main())
