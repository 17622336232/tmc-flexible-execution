"""Keep the original ray physics; isolate it from expensive RGB rendering.

Two independent Bullet clients load the same SDF. The range client contains
identical collision geometry, receives the same actual body poses, and mirrors
every physical edit BEFORE acknowledging it. No camera render can hold its lock
or Python interpreter. Native map/controller freshness checks remain unchanged.
"""
import argparse
import atexit
import copy
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from experiments.task_attack.cpu_world import SceneRenderer


class CollisionRenderer(SceneRenderer):
    def _model(self,model,name,collisions=True):
        model=copy.deepcopy(model)
        for link in model.findall('link'):
            for visual in list(link.findall('visual')):link.remove(visual)
        return super()._model(model,name,collisions)


def imports():
    import rospy
    from gazebo_msgs.msg import ModelState,ModelStates,ContactsState,ContactState
    from gazebo_msgs.srv import SetModelState,SetModelStateResponse
    from geometry_msgs.msg import Pose,Twist,Point32
    from sensor_msgs.msg import Image,CameraInfo,PointCloud
    from std_msgs.msg import Header
    return locals()


def range_main(world,diagnostics):
    g=imports();rospy=g['rospy'];rospy.init_node('task_attack_lidar_world')
    renderer=CollisionRenderer(world);lock=threading.Lock()
    robot=[[-6.,-3.,.18],renderer.p.getQuaternionFromEuler([0,0,.74])]
    def update(message):
        if message.model_name!='vla_uav':return
        p,q=message.pose.position,message.pose.orientation
        with lock:robot[:]=[[p.x,p.y,p.z],[q.x,q.y,q.z,q.w]]
    rospy.Subscriber('/gazebo/set_model_state',g['ModelState'],update,queue_size=1)
    def apply(request):
        item=request.model_state;p,q=item.pose.position,item.pose.orientation
        try:
            renderer.set_pose(item.model_name,[p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
            return g['SetModelStateResponse'](True,'collision geometry mirrored')
        except ValueError as error:return g['SetModelStateResponse'](False,str(error))
    rospy.Service('/pilot/lidar/set_model_state',g['SetModelState'],apply)
    model_pub=rospy.Publisher('/gazebo/model_states',g['ModelStates'],queue_size=1)
    lidar_pub=rospy.Publisher('/vla/lidar/points_raw',g['PointCloud'],queue_size=1)
    contact_pub=rospy.Publisher('/vla/uav_contacts',g['ContactsState'],queue_size=1)
    def model_tick(_):
        with lock:pos,quat=[list(x) for x in robot]
        message=g['ModelStates']();message.name=['vla_uav'];pose=g['Pose']()
        pose.position.x,pose.position.y,pose.position.z=pos
        pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=quat
        message.pose=[pose];message.twist=[g['Twist']()];model_pub.publish(message)
    timer=rospy.Timer(rospy.Duration(.05),model_tick);last=None
    log=Path(diagnostics).open('a',buffering=1) if diagnostics else None
    try:
        while not rospy.is_shutdown():
            begin=time.monotonic()
            with renderer.lock:
                with lock:pos,quat=[list(x) for x in robot]
                stamp=rospy.Time.now();points=renderer.lidar(pos,quat,stamp.to_sec());contact=renderer.contacts()
            message=g['PointCloud'](header=g['Header'](stamp=stamp,frame_id='mid360_link'))
            message.points=[g['Point32'](*point) for point in points.tolist()];lidar_pub.publish(message)
            bumper=g['ContactsState']();bumper.header.stamp=stamp
            if contact:
                item=g['ContactState']();item.collision1_name='vla_uav::body_collision';item.collision2_name=contact;bumper.states=[item]
            contact_pub.publish(bumper)
            end=time.monotonic()
            if log:
                log.write(json.dumps(dict(capture_ros=stamp.to_sec(),publish_wall=time.time(),compute_seconds=end-begin,
                    capture_interval_seconds=None if last is None else begin-last,points=len(points),
                    sensor='Actual original 300x40 finite rays, 8m range, original 5mm noise'))+'\n')
            last=begin
            time.sleep(max(0.,.125-(time.monotonic()-begin)))
    finally:
        timer.shutdown();renderer.close()
        if log:log.close()


def image_main(world,diagnostics):
    g=imports();rospy=g['rospy'];rospy.init_node('task_attack_cpu_world')
    renderer=SceneRenderer(world);lock=threading.Lock()
    robot=[[-6.,-3.,.18],renderer.p.getQuaternionFromEuler([0,0,.74])]
    command=[sys.executable,'-m','experiments.task_attack_sensor_proxy.sensor_backend','--world',str(world),'--range-process']
    if diagnostics:command+=['--diagnostics',str(diagnostics)]
    # Same process group as the supervised image process: cleanup cannot leave
    # a hidden sensor worker publishing after the experiment has ended.
    scanner=subprocess.Popen(command)
    def stop_scanner():
        if scanner.poll() is None:
            scanner.send_signal(signal.SIGINT)
            try:scanner.wait(timeout=4)
            except subprocess.TimeoutExpired:
                scanner.terminate()
                try:scanner.wait(timeout=3)
                except subprocess.TimeoutExpired:scanner.kill();scanner.wait(timeout=2)
    atexit.register(stop_scanner)
    def update(message):
        if message.model_name!='vla_uav':return
        p,q=message.pose.position,message.pose.orientation
        with lock:robot[:]=[[p.x,p.y,p.z],[q.x,q.y,q.z,q.w]]
    rospy.Subscriber('/gazebo/set_model_state',g['ModelState'],update,queue_size=1)
    rospy.wait_for_service('/pilot/lidar/set_model_state',timeout=30)
    mirror=rospy.ServiceProxy('/pilot/lidar/set_model_state',g['SetModelState'])
    def apply(request):
        item=request.model_state;p,q=item.pose.position,item.pose.orientation
        if item.model_name not in renderer.groups:return g['SetModelStateResponse'](False,'unknown model')
        try:
            acknowledged=mirror(request)
            if not acknowledged.success:return g['SetModelStateResponse'](False,'collision mirror rejected physical edit')
            renderer.set_pose(item.model_name,[p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
            return g['SetModelStateResponse'](True,'physical edit applied to BOTH actual camera and finite-range collision worlds')
        except Exception as error:return g['SetModelStateResponse'](False,type(error).__name__)
    rospy.Service('/gazebo/set_model_state',g['SetModelState'],apply)
    rgb_pub=rospy.Publisher('/vla/camera/image_raw',g['Image'],queue_size=1)
    info_pub=rospy.Publisher('/vla/camera/camera_info',g['CameraInfo'],queue_size=1,latch=True)
    external_pub=rospy.Publisher('/pilot/external/camera/image_raw',g['Image'],queue_size=1)
    fx=640/(2*math.tan(1.3962634016/2));info=g['CameraInfo']();info.width=640;info.height=360;info.distortion_model='plumb_bob'
    info.D=[0.]*5;info.K=[fx,0,320,0,fx,180,0,0,1];info.R=[1,0,0,0,1,0,0,0,1];info.P=[fx,0,320,0,0,fx,180,0,0,0,1,0]
    def image(array,stamp,frame):
        return g['Image'](header=g['Header'](stamp=stamp,frame_id=frame),height=array.shape[0],width=array.shape[1],
            encoding='rgb8',is_bigendian=False,step=array.shape[1]*3,data=array.tobytes())
    last_external=0.;rate=rospy.Rate(3)
    try:
        while not rospy.is_shutdown():
            if scanner.poll() is not None:raise RuntimeError('Independent LiDAR/odometry process exited; sensor protection remains active')
            with renderer.lock:
                with lock:pos,quat=[list(x) for x in robot]
                renderer.set_pose('vla_uav',pos,quat);rotation=np.array(renderer.p.getMatrixFromQuaternion(quat)).reshape(3,3)
                origin=np.array(pos)+rotation.dot([.17,0,.02]);stamp=rospy.Time.now();rgb=renderer.camera(origin,0.,orientation=quat)
            rgb_pub.publish(image(rgb,stamp,'camera_optical'));info.header=g['Header'](stamp=stamp,frame_id='camera_optical');info_pub.publish(info)
            if time.monotonic()-last_external>=.9:
                stamp=rospy.Time.now();rgb=renderer.camera([0.,-9.,6.5],math.pi/2,.4,width=960,height=540,hfov=1.55,near=.1,far=35.)
                external_pub.publish(image(rgb,stamp,'pilot_external_optical'));last_external=time.monotonic()
            rate.sleep()
    finally:
        renderer.close()
        stop_scanner()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--world',type=Path,required=True)
    parser.add_argument('--range-process',action='store_true');parser.add_argument('--diagnostics',type=Path)
    args,_=parser.parse_known_args()
    (range_main if args.range_process else image_main)(args.world,args.diagnostics)
