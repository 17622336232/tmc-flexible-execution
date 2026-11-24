"""Optional CPU renderer for background experiments; not Gazebo or motor SITL.

Reads the SAME generated SDF geometry/textures. Actual pinhole RGB and finite
range ray tests feed the unchanged ROS flight controller, mapper and planner.
No task, scoring truth, synthetic detections or canned routes enter this node.
"""
import argparse
import math
from pathlib import Path
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np


def bullet_module():
    try:
        import pybullet
    except ImportError:
        dependency = Path(__file__).resolve().parents[2]/'.runtime/task_attack_cpu_deps'
        if not dependency.is_dir():
            raise RuntimeError('Install the pinned optional CPU dependency; see experiment README')
        sys.path.insert(0, str(dependency))
        import pybullet
    return pybullet


def numbers(node, name, default):
    value = node.findtext(name)
    return list(map(float, value.split())) if value else list(default)


class SceneRenderer:
    def __init__(self, world, robot=True):
        self.p = bullet_module()
        self.client = self.p.connect(self.p.DIRECT)
        self.groups = {}
        self.poses = {}
        self.collision_bodies = []
        self.root = Path(world).resolve().parent
        self.lock = threading.RLock()
        self.textures = {}
        self.plane = self.root/'assets/cpu_panel.obj'
        self.plane.write_text('v -0.5 0 -0.5\nv 0.5 0 -0.5\nv 0.5 0 0.5\nv -0.5 0 0.5\n'
                              'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n'
                              'f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n'
                              'f 3/3 2/2 1/1\nf 4/4 3/3 1/1\n')
        sdf = ET.parse(str(world)).getroot().find('world')
        for model in sdf.findall('model'):
            self._model(model, model.attrib['name'])
        if robot:
            robot_path = Path(__file__).resolve().parents[2]/'sim_ws/src/vla_sim/models/vla_uav/model_indoor.sdf'
            self._model(ET.parse(str(robot_path)).getroot().find('model'), 'vla_uav', collisions=True)
            self.set_pose('vla_uav', [-6., -3., .18], self.p.getQuaternionFromEuler([0,0,.74]))

    def _compose(self, a, b):
        return self.p.multiplyTransforms(a[:3], self.p.getQuaternionFromEuler(a[3:]),
                                         b[:3], self.p.getQuaternionFromEuler(b[3:]))

    def _model(self, model, name, collisions=True):
        self.groups[name] = []
        origin = numbers(model, 'pose', [0]*6)
        for link in model.findall('link'):
            local = numbers(link, 'pose', [0]*6)
            for item in list(link.findall('visual')) + (list(link.findall('collision')) if collisions else []):
                if item.find('geometry') is None:
                    continue
                pos, orn = self._compose(local, numbers(item,'pose',[0]*6))
                geometry = item.find('geometry')
                kwargs = {}
                if geometry.find('box') is not None:
                    shape = self.p.GEOM_BOX
                    kwargs['halfExtents'] = [v/2 for v in numbers(geometry,'box/size',[1,1,1])]
                elif geometry.find('cylinder') is not None:
                    shape = self.p.GEOM_CYLINDER
                    kwargs.update(radius=float(geometry.findtext('cylinder/radius')),
                                  length=float(geometry.findtext('cylinder/length')))
                elif geometry.find('sphere') is not None:
                    shape = self.p.GEOM_SPHERE
                    kwargs['radius'] = float(geometry.findtext('sphere/radius'))
                elif geometry.find('mesh') is not None:
                    uri = geometry.findtext('mesh/uri','')
                    if not uri.endswith('panel.dae'):
                        raise ValueError('Unsupported CPU mesh: '+uri)
                    shape = self.p.GEOM_MESH
                    kwargs.update(fileName=str(self.plane),meshScale=numbers(geometry,'mesh/scale',[1,1,1]))
                else:
                    raise ValueError('Unsupported CPU geometry')
                collision = item.tag == 'collision'
                if collision:
                    if 'length' in kwargs: kwargs['height'] = kwargs.pop('length')
                    collision_id = self.p.createCollisionShape(shape, physicsClientId=self.client, **kwargs)
                    visual_id = -1
                else:
                    color = numbers(item,'material/diffuse',[.6,.65,.7,1])
                    visual_id = self.p.createVisualShape(shape,rgbaColor=color,physicsClientId=self.client,**kwargs)
                    collision_id = -1
                body = self.p.createMultiBody(baseMass=0,baseCollisionShapeIndex=collision_id,
                    baseVisualShapeIndex=visual_id,basePosition=pos,baseOrientation=orn,physicsClientId=self.client)
                if collision:
                    # Bullet otherwise draws an implicit grey copy of each
                    # collision shape, causing coplanar z-fighting with RGB.
                    self.p.changeVisualShape(body,-1,rgbaColor=[0,0,0,0],physicsClientId=self.client)
                material = item.findtext('material/script/name','')
                if not collision and material.startswith('Pilot/'):
                    texture = self.root/'assets/materials/textures'/(material.split('/',1)[1]+'.png')
                    key = str(texture)
                    if key not in self.textures:
                        self.textures[key] = self.p.loadTexture(key,physicsClientId=self.client)
                    self.p.changeVisualShape(body,-1,textureUniqueId=self.textures[key],
                                             rgbaColor=[1,1,1,1],physicsClientId=self.client)
                self.groups[name].append((body,list(pos),list(orn)))
                if collision:self.collision_bodies.append((name,body))
        self.set_pose(name,origin[:3],self.p.getQuaternionFromEuler(origin[3:]))

    def set_pose(self, name, position, orientation):
        if name not in self.groups:raise ValueError('Unknown rendered model: '+name)
        with self.lock:
            for body,pos,orn in self.groups[name]:
                xyz,quat = self.p.multiplyTransforms(position,orientation,pos,orn)
                self.p.resetBasePositionAndOrientation(body,xyz,quat,physicsClientId=self.client)
            self.poses[name] = (list(position),list(orientation))

    def camera(self, position, yaw, pitch=0., width=640, height=360, hfov=1.3962634016, near=.05, far=30., orientation=None):
        with self.lock:
            if orientation is None:
                forward = np.array([math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),-math.sin(pitch)])
                up = np.array([math.sin(pitch)*math.cos(yaw),math.sin(pitch)*math.sin(yaw),math.cos(pitch)])
            else:
                rotation=np.array(self.p.getMatrixFromQuaternion(orientation)).reshape(3,3)
                forward,up=rotation[:,0],rotation[:,2]
            view = self.p.computeViewMatrix(list(position),(np.array(position)+forward).tolist(),up.tolist())
            vfov = math.degrees(2*math.atan(math.tan(hfov/2)*height/width))
            projection = self.p.computeProjectionMatrixFOV(vfov,width/height,near,far)
            result = self.p.getCameraImage(width,height,view,projection,shadow=0,
                lightDirection=[-1,-2,4],lightAmbientCoeff=.7,lightDiffuseCoeff=.3,
                renderer=self.p.ER_TINY_RENDERER,physicsClientId=self.client)
            return np.asarray(result[2],dtype=np.uint8).reshape(height,width,4)[:,:,:3].copy()

    def lidar(self, position, yaw, stamp, noise_seed=360):
        horizontal = np.linspace(-math.pi,math.pi,300)
        vertical = np.linspace(-.12217304764,.90757121104,40)
        h,v = np.meshgrid(horizontal,vertical)
        local = np.stack((np.cos(v)*np.cos(h),np.cos(v)*np.sin(h),np.sin(v)),axis=-1).reshape(-1,3)
        if isinstance(yaw,(list,tuple,np.ndarray)):
            rotation=np.array(self.p.getMatrixFromQuaternion(yaw)).reshape(3,3)
            orientation=list(yaw)
        else:
            rotation = np.array([[math.cos(yaw),-math.sin(yaw),0],
                                 [math.sin(yaw), math.cos(yaw),0],[0,0,1]])
            orientation=self.p.getQuaternionFromEuler([0,0,yaw])
        origin = np.array(position)+rotation.dot([0,0,.155])
        directions = local.dot(rotation.T)
        with self.lock:
            # RGB and range run at different rates. A ray must see the robot
            # collision body at THIS acquisition pose, not the last 3-Hz RGB
            # pose. Otherwise the sensor hits its own stale body and deposits
            # phantom obstacles along the flight path. Keep real collisions;
            # do not mask self hits or weaken the mapper's clearance checks.
            if 'vla_uav' in self.groups:self.set_pose('vla_uav',position,orientation)
            returns = self.p.rayTestBatch(np.tile(origin,(len(local),1)).tolist(),
                (origin+8.*directions).tolist(),numThreads=2,physicsClientId=self.client)
        distances = np.array([8.*r[2] if r[0]>=0 else 8. for r in returns])
        hit = distances<8.
        rng = np.random.RandomState((int(stamp*1e6)^noise_seed)&0xffffffff)
        distances[hit] = np.clip(distances[hit]+rng.normal(0,.005,int(hit.sum())),.1,7.99)
        return (local*distances[:,None]).astype(np.float32)

    def close(self):
        self.p.disconnect(physicsClientId=self.client)

    def contacts(self):
        with self.lock:
            robot=[body for name,body in self.collision_bodies if name=='vla_uav']
            for name,other in self.collision_bodies:
                if name=='vla_uav':continue
                for body in robot:
                    if self.p.getClosestPoints(body,other,0.,physicsClientId=self.client):return name
        return None


def ros_main(world):
    import rospy
    from gazebo_msgs.msg import ModelState,ModelStates,ContactsState,ContactState
    from gazebo_msgs.srv import SetModelState,SetModelStateResponse
    from geometry_msgs.msg import Pose,Twist,Point32
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image,CameraInfo,PointCloud
    from std_msgs.msg import Header
    rospy.init_node('task_attack_cpu_world')
    renderer = SceneRenderer(world)
    state_lock = threading.Lock()
    robot_pose = [[-6.,-3.,.18],renderer.p.getQuaternionFromEuler([0,0,.74])]
    external = ([0.,-9.,6.5],math.pi/2,.4)
    pub_models = rospy.Publisher('/gazebo/model_states',ModelStates,queue_size=1)
    pub_rgb = rospy.Publisher('/vla/camera/image_raw',Image,queue_size=1)
    pub_info = rospy.Publisher('/vla/camera/camera_info',CameraInfo,queue_size=1,latch=True)
    pub_ext = rospy.Publisher('/pilot/external/camera/image_raw',Image,queue_size=1)
    pub_lidar = rospy.Publisher('/vla/lidar/points_raw',PointCloud,queue_size=1)
    pub_contacts = rospy.Publisher('/vla/uav_contacts',ContactsState,queue_size=1)
    def update(message):
        if message.model_name != 'vla_uav':return
        p,q = message.pose.position,message.pose.orientation
        with state_lock:robot_pose[:] = [[p.x,p.y,p.z],[q.x,q.y,q.z,q.w]]
    def service(request):
        m = request.model_state;p,q = m.pose.position,m.pose.orientation
        try:
            renderer.set_pose(m.model_name,[p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
            return SetModelStateResponse(True,'physical rendered model pose changed')
        except ValueError as e:return SetModelStateResponse(False,str(e))
    rospy.Subscriber('/gazebo/set_model_state',ModelState,update,queue_size=1)
    rospy.Service('/gazebo/set_model_state',SetModelState,service)
    def models(_):
        with state_lock:pos,quat = [list(x) for x in robot_pose]
        msg = ModelStates();msg.name=['vla_uav'];pose=Pose()
        pose.position.x,pose.position.y,pose.position.z=pos
        pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=quat
        msg.pose=[pose];msg.twist=[Twist()];pub_models.publish(msg)
    timer=rospy.Timer(rospy.Duration(.05),models)
    fx=640/(2*math.tan(1.3962634016/2))
    info=CameraInfo();info.width=640;info.height=360;info.distortion_model='plumb_bob'
    info.D=[0.]*5;info.K=[fx,0,320,0,fx,180,0,0,1];info.R=[1,0,0,0,1,0,0,0,1]
    info.P=[fx,0,320,0,0,fx,180,0,0,0,1,0]
    def image(array,stamp,frame):
        msg=Image(header=Header(stamp=stamp,frame_id=frame),height=array.shape[0],width=array.shape[1],
                  encoding='rgb8',is_bigendian=False,step=array.shape[1]*3,data=array.tobytes())
        return msg
    stop=threading.Event()
    def scan_loop():
        while not stop.is_set() and not rospy.is_shutdown():
            tick=time.monotonic()
            with state_lock:pos,quat=[list(x) for x in robot_pose]
            # Acquisition time is taken after gaining the renderer lock,
            # never before a possibly long camera render wait.
            with renderer.lock:
                with state_lock:pos,quat=[list(x) for x in robot_pose]
                stamp=rospy.Time.now()
                points=renderer.lidar(pos,quat,stamp.to_sec())
            msg=PointCloud(header=Header(stamp=stamp,frame_id='mid360_link'))
            msg.points=[Point32(*point) for point in points.tolist()];pub_lidar.publish(msg)
            stop.wait(max(0.,.125-(time.monotonic()-tick)))
    scanner=threading.Thread(target=scan_loop,daemon=True);scanner.start()
    last_external=0.
    try:
        rate=rospy.Rate(3)
        while not rospy.is_shutdown():
            with renderer.lock:
                with state_lock:pos,quat=[list(x) for x in robot_pose]
                renderer.set_pose('vla_uav',pos,quat)
                contact=renderer.contacts()
            contact_msg=ContactsState();contact_msg.header.stamp=rospy.Time.now()
            if contact:
                state=ContactState();state.collision1_name='vla_uav::body_collision';state.collision2_name=contact
                contact_msg.states=[state]
            pub_contacts.publish(contact_msg)
            with renderer.lock:
                with state_lock:pos,quat=[list(x) for x in robot_pose]
                renderer.set_pose('vla_uav',pos,quat)
                rotation=np.array(renderer.p.getMatrixFromQuaternion(quat)).reshape(3,3)
                origin=np.array(pos)+rotation.dot([.17,0,.02])
                stamp=rospy.Time.now()
                rgb=renderer.camera(origin,0.,orientation=quat)
            pub_rgb.publish(image(rgb,stamp,'camera_optical'))
            info.header=Header(stamp=stamp,frame_id='camera_optical');pub_info.publish(info)
            # Yield fairly to the independent range acquisition thread.
            # Sensor freshness thresholds remain unchanged.
            time.sleep(.015)
            if time.monotonic()-last_external>=.9:
                ext_stamp=rospy.Time.now()
                rgb=renderer.camera(*external,width=960,height=540,hfov=1.55,near=.1,far=35.)
                pub_ext.publish(image(rgb,ext_stamp,'pilot_external_optical'));last_external=time.monotonic()
                time.sleep(.015)
            rate.sleep()
    finally:
        stop.set();scanner.join(timeout=2);timer.shutdown();renderer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world',required=True)
    args,_=parser.parse_known_args()
    ros_main(args.world)
