"""ROS/Gazebo owner and public RGB gateway. No attack policy imports ROS."""
import base64
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
import time


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2))
    tmp.replace(path)


def append_json(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n')


class NativeNavigationError(RuntimeError):
    """The existing planner explicitly rejected/stopped this navigation."""


class GazeboBackend:
    def __init__(self, root, scene, event, *, archive_external_rgb=True):
        import rospy
        from sensor_msgs.msg import Image
        from rosgraph_msgs.msg import Log
        from gazebo_msgs.srv import SetModelState
        from bridge_ros.ros_adapter import ROSAdapter
        from uav_bridge.config import load_config
        self.root, self.scene, self.event = Path(root), scene, event
        self.lock = threading.RLock()
        self.frames = deque(maxlen=120)
        self.external_seq = 0
        self.latest_external = self.initial_external = None
        self.started = None
        self.finished = False
        self.attack_used = False
        self.attack = None
        self.previous_robot = None
        self.planner_failures = deque(maxlen=32)
        self.execution_deadline_monotonic = None
        self.last_external_wall = 0
        self.archive_external_rgb = archive_external_rgb
        config = load_config()
        config.update(mode='gazebo',navigation_mode='perception')
        config['photo']['enabled'] = False
        config['ros'] = dict(camera_fps=5.)
        self.config = config
        self.adapter = ROSAdapter(config)
        self.ros = rospy
        self.external_sub = rospy.Subscriber('/pilot/external/camera/image_raw',Image,
                                              self._external_image,queue_size=1,buff_size=4*1024*1024)
        self.planner_log_sub = rospy.Subscriber('/rosout_agg',Log,self._planner_log,queue_size=32)
        self.set_model = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)

    def _planner_log(self,msg):
        # Forward actual terminal feedback instead of silently spending the
        # entire 120-second tool budget in a planner emergency state. Do not
        # treat intermediate optimizer warnings as navigation failures.
        if msg.name!='/vla_planner' or msg.level<msg.WARN:return
        lower=msg.msg.lower()
        if 'emergency stop' not in lower and 'ignore this goal' not in lower:return
        item=dict(arrival_monotonic=time.monotonic(),ros_stamp=msg.header.stamp.to_sec(),message=msg.msg)
        with self.lock:self.planner_failures.append(item)
        self.event('native_navigation_terminal_feedback',message=msg.msg)

    def _native_failure_since(self,started):
        with self.lock:
            return next((x['message'] for x in reversed(self.planner_failures)
                if x['arrival_monotonic']>=started[0] and x['ros_stamp']>=started[1]),None)

    def _external_image(self, msg):
        import cv2
        import numpy as np
        now=time.time()
        if now-self.last_external_wall < .9:
            return
        self.last_external_wall=now
        try:
            channels=3
            raw=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)
            frame=raw[:,:msg.width*channels].reshape(msg.height,msg.width,channels)
            if msg.encoding=='rgb8':frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            elif msg.encoding!='bgr8':return
            ok,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,88])
            if not ok:return
            # A frozen pixel-only detector. No model states or TF are consulted.
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(hsv,np.array([82,170,95]),np.array([108,255,255]))
            n,_,stats,centroids=cv2.connectedComponentsWithStats(mask)
            candidates=[j for j in range(1,n) if 3 <= stats[j,4] <= 650]
            robot=None
            if candidates:
                if self.previous_robot is None:
                    selected=max(candidates,key=lambda j:stats[j,4])
                else:
                    selected=min(candidates,key=lambda j:sum((centroids[j,k]-self.previous_robot[k])**2 for k in (0,1)))
                x,y,w,h,_=map(int,stats[selected])
                self.previous_robot=centroids[selected].tolist()
                robot=dict(bbox=[x/msg.width,y/msg.height,(x+w)/msg.width,(y+h)/msg.height],
                           centroid=[float(centroids[selected,0]/msg.width),float(centroids[selected,1]/msg.height)],
                           source='external_rgb_color_components',uncertain=True)
            with self.lock:
                self.external_seq+=1
                item=dict(id=self.external_seq,capture_ros=msg.header.stamp.to_sec(),arrival_wall=now,
                          width=msg.width,height=msg.height,robot_pixels=robot,
                          sha256=hashlib.sha256(jpg.tobytes()).hexdigest(),data=jpg.tobytes())
                self.frames.append(item);self.latest_external=item
                if self.initial_external is None:self.initial_external=item
            if self.archive_external_rgb:
                path=self.root/'public/external'/('%06d.jpg'%item['id'])
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(item['data'])
                append_json(self.root/'public/external_frames.jsonl',{k:v for k,v in item.items() if k!='data'})
        except Exception as error:
            self.event('external_camera_error',type=type(error).__name__)

    def wait_ready(self, timeout=120):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            s=self.adapter.snapshot()
            if s['connected'] and s['planning_ready'] and s['camera_status'].get('available') and self.latest_external:
                self.event('sensors_ready',position=s['position'],mode=s['mode'])
                return
            time.sleep(.2)
        raise RuntimeError('ROS / RGB / LiDAR / external RGB not ready within startup timeout')

    def start(self):
        with self.lock:self.started=time.time()

    def public_state(self):
        with self.lock:
            return dict(elapsed_seconds=0. if self.started is None else time.time()-self.started,
                        started=self.started is not None,finished=self.finished,
                        edit_used=self.attack_used,external_available=self.latest_external is not None)

    @staticmethod
    def serialize_frame(frame):
        return dict({k:v for k,v in frame.items() if k!='data'},
                    jpeg_base64=base64.b64encode(frame['data']).decode('ascii'))

    def public_frames(self, initial=False):
        with self.lock:
            if initial:
                frames=[self.initial_external] if self.initial_external else []
                tracks=[]
            else:
                all_frames=list(self.frames)
                frames=[]
                if all_frames:
                    latest=all_frames[-1]
                    def visual_distance(frame):
                        a=frame.get('robot_pixels')
                        ends=[f.get('robot_pixels') for f in (all_frames[0],latest)]
                        if not a or not all(ends):return 0.
                        return min(sum((x-y)**2 for x,y in zip(a['centroid'],b['centroid'])) for b in ends)
                    # Retain actual motion across a long inference hover.
                    for candidate in (all_frames[0],max(all_frames,key=visual_distance),latest):
                        if candidate['id'] not in {f['id'] for f in frames}:
                            frames.append(candidate)
                    frames.sort(key=lambda f:f['id'])
                tracks=[{k:f[k] for k in ('id','capture_ros','arrival_wall','robot_pixels')} for f in all_frames]
            return dict(state=self.public_state(),frames=[self.serialize_frame(f) for f in frames],
                        external_pixel_history=tracks)

    def edit(self, action_id):
        from gazebo_msgs.msg import ModelState
        with self.lock:
            if self.started is None or self.finished:raise ValueError('Episode is not active')
            if self.attack_used:raise ValueError('The one-edit experimental budget has been consumed')
            action=next((x for x in self.scene['catalog']['editable_actions'] if x['action_id']==action_id),None)
            if action is None:raise ValueError('Unknown public action handle')
            self.attack_used=True
        state=self.scene['action_states'][action_id]
        request=ModelState();request.model_name=state['model_name'];request.reference_frame='world'
        request.pose.position.x,request.pose.position.y,request.pose.position.z=state['position']
        request.pose.orientation.z=math.sin(state['yaw']/2);request.pose.orientation.w=math.cos(state['yaw']/2)
        start=time.time();start_ros=self.ros.Time.now().to_sec();before=self.external_seq
        try:
            result=self.set_model(request)
            record=dict(action_id=action_id,dispatch_wall=start,ack_wall=time.time(),
                        dispatch_ros=start_ros,ack_ros=self.ros.Time.now().to_sec(),
                        service_success=bool(result.success),last_external_frame_before=before)
        except Exception as error:
            record=dict(action_id=action_id,dispatch_wall=start,ack_wall=time.time(),
                        dispatch_ros=start_ros,ack_ros=self.ros.Time.now().to_sec(),
                        service_success=False,error_type=type(error).__name__,last_external_frame_before=before)
        with self.lock:self.attack=record
        write_json(self.root/'private/edit.json',record)
        self.event('physical_edit',**record)
        # Public acknowledgment reveals no downstream perception or score.
        return dict(acknowledged=True,budget_consumed=True)

    def observe(self):
        import numpy as np
        # Native arrival permits a small residual yaw error; an RGB/range pair
        # captured during that final turn can legitimately fail registration.
        # Wait for fresh measured samples within the existing local-check
        # budget; never loosen motion/skew checks or rerun a model decision.
        until=time.monotonic()+5.
        while True:
            try:
                obs=self.adapter.visual_observation();break
            except RuntimeError as error:
                transient=any(text in str(error) for text in ('RGB 与 LiDAR 采样期间',
                    'RGB 与实际 LiDAR 时间差过大','CameraInfo 与 RGB 时间不同步'))
                if not transient or time.monotonic()>=until:raise
                time.sleep(.05)
        context=self.adapter.photo_camera_context(obs,include_demo_targets=False)
        with self.adapter._visual.lock:
            projection=self.adapter._visual.registry[obs['observation_id']]['projection']
            uv=projection['uv'];pts=projection['points_world'];depth=projection['depth']
            grid=[]
            for row in range(4):
                for col in range(8):
                    sel=((uv[:,0]>=col*obs['width']/8)&(uv[:,0]<(col+1)*obs['width']/8)&
                         (uv[:,1]>=row*obs['height']/4)&(uv[:,1]<(row+1)*obs['height']/4))
                    indices=np.flatnonzero(sel)
                    if len(indices):
                        j=indices[np.argsort(depth[indices])[len(indices)//2]]
                        grid.append(dict(pixel_cell=[col,row],world_xyz=[round(float(x),3) for x in pts[j]],
                                         measured_points=int(len(indices))))
        return obs,dict(position=obs['position'],yaw=obs['yaw'],camera=context,
                        measured_surface_grid=dict(columns=8,rows=4,cells=grid,
                            description='Median real LiDAR hit per image cell. These are obstacle surfaces, not free viewpoints.'))

    def wait_motion(self, goal=None, timeout=120, landing=False,native_request_started=None):
        end=time.monotonic()+timeout;stable=None
        deadline=getattr(self,'execution_deadline_monotonic',None)
        if deadline is not None:end=min(end,deadline)
        while time.monotonic()<end:
            if self.ros.is_shutdown():raise RuntimeError('ROS shutdown')
            if native_request_started is not None:
                failure=self._native_failure_since(native_request_started)
                if failure:raise NativeNavigationError('Original planner stopped this request: '+failure)
            state=self.adapter.snapshot()
            if state.get('command_error'):raise RuntimeError(state['command_error'])
            if landing:
                reached=state['landed'] and not state['armed']
            else:
                distance=math.sqrt(sum((state['position'][a]-goal[a])**2 for a in ('x','y','z')))
                yaw=abs(math.atan2(math.sin(state['yaw']-goal['yaw']),math.cos(state['yaw']-goal['yaw'])))
                speed=math.sqrt(sum(v*v for v in state['velocity'].values()))
                reached=(state['connected'] and state['flight_mode'] in ('HOVERING','FLYING')
                         and not state['planning_request_pending'] and distance<=self.config['arrival_tolerance_m']
                         and yaw<=math.radians(10) and speed<=self.config['arrival_speed_mps'])
            if reached:
                if stable is None:stable=time.monotonic()
                if time.monotonic()-stable>=self.config['arrival_dwell_seconds']:return state
            else:stable=None
            time.sleep(.1)
        state=self.adapter.snapshot()
        if deadline is not None and time.monotonic()>=deadline:
            raise RuntimeError('Experimental episode budget exhausted during navigation')
        detail={k:state.get(k) for k in ('position','velocity','yaw','flight_mode','planning_request_pending','command_error')}
        raise RuntimeError('Original navigation timeout: '+json.dumps(detail,separators=(',',':')))

    def takeoff(self):
        s=self.adapter.snapshot();self.adapter.takeoff(self.config['takeoff_height'])
        return self.wait_motion(dict(s['position'],z=self.config['takeoff_height'],yaw=s['yaw']))

    def check_goal(self, goal):
        if set(goal)!={'x','y','z','yaw'} or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in goal.values()):
            raise ValueError('move requires finite x/y/z and yaw in radians')
        return self.adapter._goal_query.ensure(goal)

    def move(self, goal):
        # Reuse the original mapper's existing goal-admission query. Otherwise
        # the original FSM rejects an outside-local-map final point and returns
        # to WAIT_TARGET, leaving a direct adapter caller waiting for 120s.
        # This is native map feedback, not a new radius / step-size restriction.
        self.check_goal(goal)
        requested=(time.monotonic(),self.ros.Time.now().to_sec())
        self.adapter.goto(**goal)
        try:return self.wait_motion(goal,native_request_started=requested)
        except Exception:
            self.adapter.hold()
            raise

    def approach(self, observation, bbox, standoff):
        measured=self.adapter.resolve_visual_target(observation,bbox,standoff)
        self.move(measured['goal'])
        return measured

    def close(self):
        with self.lock:self.finished=True
        self.external_sub.unregister()
        self.planner_log_sub.unregister()
        self.adapter.close()


class PublicGateway:
    """Only a narrow RGB/menu/edit API is exposed to the observer subprocess."""
    def __init__(self, backend, capability):
        owner=self
        self.backend=backend
        self.capability=capability
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def reply(self,value,status=200):
                raw=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
                self.send_response(status);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def do_GET(self):
                if not self.path.startswith('/'+capability+'/'):return self.reply({'error':'not found'},404)
                path=self.path.split('?')[0].rsplit('/',1)[-1]
                if path=='catalog':return self.reply(backend.scene['catalog'])
                if path=='state':return self.reply(backend.public_state())
                if path=='frames':return self.reply(backend.public_frames())
                if path=='initial':return self.reply(backend.public_frames(initial=True))
                self.reply({'error':'not found'},404)
            def do_POST(self):
                if self.path!='/'+capability+'/edit':return self.reply({'error':'not found'},404)
                try:
                    length=int(self.headers.get('Content-Length','0'))
                    if not 0<length<4096:raise ValueError('Invalid request size')
                    body=json.loads(self.rfile.read(length))
                    if not isinstance(body,dict) or set(body)!={'action_id'}:raise ValueError('Only action_id is accepted')
                    self.reply(backend.edit(body['action_id']))
                except (ValueError,TypeError) as error:self.reply({'error':str(error)},400)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url='http://127.0.0.1:%d/%s'%(self.server.server_port,capability)

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
