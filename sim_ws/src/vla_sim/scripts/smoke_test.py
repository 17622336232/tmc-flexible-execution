#!/usr/bin/env python3
"""Headless integration acceptance: real ray cloud + Diff-Planner + controller.
Run only against the dedicated simulation ROS master, never an aircraft.
"""
import argparse
import json
import math
import time
import rospy
import tf.transformations as tr
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import TakeoffLand, PositionCommand
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, Empty


class Probe:
    def __init__(self):
        self.odom, self.state, self.clouds, self.commands, self.points = None, '', 0, 0, 0
        self.detail = {}
        self.samples = []
        self.target = None
        rospy.Subscriber('/odom', Odometry, lambda msg: setattr(self, 'odom', msg))
        rospy.Subscriber('/vla/flight_state', String, lambda msg: setattr(self, 'state', msg.data))
        rospy.Subscriber('/cloud_registered', PointCloud2, self.cloud)
        rospy.Subscriber('/position_cmd', PositionCommand, self.command)
        rospy.Subscriber('/diff_vla/sim/state', String, lambda msg: setattr(self, 'detail', json.loads(msg.data)))
        self.takeoff_pub = rospy.Publisher('/px4ctrl/takeoff_land', TakeoffLand, queue_size=1)
        self.goal_pub = rospy.Publisher('/goal', PoseStamped, queue_size=1)
        self.hold_pub = rospy.Publisher('/vla/hold', Empty, queue_size=1)
        self.desired_yaw = 0.0
        self.yaw_pub = rospy.Publisher('/planning/yaw', PositionCommand, queue_size=1)
        self.yaw_timer = rospy.Timer(rospy.Duration(0.1), self.yaw_command)
        self.start = time.monotonic()
        self.sim_start = rospy.get_time()
        self.results = []

    def yaw_command(self, _):
        self.yaw_pub.publish(PositionCommand(yaw=self.desired_yaw))

    def yaw_error(self):
        if not self.odom: return 999
        q=self.odom.pose.pose.orientation
        actual=tr.euler_from_quaternion([q.x,q.y,q.z,q.w])[2]
        return abs(math.atan2(math.sin(actual-self.desired_yaw),math.cos(actual-self.desired_yaw)))

    def cloud(self,msg):
        self.clouds += 1
        self.points = max(self.points, msg.width*msg.height)

    def command(self,msg):
        self.commands += 1
        self.target = [msg.position.x,msg.position.y,msg.position.z]
        if self.odom and self.commands % 5 == 0:
            self.samples.append({'t':rospy.get_time(),'p':self.position(),'target':self.target,'id':msg.trajectory_id,'state':self.state})

    def wait(self, condition, timeout, label):
        until = time.monotonic()+timeout
        while not rospy.is_shutdown() and time.monotonic() < until:
            if self.state == 'FAULT':
                raise RuntimeError('Collision/fault: '+str(self.detail))
            if condition():
                self.results.append({'check':label,'elapsed_s':round(time.monotonic()-self.start,2),'state':self.state})
                print(json.dumps(self.results[-1]),flush=True)
                return
            time.sleep(.05)
        raise RuntimeError(label+' timed out; state='+self.state+' position='+str(self.position()))

    def position(self):
        if self.odom is None: return None
        p=self.odom.pose.pose.position
        return [p.x,p.y,p.z]

    def distance(self,target):
        return math.sqrt(sum((a-b)**2 for a,b in zip(self.position(),target))) if self.odom else 1e9

    def goal(self,target):
        m=PoseStamped()
        m.header.stamp=rospy.Time.now()
        m.header.frame_id='world'
        m.pose.position.x,m.pose.position.y,m.pose.position.z=target
        m.pose.orientation.w=1
        self.goal_pub.publish(m)

    def run(self,full):
        self.wait(lambda:self.odom and self.clouds>8 and self.detail.get('connected'),45,'lidar and Gazebo ready')
        assert self.state=='GROUNDED' and abs(self.position()[2]-.18)<.03,'must start grounded'
        before=self.position()
        time.sleep(2)
        assert self.distance(before)<.02,'spontaneous flight'
        self.takeoff_pub.publish(TakeoffLand(takeoff_land_cmd=TakeoffLand.TAKEOFF))
        self.wait(lambda:self.state=='HOVERING' and abs(self.position()[2]-1.2)<.06,20,'takeoff to 1.2 m')
        # First navigation must be a direct landmark request from the launch pad;
        # a preparatory waypoint can hide a rolling-target-in-obstacle failure.
        route=[(0,0,1.2)]
        if full: route += [(-6,-3,1.2),(5,2,1.2),(-6,-3,1.2),(5,-3,1.2)]
        route += [(-6,-3,1.2)]
        for target in route:
            old=self.commands
            self.desired_yaw = math.pi/2 if target == (5,2,1.2) else 0.0
            self.goal(target)
            self.wait(lambda:self.commands>old+10,10,'planner publishes new trajectory')
            self.wait(lambda:self.distance(target)<.20 and self.state=='HOVERING',90,'arrived '+str(target))
            self.wait(lambda:self.yaw_error()<.08,10,'heading aligned '+str(round(self.desired_yaw,3)))
        self.desired_yaw = -math.pi/2
        self.wait(lambda:self.yaw_error()<.08,10,'endpoint yaw control')
        self.hold_pub.publish(Empty())
        time.sleep(1)
        before=self.position()
        time.sleep(2)
        assert self.distance(before)<.06,'hold revived old trajectory'
        self.takeoff_pub.publish(TakeoffLand(takeoff_land_cmd=TakeoffLand.LAND))
        self.wait(lambda:self.state=='GROUNDED',20,'landed and disarmed')
        time.sleep(2)
        assert self.position()[2]<.21,'old trajectory revived after landing'
        errors=sorted(math.sqrt(sum((a-b)**2 for a,b in zip(row['p'],row['target']))) for row in self.samples if row['state']=='FLYING')
        assert errors and max(errors)<0.25, 'trajectory tracking error exceeds 0.25 m'
        tracking={'mean':sum(errors)/len(errors),'p95':errors[int(.95*(len(errors)-1))],'max':max(errors)}
        return {'success':True,'tracking_error_m':tracking,'checks':self.results,'cloud_frames':self.clouds,'max_cloud_points':self.points,'position_commands':self.commands,'contacts':self.detail.get('contacts'),'realtime_factor':round((rospy.get_time()-self.sim_start)/(time.monotonic()-self.start),3),'elapsed_s':round(time.monotonic()-self.start,2)}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--cycles',type=int,default=1)
    opts,_=parser.parse_known_args()
    rospy.init_node('vla_smoke_test')
    if not rospy.get_param('/use_sim_time',False) or not rospy.has_param('/vla_flight_controller/takeoff_height'):
        raise SystemExit('Refusing to run: dedicated VLA simulation controller not detected')
    p=Probe()
    try:
        if not 1 <= opts.cycles <= 5:
            raise ValueError('--cycles must be between 1 and 5')
        for cycle in range(opts.cycles):
            result=p.run(opts.full)
        result['completed_cycles']=opts.cycles
    except Exception as exc:
        p.hold_pub.publish(Empty())
        result={'success':False,'error':str(exc),'checks':p.results}
    text=json.dumps(result,indent=2)
    print(text)
    if opts.output:
        with open(opts.output,'w') as f: f.write(text+'\n')
        with open(opts.output+'.trace.json','w') as f: json.dump(p.samples,f)
    raise SystemExit(0 if result['success'] else 1)
