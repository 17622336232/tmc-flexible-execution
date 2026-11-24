#!/usr/bin/env python3
"""Simulation-only acceptance for the body-mounted RGB camera and optical TF."""
import argparse,json,time,math
from pathlib import Path
import rospy,tf,cv2,numpy as np
from sensor_msgs.msg import Image,CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from quadrotor_msgs.msg import TakeoffLand
from cv_bridge import CvBridge
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',default='/tmp/vla-camera')
args,_=parser.parse_known_args()
output_dir=Path(args.output_dir)
output_dir.mkdir(parents=True,exist_ok=True)
rospy.init_node('onboard_camera_acceptance')
assert rospy.get_param('/use_sim_time',False)
assert rospy.has_param('/vla_flight_controller/takeoff_height')
state={'flight':'','odom':None,'image':None,'camera':None}
frames=[]
def image_cb(m):
 state['image']=m;frames.append((m.header.stamp.to_sec(),time.monotonic()))
rospy.Subscriber('/vla/camera/image_raw',Image,image_cb,queue_size=1)
rospy.Subscriber('/vla/camera/camera_info',CameraInfo,lambda m:state.update(camera=m),queue_size=1)
rospy.Subscriber('/odom',Odometry,lambda m:state.update(odom=m),queue_size=1)
rospy.Subscriber('/vla/flight_state',String,lambda m:state.update(flight=m.data),queue_size=1)
pub=rospy.Publisher('/px4ctrl/takeoff_land',TakeoffLand,queue_size=1)
goalpub=rospy.Publisher('/goal',PoseStamped,queue_size=1)
listener=tf.TransformListener()
def wait(fn,seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end and not rospy.is_shutdown():
  if state['flight']=='FAULT':raise RuntimeError('controller FAULT')
  if fn():return
  time.sleep(.05)
 raise RuntimeError('condition timeout: '+state['flight'])
wait(lambda:len(frames)>8 and state['camera'] and state['flight']=='GROUNDED',40)
msg=state['image'];info=state['camera']
assert (msg.width,msg.height)==(640,360) and msg.header.frame_id=='camera_optical'
assert info.header.frame_id=='camera_optical' and info.K[0]>0 and info.K[4]>0
listener.waitForTransform('base_link','camera_optical',rospy.Time(0),rospy.Duration(5))
translation,rotation=listener.lookupTransform('base_link','camera_optical',rospy.Time(0))
axes=tf.transformations.quaternion_matrix(rotation)[:3,:3]
assert np.allclose(translation,[.17,0,.02],atol=1e-5)
assert np.allclose(axes[:,2],[1,0,0],atol=1e-5)
assert np.allclose(axes[:,0],[0,-1,0],atol=1e-5)
assert np.allclose(axes[:,1],[0,0,-1],atol=1e-5)
pub.publish(TakeoffLand(takeoff_land_cmd=1))
wait(lambda:state['flight']=='HOVERING' and abs(state['odom'].pose.pose.position.z-1.2)<.04,20)
time.sleep(1)
bridge=CvBridge()
first=bridge.imgmsg_to_cv2(state['image'],desired_encoding='bgr8')
assert np.std(first)>15
cv2.imwrite(str(output_dir/'onboard-rgb.png'),first)
pose=PoseStamped();pose.header.frame_id='world';pose.header.stamp=rospy.Time.now()
pose.pose.position.x=-5;pose.pose.position.y=-3;pose.pose.position.z=1.2;pose.pose.orientation.w=1
goalpub.publish(pose)
wait(lambda:state['flight']=='HOVERING' and abs(state['odom'].pose.pose.position.x+5)<.1,30)
time.sleep(1)
second=bridge.imgmsg_to_cv2(state['image'],desired_encoding='bgr8')
cv2.imwrite(str(output_dir/'onboard-rgb-moved.png'),second)
difference=float(np.abs(first.astype(float)-second.astype(float)).mean())
assert difference>2, 'camera image did not change after aircraft translated'
pub.publish(TakeoffLand(takeoff_land_cmd=2))
wait(lambda:state['flight']=='GROUNDED',20)
span=frames[-1][0]-frames[0][0];wall=frames[-1][1]-frames[0][1]
report={'success':True,'width':msg.width,'height':msg.height,'encoding':msg.encoding,'frame':msg.header.frame_id,'camera_translation_in_body':translation,'optical_axes_in_body':axes.tolist(),'camera_fx':info.K[0],'received_frames':len(frames),'sim_image_hz':(len(frames)-1)/span,'wall_image_hz':(len(frames)-1)/wall,'realtime_factor':span/wall,'pixel_mean_absolute_change_after_1m':difference,'landed':state['flight']=='GROUNDED','ros_master':__import__('os').environ.get('ROS_MASTER_URI',''),'performance_context':'headless vehicle with RGB; other GUI tests may run concurrently'}
open(output_dir/'onboard-camera-test.json','w').write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
