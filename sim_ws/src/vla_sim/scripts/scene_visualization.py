#!/usr/bin/env python3
"""RViz presentation only. World geometry is never published to the planner."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import rospy
import tf.transformations as tr
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray


def transform(elem):
    value = [float(x) for x in elem.findtext('pose', '0 0 0 0 0 0').split()]
    return tr.concatenate_matrices(tr.translation_matrix(value[:3]), tr.euler_matrix(*value[3:]))


def visual_marker(visual, matrix, ident, namespace='scene'):
    m = Marker()
    m.header.frame_id, m.ns, m.id = 'world', namespace, ident
    m.action = Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = matrix[:3,3]
    m.pose.orientation = Quaternion(*tr.quaternion_from_matrix(matrix))
    geom = visual.find('geometry')
    if geom.find('box') is not None:
        m.type = Marker.CUBE
        m.scale.x, m.scale.y, m.scale.z = map(float, geom.findtext('box/size').split())
    elif geom.find('cylinder') is not None:
        m.type = Marker.CYLINDER
        m.scale.x = m.scale.y = 2*float(geom.findtext('cylinder/radius'))
        m.scale.z = float(geom.findtext('cylinder/length'))
    elif geom.find('sphere') is not None:
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 2*float(geom.findtext('sphere/radius'))
    else:
        return None
    color = list(map(float, visual.findtext('material/diffuse', '0.4 0.5 0.6 1').split()))
    m.color.r, m.color.g, m.color.b, m.color.a = color
    return m


class Visuals:
    def __init__(self):
        self.scene_pub = rospy.Publisher('/vla/visual/scene', MarkerArray, queue_size=1, latch=True)
        self.landmarks_pub = rospy.Publisher('/vla/visual/landmarks', MarkerArray, queue_size=1, latch=True)
        self.drone_pub = rospy.Publisher('/vla/visual/drone', MarkerArray, queue_size=1)
        self.path_pub = rospy.Publisher('/vla/visual/trail', Path, queue_size=1)
        self.goal_pub = rospy.Publisher('/vla/visual/goal', Marker, queue_size=1, latch=True)
        self.trail = Path()
        self.trail.header.frame_id = 'world'
        self.last_stamp = 0
        self.last_visual_stamp = -1.0
        root = ET.parse(rospy.get_param('~world_file')).getroot()
        scene = MarkerArray()
        for model in root.findall('world/model'):
            for link in model.findall('link'):
                for visual in link.findall('visual'):
                    matrix = tr.concatenate_matrices(transform(model), transform(link), transform(visual))
                    m = visual_marker(visual, matrix, len(scene.markers))
                    if m is not None:
                        scene.markers.append(m)
        with open(rospy.get_param('~landmarks_file'), encoding='utf-8') as source:
            labels = json.load(source)
        landmarks = MarkerArray()
        for name, point in labels['landmarks'].items():
            m = Marker()
            m.header.frame_id, m.ns, m.id, m.type = 'world', 'landmarks', len(landmarks.markers), Marker.TEXT_VIEW_FACING
            m.pose.position.x, m.pose.position.y, m.pose.position.z = point['x'], point['y'], 0.15
            m.pose.orientation.w = 1
            m.text = name.upper()
            m.scale.z = 0.26
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.85, 1.0, 1.0
            landmarks.markers.append(m)
        self.scene_pub.publish(scene)
        self.landmarks_pub.publish(landmarks)
        self.model = ET.parse(rospy.get_param('~model_file')).getroot().find('model')
        rospy.Subscriber('/odom', Odometry, self.odom, queue_size=1)
        rospy.Subscriber('/goal', PoseStamped, self.goal, queue_size=1)

    def odom(self, msg):
        if rospy.is_shutdown():
            return
        if not (self.drone_pub.get_num_connections() or self.path_pub.get_num_connections()):
            return
        if msg.header.stamp.to_sec()-self.last_visual_stamp < 0.04:
            return
        self.last_visual_stamp = msg.header.stamp.to_sec()
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        matrix = tr.concatenate_matrices(tr.translation_matrix([p.x,p.y,p.z]), tr.quaternion_matrix([q.x,q.y,q.z,q.w]))
        array = MarkerArray()
        for link in self.model.findall('link'):
            for visual in link.findall('visual'):
                m = visual_marker(visual, tr.concatenate_matrices(matrix,transform(link),transform(visual)), len(array.markers), 'drone')
                if m is not None:
                    array.markers.append(m)
        if rospy.is_shutdown():
            return
        self.drone_pub.publish(array)
        if msg.header.stamp.to_sec()-self.last_stamp > 0.15:
            self.last_stamp = msg.header.stamp.to_sec()
            pose = PoseStamped(header=msg.header, pose=msg.pose.pose)
            self.trail.poses.append(pose)
            self.trail.poses = self.trail.poses[-1800:]
            self.trail.header.stamp = msg.header.stamp
            self.path_pub.publish(self.trail)

    def goal(self, msg):
        m = Marker()
        m.header, m.pose, m.ns, m.id, m.type = msg.header, msg.pose, 'goal', 0, Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 0.22
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.65, 0.15, 0.85
        self.goal_pub.publish(m)


if __name__ == '__main__':
    rospy.init_node('vla_scene_visualization')
    Visuals()
    rospy.spin()
