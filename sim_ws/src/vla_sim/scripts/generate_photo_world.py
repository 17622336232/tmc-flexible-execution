#!/usr/bin/env python3
"""Generate the independent photo courtyard; never load landmark coordinates.

The standing subject is an attributed textured Gazebo mesh with matching mesh
collision. All placement below is simulator geometry, not a planner/VLM prior.
No ROS nodes, sensors, controllers or perception metadata are published here.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    'paving': '.66 .69 .68 1', 'seam': '.49 .53 .52 1',
    'cream': '.82 .80 .73 1', 'stone': '.61 .66 .65 1',
    'white': '.93 .93 .86 1', 'wood': '.48 .29 .15 1',
    'wood_light': '.64 .43 .24 1', 'teal': '.10 .33 .35 1',
    'grass': '.24 .39 .20 1', 'soil': '.24 .19 .13 1',
    'leaf': '.17 .34 .18 1', 'leaf_light': '.28 .44 .22 1',
    'metal': '.21 .27 .27 1', 'gold': '.76 .58 .22 1',
}


def add(parent, tag, value=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if value is not None:
        node.text = str(value)
    return node


def numbers(values):
    return ' '.join(format(x, '.9g') for x in values)


def model(world, name, pose=(0, 0, 0, 0, 0, 0)):
    node = add(world, 'model', name=name)
    add(node, 'static', 'true')
    add(node, 'pose', numbers(pose))
    return add(node, 'link', name='body')


def shape(link, name, kind, dimensions, position, color, rotation=(0, 0, 0)):
    for role in ('visual', 'collision'):
        part = add(link, role, name=name+'_'+role)
        add(part, 'pose', numbers((*position, *rotation)))
        geom = add(add(part, 'geometry'), kind)
        if kind == 'box':
            add(geom, 'size', numbers(dimensions))
        elif kind == 'cylinder':
            add(geom, 'radius', dimensions[0]); add(geom, 'length', dimensions[1])
        elif kind == 'sphere':
            add(geom, 'radius', dimensions[0])
        if role == 'visual':
            material = add(part, 'material')
            add(material, 'ambient', COLORS[color])
            add(material, 'diffuse', COLORS[color])
            add(material, 'specular', '.08 .08 .08 1')
            add(part, 'cast_shadows', 'false')


def box(link, name, size, position, color):
    shape(link, name, 'box', size, position, color)


def tree(world, name, x, y, height=2.8):
    link = model(world, name, (x, y, 0, 0, 0, 0))
    box(link, 'planter', (.95, .95, .34), (0, 0, .17), 'stone')
    box(link, 'soil', (.83, .83, .03), (0, 0, .35), 'soil')
    shape(link, 'trunk', 'cylinder', (.105, height-.7), (0, 0, (height-.7)/2+.3), 'wood')
    shape(link, 'crown', 'sphere', (.72,), (0, 0, height-.35), 'leaf')
    shape(link, 'crown_side', 'sphere', (.52,), (.32, -.14, height-.5), 'leaf_light')


def bench(world, name, x, y):
    link = model(world, name, (x, y, 0, 0, 0, 0))
    for i, px in enumerate((-.7, .7)):
        box(link, 'leg_'+str(i), (.09, .54, .43), (px, 0, .215), 'metal')
    box(link, 'seat', (1.8, .58, .09), (0, 0, .475), 'wood_light')
    box(link, 'back', (1.8, .08, .48), (0, .27, .79), 'wood_light')
    box(link, 'rail', (1.8, .075, .05), (0, .27, 1.055), 'metal')


def build():
    sdf = ET.Element('sdf', version='1.6')
    world = add(sdf, 'world', name='vla_photo_courtyard')
    add(world, 'gravity', '0 0 -9.81')
    physics = add(world, 'physics', name='default_physics', type='ode')
    add(physics, 'max_step_size', '.005')
    add(physics, 'real_time_update_rate', '200')
    solver = add(add(physics, 'ode'), 'solver')
    add(solver, 'type', 'quick'); add(solver, 'iters', '30')
    scene = add(world, 'scene')
    add(scene, 'ambient', '.56 .58 .58 1')
    add(scene, 'background', '.68 .79 .87 1')
    add(scene, 'shadows', 'false'); add(scene, 'grid', 'false')
    camera = add(add(world, 'gui', fullscreen='0'), 'camera', name='courtyard_overview')
    add(camera, 'pose', '-8.5 -8 5.8 0 .610379 .816316')
    add(camera, 'view_controller', 'orbit'); add(camera, 'projection_type', 'perspective')
    light = add(world, 'light', name='afternoon_daylight', type='directional')
    add(light, 'pose', '-7 -4 10 0 0 0')
    add(light, 'diffuse', '.68 .65 .58 1')
    add(light, 'specular', '.08 .08 .08 1')
    add(light, 'direction', '.55 .12 -.83')
    add(light, 'cast_shadows', 'false')

    floor = model(world, 'courtyard_paving')
    box(floor, 'slab', (18.0, 13.0, .12), (0, 0, -.06), 'paving')
    for i, x in enumerate(range(-8,9,2)):
        box(floor, 'joint_x_'+str(i), (.012, 12.8, .002), (x, 0, .001), 'seam')
    for i, y in enumerate(range(-6,7,2)):
        box(floor, 'joint_y_'+str(i), (17.8, .012, .002), (0, y, .001), 'seam')

    # Campus facade stays at the perimeter. The former wall only 2m behind
    # the person made back-view photography physically unreachable. Geometry
    # is still visual+collision paired and is NEVER exported as a task prior.
    wall = model(world, 'campus_backdrop')
    box(wall, 'wall', (.20, 11.8, 3.15), (8.7, 0, 1.575), 'cream')
    box(wall, 'base', (.24, 11.8, .28), (8.67, 0, .14), 'stone')
    box(wall, 'cap', (.28, 11.9, .10), (8.67, 0, 3.19), 'white')
    for i, y in enumerate((-4.8,-1.6,1.6,4.8)):
        box(wall, 'side_panel_'+str(i), (.024, 1.2, 2.4), (8.588, y, 1.48), 'teal')
        for j in range(3):
            box(wall, 'wood_detail_%d_%d'%(i,j), (.035, .035, 2.2),
                (8.566, y-.2+j*.2, 1.48), 'wood_light')
    # Side walls permit local horizontal free-space evidence without using a
    # global map. The front-to-subject flight corridor remains unobstructed.
    sides = model(world, 'courtyard_side_walls')
    box(sides, 'west', (.14, 11.8, 2.8), (-8.8, 0, 1.4), 'cream')
    box(sides, 'north', (17.6, .14, 2.8), (0, 5.9, 1.4), 'cream')
    box(sides, 'south_low', (17.6, .14, .56), (0, -5.9, .28), 'stone')
    beds = model(world, 'garden_beds')
    box(beds, 'north_grass', (15.6, 1.1, .06), (0, 5.1, .03), 'grass')
    box(beds, 'south_grass', (15.6, .7, .06), (0, -5.4, .03), 'grass')
    box(beds, 'north_curb', (15.7, .10, .13), (0, 4.5, .065), 'stone')
    box(beds, 'south_curb', (15.7, .10, .13), (0, -5., .065), 'stone')
    tree(world, 'north_tree', -6.9, 5.1, 2.7)
    tree(world, 'south_tree', 6.8, -5.3, 2.6)
    bench(world, 'campus_bench', -4.7, 4.9)
    bench(world, 'campus_bench_east', 4.6, 4.9)

    # Two real-sized parked vehicles, kept north of the person's orbit and the
    # south-tree approach. Authored poses belong ONLY to the simulator world.
    # The planner still sees finite-range physical LiDAR; the VLM sees RGB.
    for color, x, yaw in (('white', -3.7, 3.141592653589793), ('red', 3.7, 0.0)):
        bay = model(world, 'parking_bay_'+color, (x, 3.15, 0, 0, 0, 0))
        for side, y in (('left', -1.24), ('right', 1.24)):
            box(bay, side, (4.8, .045, .003), (0, y, .003), 'white')
        box(bay, 'back', (.045, 2.48, .003), (2.4 if color=='white' else -2.4, 0, .003), 'white')
        car = add(world, 'include')
        add(car, 'uri', 'model://photo_car_'+color)
        add(car, 'name', 'courtyard_car_'+color)
        add(car, 'pose', numbers((x, 3.15, .005, 0, 0, yaw)))
        add(car, 'static', 'true')

    # One real human-shaped mesh, no semantic labels/coordinates exported to
    # the task layer. Source mesh's -Y front rotates to world -X.
    subject = add(world, 'include')
    add(subject, 'uri', 'model://photo_person_standing')
    add(subject, 'name', 'photo_subject')
    add(subject, 'pose', '-1.8 -3 0 0 0 -1.570796326794897')
    add(subject, 'static', 'true')
    return sdf


def main():
    destination = ROOT/'worlds'/'vla_photo_courtyard.world'
    tree = build()
    xml = minidom.parseString(ET.tostring(tree, encoding='utf-8'))
    destination.write_text(xml.toprettyxml(indent='  '), encoding='utf-8')
    print('Generated', destination)
    print('%d paired primitive visuals/collisions plus one person and two parked car meshes' % len(tree.findall('.//visual')))


if __name__ == '__main__':
    main()
