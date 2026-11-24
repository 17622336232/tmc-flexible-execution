#!/usr/bin/env python3
"""Create two parked, low-poly car assets from attributed Gazebo source files.

Offline authoring only. The shipped models require no download at launch, and
their geometry/placement is never exported as a perception or model-task prior.
Textures stay unchanged; paint colors are separate Wavefront materials.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom


ROOT = Path(__file__).resolve().parents[1]
REVISION = '8163eb4b5e7e21985c6591d1c0bfb56468c0093f'
SOURCE = 'https://github.com/osrf/gazebo_models/tree/' + REVISION + '/hatchback'
FILES = ('hatchback.obj', 'hatchback.mtl', 'hatchback.png', 'wheels3.png', 'LICENSE')


def element(parent, tag, value=None, **attributes):
    node = ET.SubElement(parent, tag, attributes)
    if value is not None:
        node.text = str(value)
    return node


def xml(path, root):
    path.write_text(minidom.parseString(ET.tostring(root)).toprettyxml(indent='  '))


def build(source):
    inputs = {name: (source/name).read_bytes() for name in FILES}
    original = inputs['hatchback.obj'].decode().splitlines()
    points = [tuple(map(float, line.split()[1:4])) for line in original if line.startswith('v ')]
    low = [min(p[axis] for p in points) for axis in range(3)]
    high = [max(p[axis] for p in points) for axis in range(3)]
    center = [(low[axis]+high[axis])/2 for axis in range(2)]
    scale = .0254
    # Upstream inch coordinates, front toward -Y. Local model uses metres,
    # a centered footprint, front +X, with the lowest tire vertex on the floor.
    normalized = []
    for line in original:
        if line.startswith('v '):
            x,y,z = map(float, line.split()[1:4])
            normalized.append('v %.9g %.9g %.9g' % (-(y-center[1])*scale,
                                                    (x-center[0])*scale, (z-low[2])*scale))
        elif line.startswith('vn '):
            x,y,z = map(float, line.split()[1:4])
            normalized.append('vn %.9g %.9g %.9g' % (-y,x,z))
        else:
            normalized.append(line)
    dimensions = [(high[1]-low[1])*scale, (high[0]-low[0])*scale, (high[2]-low[2])*scale]
    for color, ambient, diffuse in (('red', '.50 .028 .020', '.78 .045 .032'),
                                    ('white', '.5882 .5882 .5882', '.75 .75 .75')):
        name = 'photo_car_' + color
        output = ROOT/'models'/name
        (output/'meshes').mkdir(parents=True, exist_ok=True)
        (output/'materials/textures').mkdir(parents=True, exist_ok=True)
        names = {'Hatchback': 'VLA_'+color+'_Paint', 'Wheels3': 'VLA_'+color+'_Wheels'}
        geometry = '\n'.join('usemtl '+names[line.split()[1]] if line.startswith('usemtl ') else line
                             for line in normalized) + '\n'
        (output/'meshes/hatchback.obj').write_text(geometry)
        paint, material = False, []
        for line in inputs['hatchback.mtl'].decode().splitlines():
            fields = line.split()
            if not fields:
                material.append(''); continue
            if fields[0] == 'newmtl':
                paint = fields[1] == 'Hatchback'
                line = 'newmtl ' + names[fields[1]]
            elif fields[0] in ('Ka','Kd') and paint:
                line = fields[0]+' '+(ambient if fields[0]=='Ka' else diffuse)
            elif fields[0] in ('map_Ka','map_Kd'):
                line = fields[0]+' ../materials/textures/'+('hatchback.png' if paint else 'wheels3.png')
            material.append(line)
        (output/'meshes/hatchback.mtl').write_text('\n'.join(material)+'\n')
        for texture in ('hatchback.png','wheels3.png'):
            shutil.copyfile(source/texture, output/'materials/textures'/texture)
        shutil.copyfile(source/'LICENSE', output/'LICENSE')
        sdf = ET.Element('sdf', version='1.6')
        model = element(sdf, 'model', name=name)
        element(model, 'static', 'true')
        link = element(model, 'link', name='body')
        for role in ('collision','visual'):
            node = element(link, role, name='car_'+role)
            mesh = element(element(node, 'geometry'), 'mesh')
            element(mesh, 'uri', 'model://'+name+'/meshes/hatchback.obj')
            element(mesh, 'scale', '1 1 1')
            if role == 'visual':
                element(node, 'cast_shadows', 'false')
        xml(output/'model.sdf', sdf)
        config = ET.Element('model')
        element(config, 'name', 'Photo courtyard car / '+color)
        element(config, 'version', '1.0')
        element(config, 'sdf', 'model.sdf', version='1.6')
        author = element(config, 'author'); element(author, 'name', 'Nate Koenig; Diff-VLA local adaptation')
        element(config, 'description', 'Static '+color+' hatchback with matching mesh collision. CC BY 3.0.')
        xml(output/'model.config', config)
        (output/'ATTRIBUTION.md').write_text(
            '# Courtyard car\n\nSource: [Open Robotics Gazebo hatchback]('+SOURCE+').\n\n'
            'Author: Nate Koenig. Copyright 2012 Nathan Koenig. '
            'License: Creative Commons Attribution 3.0 Unported; see LICENSE.\n\n'
            'Local changes: normalize inches to metres, center the footprint, rotate front to +X, '
            'put tire bottoms at local Z=0, assign unique '+color+' paint/wheel material names, '
            'and wrap as a static parked model. Original PNG textures are unchanged. '
            'Visual and collision use the same 788-face mesh. No dynamics/control plugin.\n')
        manifest = dict(upstream=SOURCE, revision=REVISION,
                        source_sha256={key: hashlib.sha256(value).hexdigest() for key,value in inputs.items()},
                        dimensions_m=dict(zip(('length','width_including_mirrors','height'), dimensions)),
                        faces=sum(line.startswith('f ') for line in original),
                        geometry_transform='x=-(source_y-center_y)*0.0254; y=(source_x-center_x)*0.0254; z=(source_z-min_z)*0.0254',
                        texture_pixels_modified=False, paint=color,
                        files_sha256={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in output.rglob('*') if p.is_file() and p.name!='SOURCE_MANIFEST.json'})
        (output/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(name, 'dimensions_m=', dimensions, 'faces=', manifest['faces'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Local upstream OBJ/MTL/PNG/LICENSE directory')
    build(parser.parse_args().source)
