#!/usr/bin/env python3
"""Build a same-person, textured 3D hands-on-hips asset for scene review.

This is simulator asset authoring, never a perception/pose classifier. The
original COLLADA has an unbound, differently posed skeleton; no skin weights.
Its disconnected bare-arm components and connected shirt sleeve boundaries
provide explicit authoring topology. No runtime recognizer reads this file.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'models/photo_person_standing'
OUTPUT = ROOT / 'models/photo_person_hands_on_hips'
NS = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
ET.register_namespace('', NS['c'])


def source_array(mesh, suffix):
    source = next(s for s in mesh.findall('c:source', NS)
                  if s.attrib['id'].endswith(suffix))
    return source, np.fromstring(source.find('c:float_array', NS).text,
                                sep=' ').reshape(-1, 3)


def polygons(mesh):
    out = []
    for poly in mesh.findall('c:polylist', NS):
        inputs = poly.findall('c:input', NS)
        stride = max(int(x.attrib['offset']) for x in inputs) + 1
        offsets = {x.attrib['semantic']: int(x.attrib['offset']) for x in inputs}
        corners = np.fromstring(poly.find('c:p', NS).text, sep=' ', dtype=int).reshape(-1, stride)
        counts = np.fromstring(poly.find('c:vcount', NS).text, sep=' ', dtype=int)
        at = 0
        for count in counts:
            face = corners[at:at + count]
            for k in range(1, count - 1):
                out.append((face[[0, k, k + 1]], offsets))
            at += count
    return out


def topology(n, triangles):
    adjacent = [set() for _ in range(n)]
    edges = Counter()
    for corners, offsets in triangles:
        v = corners[:, offsets['VERTEX']]
        for a, b in zip(v, np.roll(v, -1)):
            adjacent[a].add(int(b)); adjacent[b].add(int(a))
            edges[tuple(sorted((int(a), int(b))))] += 1
    unseen = set(range(n)); components = []
    while unseen:
        seed = unseen.pop(); component = [seed]; stack = [seed]
        while stack:
            for x in adjacent[stack.pop()]:
                if x in unseen:
                    unseen.remove(x); stack.append(x); component.append(x)
        components.append(np.array(sorted(component), dtype=int))
    boundary = defaultdict(set)
    for (a, b), count in edges.items():
        if count == 1: boundary[a].add(b); boundary[b].add(a)
    unseen = set(boundary); loops = []
    while unseen:
        seed = unseen.pop(); loop = [seed]; stack = [seed]
        while stack:
            for x in boundary[stack.pop()]:
                if x in unseen:
                    unseen.remove(x); stack.append(x); loop.append(x)
        loops.append(np.array(sorted(loop), dtype=int))
    return adjacent, components, loops


def rotation_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    cross = np.cross(a, b); c = np.dot(a, b)
    k = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def rig(side):
    # These fitted joints are asset-authoring coordinates, never target input.
    if side > 0:
        shoulder = np.array([.202, .005, 1.515])
        elbow = np.array([.224, -.0155, 1.240])
        wrist = np.array([.233, -.053, .966])
    else:
        shoulder = np.array([-.210, -.030, 1.515])
        elbow = np.array([-.2347, -.0557, 1.240])
        wrist = np.array([-.236, -.077, .965])
    elbow_direction = np.array([side * .203, .012, -.185])
    target_elbow = shoulder + elbow_direction / np.linalg.norm(elbow_direction) * np.linalg.norm(elbow - shoulder)
    wrist_direction = np.array([-side * .200, -.100, -.155])
    target_wrist = target_elbow + wrist_direction / np.linalg.norm(wrist_direction) * np.linalg.norm(wrist - elbow)
    upper = rotation_between(elbow - shoulder, target_elbow - shoulder)
    forearm = rotation_between(wrist - elbow, target_wrist - target_elbow)
    # Palms/fingers retain their original downward orientation at the hip.
    return shoulder, elbow, wrist, target_elbow, target_wrist, upper, forearm


def transform_arm(points, side):
    s, e, w, te, tw, ru, rf = rig(side)
    upper = (points - s) @ ru.T + s
    lower = (points - e) @ rf.T + te
    hand = points - w + tw
    wu = smoothstep((points[:, 2] - 1.18) / .11)
    wh = 1 - smoothstep((points[:, 2] - .94) / .06)
    wf = 1 - wu - wh
    return upper * wu[:, None] + lower * wf[:, None] + hand * wh[:, None]


def shirt_weights(points, adjacent, loops, side):
    sleeve = next(loop for loop in loops if side * points[loop, 0].mean() > .18)
    # Keep the torso/neck/hem fixed. Harmonic blending over existing topology
    # joins the cuff to the shoulder; there are no detached or invented parts.
    allowed = (side * points[:, 0] > .12) & (points[:, 2] > 1.32)
    boundary = np.zeros(len(points), bool); boundary[sleeve] = True
    weights = np.zeros(len(points)); weights[boundary] = 1
    free = np.flatnonzero(allowed & ~boundary)
    for _ in range(400):
        new = weights.copy()
        for i in free:
            ids = list(adjacent[i])
            lengths = np.linalg.norm(points[ids] - points[i], axis=1)
            inv = 1 / np.maximum(lengths, 1e-6)
            new[i] = np.dot(weights[ids], inv) / inv.sum()
        if np.max(np.abs(new - weights)) < 1e-9:
            weights = new; break
        weights = new
    return weights


def update_array(source, values):
    array = source.find('c:float_array', NS)
    array.text = ' '.join(format(float(x), '.9g') for x in values.ravel())


def deform(mesh, geometry_id):
    source, original = source_array(mesh, '-positions')
    triangles = polygons(mesh)
    adjacent, components, loops = topology(len(original), triangles)
    posed = original.copy()
    if geometry_id == 'v2_Body-mesh':
        arms = [ids for ids in components if len(ids) == 1989 and .8 < original[ids, 2].mean() < 1.4]
        assert len(arms) == 2, 'Unexpected original arm topology'
        for ids in arms:
            side = 1 if original[ids, 0].mean() > 0 else -1
            posed[ids] = transform_arm(original[ids], side)
    elif geometry_id == 'v2_tshirt02-mesh':
        for side in (-1, 1):
            weights = shirt_weights(original, adjacent, loops, side)
            s, _, _, _, _, rotation, _ = rig(side)
            moved = (original - s) @ rotation.T + s
            posed += weights[:, None] * (moved - original)
    else:
        return {'changed_vertices': 0, 'vertices': len(original)}
    moved = np.linalg.norm(posed - original, axis=1) > 1e-10
    update_array(source, posed)
    # Recompute only normals touched by deformed triangles. UV indices,
    # material bindings, face indices and all texture pixels remain unchanged.
    normal_source, normals = source_array(mesh, '-normals')
    accumulated = np.zeros_like(normals); touched = np.zeros(len(normals), bool)
    ratio = []; minimum_area = 1e9
    for corners, offsets in triangles:
        ids = corners[:, offsets['VERTEX']]; ni = corners[:, offsets['NORMAL']]
        old = original[ids]; new = posed[ids]
        normal = np.cross(new[1] - new[0], new[2] - new[0])
        old_area = np.linalg.norm(np.cross(old[1] - old[0], old[2] - old[0]))
        new_area = np.linalg.norm(normal)
        if old_area > 1e-10: ratio.append(new_area / old_area)
        minimum_area = min(minimum_area, new_area)
        for index in ni: accumulated[index] += normal
        if moved[ids].any(): touched[ni] = True
    lengths = np.linalg.norm(accumulated, axis=1)
    safe = touched & (lengths > 1e-12)
    normals[safe] = accumulated[safe] / lengths[safe, None]
    update_array(normal_source, normals)
    assert np.isfinite(posed).all() and np.isfinite(normals).all()
    return {'changed_vertices': int(moved.sum()), 'vertices': len(original),
            'area_ratio_min': min(ratio), 'area_ratio_max': max(ratio),
            'minimum_double_triangle_area': minimum_area,
            'source_bounds': [original.min(0).tolist(), original.max(0).tolist()],
            'posed_bounds': [posed.min(0).tolist(), posed.max(0).tolist()]}


def main():
    original = SOURCE / 'meshes/standing.dae'
    tree = ET.parse(original); document = tree.getroot()
    report = {}
    for geometry in document.findall('.//c:geometry', NS):
        report[geometry.attrib['id']] = deform(geometry.find('c:mesh', NS), geometry.attrib['id'])
    for ref in document.findall('.//c:image/c:init_from', NS):
        ref.text = '../../photo_person_standing/materials/textures/' + Path(ref.text).name
    # Drop the misleading unbound legacy skeleton in this baked static variant.
    for visual_scene in document.findall('.//c:visual_scene', NS):
        for node in list(visual_scene):
            if node.attrib.get('id') == 'v2': visual_scene.remove(node)
    (OUTPUT / 'meshes').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE / 'LICENSE', OUTPUT / 'LICENSE')
    output = OUTPUT / 'meshes/hands_on_hips.dae'
    tree.write(output, encoding='utf-8', xml_declaration=True)
    model = ET.parse(SOURCE / 'model.sdf')
    model.getroot().find('model').set('name', 'photo_person_hands_on_hips')
    for uri in model.findall('.//mesh/uri'):
        uri.text = 'model://photo_person_hands_on_hips/meshes/hands_on_hips.dae'
    model.write(OUTPUT / 'model.sdf', encoding='utf-8', xml_declaration=True)
    (OUTPUT / 'model.config').write_text('''<?xml version="1.0"?>
<model><name>Same person with hands on hips</name><version>1.0</version>
<sdf version="1.6">model.sdf</sdf><author><name>Marina Kollmitz; local pose derivative</name></author>
<description>Same textured person, procedurally posed arms and sleeves; CC BY 3.0. Simulator-only asset.</description></model>
''')
    manifest = {'original_mesh_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                'derived_mesh_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'generator': Path(__file__).name, 'geometry': report,
                'bone_length_preservation': True, 'uv_material_and_face_indices_unchanged': True,
                'source_textures_unchanged_and_shared': True,
                'runtime_perception_must_not_read_this_asset_metadata': True}
    (OUTPUT / 'DERIVATION.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
