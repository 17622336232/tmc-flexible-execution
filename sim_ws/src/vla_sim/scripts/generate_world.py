#!/usr/bin/env python3
"""Build an original, self-contained Gazebo Classic demonstration gallery.

No model downloads, meshes, textures, ground-truth point clouds, or invisible
obstacles are used. Every visual primitive has the same collision primitive.
Layout references (concepts only; no copied code/assets):
https://get.gazebosim.org/tutorials?cat=build_world&tut=building_editor
https://github.com/aws-robotics/aws-robomaker-small-warehouse-world
"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "floor": "0.63 0.68 0.72 1", "seam": "0.46 0.51 0.55 1",
    "white": "0.90 0.94 0.96 1", "slate": "0.12 0.19 0.25 1",
    "steel": "0.21 0.28 0.34 1", "cyan": "0.10 0.72 0.84 1",
    "blue": "0.12 0.37 0.54 1", "gold": "0.96 0.64 0.25 1",
    "wood": "0.58 0.37 0.20 1", "lightwood": "0.76 0.57 0.34 1",
    "glass": "0.43 0.73 0.83 1", "leaf": "0.13 0.40 0.26 1",
    "leaf_light": "0.23 0.52 0.31 1", "soil": "0.16 0.12 0.09 1",
    "screen": "0.035 0.09 0.13 1", "red": "0.85 0.27 0.18 1",
    "orange": "0.98 0.35 0.055 1", "panel": "0.73 0.80 0.84 1",
}


def add(parent, tag, value=None, **attrs):
    element = ET.SubElement(parent, tag, attrs)
    if value is not None:
        element.text = str(value)
    return element


def nums(values):
    return " ".join(f"{v:.6g}" for v in values)


def model(world, name, pose=(0, 0, 0, 0, 0, 0)):
    result = add(world, "model", name=name)
    add(result, "static", "true")
    add(result, "pose", nums(pose))
    return add(result, "link", name="body")


def part(link, name, shape, dimensions, xyz, color, rpy=(0, 0, 0), glow=0,
         transparency=0):
    """Pair one visible primitive with one exactly matching collision."""
    for kind in ("collision", "visual"):
        node = add(link, kind, name=f"{name}_{kind}")
        add(node, "pose", nums((*xyz, *rpy)))
        geometry = add(add(node, "geometry"), shape)
        if shape == "box":
            add(geometry, "size", nums(dimensions))
        elif shape == "cylinder":
            add(geometry, "radius", dimensions[0])
            add(geometry, "length", dimensions[1])
        elif shape == "sphere":
            add(geometry, "radius", dimensions[0])
        if kind == "visual":
            material = add(node, "material")
            add(material, "ambient", COLORS[color])
            add(material, "diffuse", COLORS[color])
            add(material, "specular", "0.24 0.27 0.30 1")
            if glow:
                rgba = [float(v) for v in COLORS[color].split()]
                add(material, "emissive", nums([c * glow for c in rgba[:3]] + [1]))
            if transparency:
                add(node, "transparency", transparency)


def box(link, name, size, pos, color, **kwargs):
    part(link, name, "box", size, pos, color, **kwargs)


def cyl(link, name, radius, height, pos, color, **kwargs):
    part(link, name, "cylinder", (radius, height), pos, color, **kwargs)


def plant(world, name, x, y, scale=1):
    link = model(world, name, (x, y, 0, 0, 0, 0))
    s = scale
    cyl(link, "pot_foot", .29*s, .07*s, (0, 0, .035*s), "slate")
    cyl(link, "ceramic_pot", .34*s, .55*s, (0, 0, .31*s), "white")
    cyl(link, "pot_rim", .36*s, .06*s, (0, 0, .58*s), "slate")
    cyl(link, "soil", .31*s, .02*s, (0, 0, .61*s), "soil")
    cyl(link, "stem", .055*s, .75*s, (0, 0, .99*s), "wood")
    for i, (dx, dy, dz, radius) in enumerate([
            (0, 0, 1.55, .39), (.23, .03, 1.24, .31),
            (-.19, .16, 1.29, .29), (.02, -.23, 1.28, .30)]):
        part(link, f"leaves_{i}", "sphere", (radius*s,),
             (dx*s, dy*s, dz*s), "leaf" if i % 2 else "leaf_light")


def pad(world, name, x, y, color, landing=False):
    link = model(world, name, (x, y, 0, 0, 0, 0))
    cyl(link, "outer_disc", .91, .006, (0, 0, .003), color)
    cyl(link, "inner_disc", .82, .006, (0, 0, .009), "slate")
    if landing:
        box(link, "h_left", (.14, 1.02, .003), (-.29, 0, .014), "white")
        box(link, "h_right", (.14, 1.02, .003), (.29, 0, .014), "white")
        box(link, "h_bar", (.58, .14, .003), (0, 0, .014), "white")
    else:
        for i, (px, py, sx, sy) in enumerate([
                (-.5, 0, .21, .06), (.5, 0, .21, .06),
                (0, -.5, .06, .21), (0, .5, .06, .21)]):
            box(link, f"target_{i}", (sx, sy, .003), (px, py, .014), "white")
        cyl(link, "target_dot", .09, .003, (0, 0, .014), color)


def workstation(world, name, x, y):
    link = model(world, name, (x, y, 0, 0, 0, 0))
    box(link, "desktop", (1.25, 2.25, .10), (0, 0, .90), "lightwood")
    for i, (px, py) in enumerate([(-.48, -.92), (.48, -.92), (-.48, .92), (.48, .92)]):
        box(link, f"leg_{i}", (.07, .07, .85), (px, py, .425), "steel")
    box(link, "cabinet", (.85, .55, .77), (.05, .72, .385), "white")
    for i in range(3):
        box(link, f"drawer_{i}", (.04, .43, .008), (-.39, .72, .21 + i*.22), "steel")
    box(link, "monitor_foot", (.30, .43, .03), (.18, -.25, .97), "steel")
    box(link, "monitor_stand", (.045, .06, .32), (.26, -.25, 1.13), "steel")
    box(link, "monitor", (.07, .83, .51), (.30, -.25, 1.42), "slate")
    box(link, "display", (.008, .73, .41), (.26, -.25, 1.42), "screen")
    for i, width in enumerate([.4, .26, .48, .19]):
        box(link, f"screen_line_{i}", (.009, width, .018),
            (.253, -.25, 1.54 - i*.07), "cyan", glow=.45)
    box(link, "keyboard", (.28, .55, .025), (-.20, -.25, .968), "slate")
    box(link, "device", (.36, .37, .22), (.12, .63, 1.06), "blue")
    cyl(link, "status_led", .025, .008, (-.065, .63, 1.1), "cyan",
        rpy=(0, math.pi/2, 0), glow=.8)


def build():
    with (ROOT / "config" / "landmarks.json").open(encoding="utf-8") as source:
        landmarks = json.load(source)["landmarks"]
    sdf = ET.Element("sdf", version="1.6")
    world = add(sdf, "world", name="vla_gallery")
    add(world, "gravity", "0 0 -9.80665")
    physics = add(world, "physics", name="gallery_ode", type="ode")
    add(physics, "max_step_size", .004)
    add(physics, "real_time_factor", 1)
    add(physics, "real_time_update_rate", 250)
    solver = add(add(physics, "ode"), "solver")
    add(solver, "type", "quick")
    add(solver, "iters", 30)
    scene = add(world, "scene")
    add(scene, "ambient", "0.38 0.42 0.48 1")
    add(scene, "background", "0.065 0.10 0.15 1")
    add(scene, "shadows", "true")
    add(scene, "grid", "false")
    camera = add(add(world, "gui", fullscreen="0"), "camera", name="gallery_overview")
    add(camera, "pose", "14 -18 15 0 0.58 2.23")
    add(camera, "view_controller", "orbit")
    add(camera, "projection_type", "perspective")
    sun = add(world, "light", name="soft_daylight", type="directional")
    add(sun, "pose", "-6 -3 12 0 0 0")
    add(sun, "diffuse", "0.58 0.65 0.73 1")
    add(sun, "specular", "0.18 0.22 0.28 1")
    add(sun, "direction", "0.3 0.25 -0.92")
    add(sun, "cast_shadows", "true")
    for name, x, y in [("warm_fill", -4, 1), ("cool_fill", 5, -1)]:
        light = add(world, "light", name=name, type="point")
        add(light, "pose", nums((x, y, 4.8, 0, 0, 0)))
        add(light, "diffuse", "0.30 0.26 0.21 1" if x < 0 else "0.22 0.28 0.34 1")
        add(light, "cast_shadows", "false")
        attenuation = add(light, "attenuation")
        for key, value in [("range", 22), ("constant", 1), ("linear", .05), ("quadratic", .014)]:
            add(attenuation, key, value)

    shell = model(world, "gallery_architecture")
    box(shell, "foundation", (17.4, 11.4, .22), (0, 0, -.16), "slate")
    box(shell, "floor", (17, 11, .10), (0, 0, -.05), "floor")
    for x in range(-8, 9, 2):
        box(shell, f"tile_x_{x}", (.012, 10.8, .002), (x, 0, .001), "seam")
    for y in range(-4, 5, 2):
        box(shell, f"tile_y_{y}", (16.8, .012, .002), (0, y, .001), "seam")
    box(shell, "north_wood_walk", (16.8, 1.05, .006), (0, 4.92, .003), "lightwood")
    for x in range(-8, 9):
        box(shell, f"wood_joint_{x}", (.018, 1.05, .002), (x, 4.92, .007), "wood")
    box(shell, "north_base", (17.1, .18, 1.04), (0, 5.5, .52), "slate")
    box(shell, "north_sill", (17.2, .32, .09), (0, 5.48, 1.085), "lightwood")
    box(shell, "north_header", (17.2, .23, .20), (0, 5.5, 3.53), "steel")
    for i, x in enumerate([-7.2, -4.8, -2.4, 0, 2.4, 4.8, 7.2]):
        box(shell, f"window_{i}", (2.24, .045, 2.27), (x, 5.5, 2.27), "glass", transparency=.58)
    for i, x in enumerate([-8.4, -6, -3.6, -1.2, 1.2, 3.6, 6, 8.4]):
        box(shell, f"mullion_{i}", (.10, .15, 2.55), (x, 5.46, 2.3), "steel")
    box(shell, "window_crossbar", (17, .13, .065), (0, 5.45, 2.55), "steel")
    box(shell, "west_wall", (.18, 11.1, 3.6), (-8.5, 0, 1.8), "white")
    box(shell, "west_inset", (.035, 6.8, 2.65), (-8.39, .8, 1.67), "slate")
    for i in range(25):
        box(shell, f"wood_slat_{i}", (.08, .055, 2.65), (-8.33, -2.45 + i*.27, 1.67), "lightwood")
    box(shell, "east_wall", (.18, 11.1, 3.6), (8.5, 0, 1.8), "white")
    box(shell, "east_base", (.21, 10.8, .45), (8.46, 0, .225), "slate")
    box(shell, "south_cutaway_wall", (17.1, .18, .48), (0, -5.5, .24), "slate")
    box(shell, "south_toprail", (17.15, .24, .055), (0, -5.5, .51), "lightwood")
    for i, (sx, sy, px, py) in enumerate([(16.7, .035, 0, 5.36), (16.7, .035, 0, -5.36),
                                             (.035, 10.7, -8.36, 0), (.035, 10.7, 8.36, 0)]):
        box(shell, f"floor_border_{i}", (sx, sy, .005), (px, py, .005), "cyan", glow=.3)

    # West wall branding is made of solid strokes, avoiding external fonts.
    brand = model(world, "vla_wall_identity", (-8.26, .2, 2.45, 0, 0, 0))
    box(brand, "backplate", (.04, 3.5, .83), (0, 0, 0), "slate")
    for i, (y, z, length, angle) in enumerate([
            (-1.20, .02, .62, .42), (-.95, .02, .62, -.42),
            (-.29, .02, .62, 0), (-.12, -.27, .38, math.pi/2),
            (.63, .02, .62, -.40), (.87, .02, .62, .40),
            (.75, -.08, .29, math.pi/2)]):
        box(brand, f"letter_stroke_{i}", (.03, .065, length), (.03, y, z),
            "cyan" if i < 2 else "white", rpy=(angle, 0, 0), glow=.25)

    # Pendant lights sit above the configured flight ceiling (2.5 m).
    for i, x in enumerate([-5.4, 0, 5.4]):
        link = model(world, f"pendant_{i}")
        box(link, "beam", (2.5, .15, .11), (x, 2.9, 3.75), "steel")
        box(link, "diffuser", (2.34, .16, .027), (x, 2.9, 3.68), "white", glow=.75)
        for j, dx in enumerate([-.95, .95]):
            cyl(link, f"hanger_{j}", .012, .55, (x+dx, 2.9, 4.07), "steel")

    for key, color in [("home", "cyan"), ("exhibit", "gold"),
                       ("window", "cyan"), ("workbench", "blue")]:
        point = landmarks[key]
        pad(world, f"{key}_target", point["x"], point["y"], color,
            landing=(key == "home"))

    # Two finite obstacles lie near nominal routes. The planner chooses detours.
    island = model(world, "exhibit_island_west", (-2.7, -1.35, 0, 0, 0, .15))
    box(island, "plinth", (1.10, .85, .12), (0, 0, .06), "slate")
    box(island, "body", (.91, .69, 1.22), (0, 0, .73), "white")
    box(island, "top", (1.10, .85, .10), (0, 0, 1.39), "lightwood")
    box(island, "accent", (.018, .55, .045), (-.465, 0, 1.17), "cyan", glow=.35)
    box(island, "sculpture", (.43, .43, .43), (0, 0, 1.70), "cyan", rpy=(.2, .25, .45))
    island = model(world, "exhibit_island_east", (2.5, 1.15, 0, 0, 0, 0))
    cyl(island, "foot", .58, .10, (0, 0, .05), "slate")
    cyl(island, "pedestal", .45, 1.23, (0, 0, .715), "blue")
    cyl(island, "top", .53, .08, (0, 0, 1.37), "lightwood")
    part(island, "orb", "sphere", (.28,), (0, 0, 1.69), "gold")

    # Observable inspection station: a physical test fixture, never a semantic
    # landmark or a source of planner/VLM coordinates. The front face and the
    # separated opaque partition both provide real LiDAR returns at hover.
    equipment = model(world, "orange_inspection_equipment", (-3.4, -3.0, 0, 0, 0, 0))
    box(equipment, "floor_foot", (.70, 1.05, .08), (0, 0, .04), "slate")
    for i, y in enumerate([-.31, .31]):
        box(equipment, f"support_{i}", (.13, .11, .84), (.08, y, .50), "steel")
    box(equipment, "support_shelf", (.62, 1.02, .10), (0, 0, .95), "steel")
    box(equipment, "cabinet", (.50, .90, .80), (0, 0, 1.40), "orange")
    box(equipment, "front_door", (.016, .80, .68), (-.258, 0, 1.40), "orange")
    for i, y in enumerate([-.43, .43]):
        box(equipment, f"edge_guard_{i}", (.035, .035, .80), (-.24, y, 1.40), "slate")
    box(equipment, "top_edge", (.50, .92, .035), (0, 0, 1.795), "steel")
    box(equipment, "status_screen", (.012, .23, .14), (-.273, .18, 1.56), "screen")
    box(equipment, "status_led", (.014, .055, .028), (-.281, .23, 1.56), "cyan", glow=.55)
    box(equipment, "door_handle", (.045, .045, .20), (-.285, -.27, 1.40), "steel")
    for i, width in enumerate([.34, .22]):
        box(equipment, f"inspection_label_{i}", (.013, width, .025),
            (-.274, .06, 1.23-i*.055), "white")

    partition = model(world, "inspection_station_partition", (-1.5, -3.0, 0, 0, 0, 0))
    box(partition, "opaque_panel", (.12, 3.0, 2.50), (0, 0, 1.25), "panel")
    box(partition, "front_inset", (.018, 2.80, 2.22), (-.069, 0, 1.31), "white")
    box(partition, "header", (.17, 3.12, .11), (0, 0, 2.535), "steel")
    for i, y in enumerate([-1.53, 1.53]):
        box(partition, f"frame_{i}", (.17, .065, 2.59), (0, y, 1.295), "steel")
        box(partition, f"foot_{i}", (.70, .16, .075), (0, y, .038), "slate")
    box(partition, "cyan_header_strip", (.020, 2.74, .035), (-.086, 0, 2.35), "cyan", glow=.2)

    # Peripheral furnishing leaves the four destination footprints unobstructed.
    workstation(world, "robotics_workbench", 7.0, -3.35)
    workstation(world, "engineering_station", 7.0, -.15)
    lounge = model(world, "north_lounge")
    box(lounge, "sofa_base", (3.0, .88, .30), (-3.9, 4.2, .24), "slate")
    box(lounge, "sofa_back", (3.0, .18, .76), (-3.9, 4.63, .62), "blue")
    for i in range(3):
        box(lounge, f"seat_{i}", (.91, .72, .18), (-4.89+i*.99, 4.16, .46), "blue")
    for i, x in enumerate([-5.41, -2.39]):
        box(lounge, f"arm_{i}", (.17, .90, .40), (x, 4.2, .53), "slate")
    cyl(lounge, "table_top", .49, .065, (-3.9, 2.9, .55), "lightwood")
    cyl(lounge, "table_stem", .055, .49, (-3.9, 2.9, .27), "steel")
    cyl(lounge, "table_foot", .28, .025, (-3.9, 2.9, .015), "steel")

    shelf = model(world, "instrument_shelves", (-7.55, 2.3, 0, 0, 0, 0))
    for i, y in enumerate([-1.2, 1.2]):
        box(shelf, f"upright_{i}", (.07, .07, 2.20), (0, y, 1.10), "steel")
    for i, z in enumerate([.12, .68, 1.24, 1.80, 2.23]):
        box(shelf, f"shelf_{i}", (.65, 2.55, .055), (0, 0, z), "lightwood")
    for i, (y, z, col) in enumerate([(-.8, .94, "white"), (.6, .94, "blue"),
                                    (-.65, 1.47, "gold"), (.60, 1.49, "white"),
                                    (-.7, .35, "slate"), (.55, .36, "slate")]):
        box(shelf, f"equipment_case_{i}", (.42, .52, .39), (0, y, z), col)
        box(shelf, f"case_handle_{i}", (.025, .18, .04), (.23, y, z), "steel")

    for i, (x, y, scale) in enumerate([(-7.4, -4.4, .83), (-7.45, 4.7, 1.0),
                                       (.2, 4.65, 1.05), (7.3, 4.6, 1.05),
                                       (2.0, -4.6, .82)]):
        plant(world, f"plant_{i}", x, y, scale)

    # Recessed display panels and geometric artwork on the east wall.
    wall = model(world, "east_wall_displays")
    for i, y in enumerate([-2.8, .1, 3.3]):
        box(wall, f"display_frame_{i}", (.055, 2.0, 1.18), (8.37, y, 2.15), "slate")
        box(wall, f"display_canvas_{i}", (.018, 1.82, 1.00), (8.33, y, 2.15), "blue")
        for j in range(4):
            box(wall, f"diagram_{i}_{j}", (.02, .25 + j*.14, .035),
                (8.314, y-.1, 2.43-j*.18), "cyan" if j % 2 else "white", glow=.2)
    return sdf


def main():
    destination = ROOT / "worlds" / "vla_gallery.world"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(build())
    # Python 3.8 (Ubuntu 20.04) has no ET.indent().
    from xml.dom import minidom
    xml = minidom.parseString(ET.tostring(tree.getroot(), encoding="utf-8"))
    destination.write_text(xml.toprettyxml(indent="  "), encoding="utf-8")
    print(f"Generated {destination}")
    print(f"{len(tree.findall('.//visual'))} visuals; "
          f"{len(tree.findall('.//collision'))} matching collisions")


if __name__ == "__main__":
    main()
