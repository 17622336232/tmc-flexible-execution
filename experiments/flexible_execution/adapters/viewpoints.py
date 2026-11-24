"""Choose map-admitted camera views from the current frozen RGB and LiDAR."""

from itertools import islice
from math import atan2, dist, hypot


def xyz(position):
    return tuple(position[axis] for axis in ("x", "y", "z"))


def surface_views(position, geometry, bbox):
    """Propose where to look again, without treating a surface hit as object identity."""
    grid = geometry.get("measured_surface_grid", {})
    columns, rows = grid.get("columns", 0), grid.get("rows", 0)
    if columns <= 0 or rows <= 0:
        return
    left, top, right, bottom = bbox
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    visible = []
    for cell in grid.get("cells", []):
        column, row = cell["pixel_cell"]
        if (left >= (column + 1) / columns or right <= column / columns
                or top >= (row + 1) / rows or bottom <= row / rows):
            continue
        x, y, _ = cell["world_xyz"]
        dx, dy = x - position["x"], y - position["y"]
        distance = hypot(dx, dy)
        if distance < 1.7:
            continue
        proximity = hypot((column + 0.5) / columns - center_x, (row + 0.5) / rows - center_y)
        visible.append((proximity, -cell["measured_points"], column, row, x, y, dx, dy, distance))
    for _, _, column, row, x, y, dx, dy, distance in sorted(visible):
        for standoff, shift in ((2.4, 0), (1.8, 0), (2.4, 0.35), (2.4, -0.35)):
            if distance <= standoff + 0.25:
                continue
            px = x - dx / distance * standoff - dy / distance * shift
            py = y - dy / distance * standoff + dx / distance * shift
            yield {"x": px, "y": py, "z": position["z"], "yaw": atan2(y - py, x - px)}, (column, row)


def _candidates(backend, observation, geometry, target, trace):
    for standoff in (3.0, 2.0, 1.5):
        try:
            measured = backend.adapter.resolve_visual_target(observation, target.bbox, standoff)
        except (RuntimeError, ValueError) as error:
            trace.event("viewpoint_rejected", 1, target_id=target.id, method="visual_target",
                        standoff=standoff, detail=str(error)[:300])
        else:
            yield measured["goal"], {"method": "visual_target", "standoff": standoff}
    for goal, cell in islice(surface_views(observation["position"], geometry, target.bbox), 5):
        try:
            backend.check_goal(goal)
        except (RuntimeError, ValueError) as error:
            trace.event("viewpoint_rejected", 1, target_id=target.id, method="surface_guided_view",
                        surface_cell=cell, detail=str(error)[:300])
        else:
            yield goal, {"method": "surface_guided_view", "surface_cell": cell}


def measured_views(backend, observation, geometry, targets, trace):
    views = []
    for target in targets:
        for goal, details in _candidates(backend, observation, geometry, target, trace):
            if any(dist(xyz(goal), xyz(known)) < 0.35 for _, known in views):
                trace.event("viewpoint_rejected", 1, target_id=target.id,
                            detail="Not a distinct measured view", **details)
                continue
            views.append((target, goal))
            trace.event("viewpoint_measured", 1, target_id=target.id, goal=goal,
                        source_frame=observation["observation_id"], **details)
            break
    return sorted(views, key=lambda pair: dist(xyz(observation["position"]), xyz(pair[1])))
