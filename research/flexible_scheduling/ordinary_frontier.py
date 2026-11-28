"""Bind ordinary controller decisions to explicit, user-supplied inspection questions and poses."""
from math import atan2, cos, dist, isfinite, radians, sin

from .native_section import expected


def camera_goals(case, position, yaw):
    offset = radians(case["angle_degrees"]) * {"left": 1, "right": -1}[case["direction"]]
    return {name: {**position, "yaw": yaw + delta} for name, delta in (("front", 0), ("side", offset))}


def pose_matches(measured, desired, metres=.10, radians_tol=.05):
    try:
        p = [measured[k] for k in ("x", "y", "z")]
        q = [desired[k] for k in ("x", "y", "z")]
        delta = measured["yaw"] - desired["yaw"]
        values = [*p, *q, measured["yaw"], desired["yaw"]]
        return (all(type(v) in (int, float) and isfinite(v) for v in values)
                and dist(p, q) <= metres and abs(atan2(sin(delta), cos(delta))) <= radians_tol)
    except (KeyError, TypeError, ValueError):
        return False


def frontier(case, prior):
    """OR gates open on an actual positive, close when every alternative is known false."""
    fields = expected(case)
    opened, skipped = set(), set()
    for key, spec in fields.items():
        parents = spec["parents"]
        if any(p not in fields or int(p[1:]) >= int(key[1:]) for p in parents):
            raise ValueError("A prerequisite must refer to an earlier declared question")
        if key in prior:
            continue
        if not parents or any(prior.get(p) is True for p in parents):
            opened.add(key)
        elif all(p in skipped or prior.get(p) is False for p in parents):
            skipped.add(key)
    return opened, skipped


def bind_reply(case, reply, prior, sources, permitted):
    """Use only booleans cited to a supplied photo at its declared view, after its gate opened."""
    fields = expected(case)
    legal, _ = frontier(case, prior)
    rows = list(reply.get("records") or [])
    if (reply.get("action") or {}).get("type") == "submit":
        rows.extend((reply.get("report") or {}).get("records") or [])
    found, issues = {}, set()
    for row in rows:
        if not isinstance(row, dict):
            issues.add("malformed-row")
            continue
        key = row.get("object_id")
        data = row.get("facts")
        if not isinstance(key, str) or key not in fields or not isinstance(data, dict):
            issues.add("unknown-record")
            continue
        frame = row.get("evidence_image_id")
        if data.get("visible") is None:
            continue
        value = data["visible"]
        if type(value) is not bool or set(data) != {"visible"}:
            issues.add("malformed:" + key)
        elif key in prior:
            if value != prior[key]:
                issues.add("conflict:" + key)
        elif key not in legal or key not in permitted:
            issues.add("premature:" + key)
        elif not isinstance(frame, str) or sources.get(frame) != fields[key]["view"]:
            issues.add("wrong-source:" + key)
        elif key in found and found[key] != (value, frame):
            issues.add("conflict:" + key)
        else:
            found[key] = value, frame
    return found, sorted(issues)

