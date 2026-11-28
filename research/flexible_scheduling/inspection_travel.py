"""Dispatch a static, two-stop inspection over one robot and one vision service."""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from time import monotonic, sleep


VISION_PROMPT = """Inspect only the attached fresh robot camera image at this inspection stop.
Locate the named object's physical identity label. For an equipment cabinet use its body identity
plate; its display or header may carry a different number. For a warehouse bay use that bay's label.
Report the identifier actually read and the tight normalized [left,top,right,bottom] box of the
identifier label on this image. For cracked, inspect the same cabinet's front housing; for
box_count, count the actual cartons in this same bay across all its shelves. Return only
{"visible_id":string or null,"label_box":[l,t,r,b] or null,"value":boolean for cracked or
nonnegative integer for box_count or null}. If the ID, association, or property is unclear,
leave the corresponding fields null. A requested identifier alone is never image evidence."""


@dataclass(frozen=True)
class Capture:
    id: str
    data: bytes

    @property
    def sha(self):
        return sha256(self.data).hexdigest()


def accepted_read(reply, stop, field, frame):
    if not isinstance(reply, dict) or set(reply) != {"visible_id", "label_box", "value"}:
        return None
    value = reply["value"]
    if (reply["visible_id"] != stop["target"] or
            (type(value) is not bool if field == "cracked" else type(value) is not int or value < 0)):
        return None
    box = reply["label_box"]
    if (not isinstance(box, list) or len(box) != 4 or
            not all(type(n) in (float, int) and isfinite(n) for n in box) or
            not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
        return None
    return dict(id=stop["target"], field=field, value=value, label_box=box,
                source=frame.id, sha256=frame.sha)


def positive(reading, field):
    if reading is None:
        return False
    return reading["value"] is True if field == "cracked" else reading["value"] > 2


def request(client, frame, stop, field):
    if field not in ("cracked", "box_count"):
        raise ValueError("Unknown field")
    data = dict(target=stop["target"], field=field, image_id=frame.id, sha256=frame.sha)
    answer = client.request(VISION_PROMPT, data, [dict(label="Onboard camera "+frame.id, data=frame.data)],
                            attempt_limit=1)
    return accepted_read(answer, stop, field, frame)


def schedule(method, grant):
    if grant not in ("mandatory-archive", "after-positive-only"):
        raise ValueError("Unknown collection permission")
    # Edges refer to the end of a job. Model B also requires a positive return from A.
    common = {"read-a": ("capture-a",), "read-b": ("positive-a", "capture-b"),
              "finish": ("home", "read-a", "branch-settled")}
    if method == "serial":
        capture_deps = ("travel-b",) if grant == "mandatory-archive" else ("travel-b","positive-a")
        return {**common, "travel-b": ("read-a",), "capture-b": capture_deps,
                "home": ("travel-b", "branch-settled")}
    if method == "strong-prefetch":
        capture_deps = ("travel-b",) if grant == "mandatory-archive" else ("travel-b","positive-a")
        return {**common, "travel-b": ("capture-a",), "capture-b": capture_deps,
                "home": ("travel-b", "capture-settled")}
    if method != "dependency-schedule":
        raise ValueError("Unknown travel policy")
    # Compile the public work order from resource and permission constraints.
    # Absolute movement to a specified stop has no visual prerequisite.
    edges = {**common, "travel-b": ("capture-a",), "capture-b": ("travel-b",),
             "home": ("travel-b", "capture-settled")}
    if grant == "after-positive-only":
        edges["capture-b"] += ("positive-a",)
    return edges


def execute(case, stops, platform, client, method, deadline):
    """The platform owns actual movements, camera receipts and event clocks. No grader is present."""
    field, grant = case["field"], case["b_capture"]
    graph = schedule(method, grant)
    platform.move("a", stops[0]["goal"])
    first = platform.take("a")
    archived = {"a": first.sha}
    readings = {}
    if graph["travel-b"] == ("read-a",):
        readings["a"] = request(client, first, stops[0], field)
        platform.move("b", stops[1]["goal"])
        active = positive(readings["a"], field)
        if grant == "mandatory-archive" or active:
            second = platform.take("b")
            archived["b"] = second.sha
            if active:
                readings["b"] = request(client, second, stops[1], field)
        platform.home()
    else:
        delivered = Future()
        with ThreadPoolExecutor(max_workers=1) as pool:
            previous = client.calls
            primary = pool.submit(request, client, first, stops[0], field)
            def continue_if_needed():
                finding = primary.result()
                if not positive(finding, field):
                    return {}
                second = delivered.result()
                if second is None:
                    return {}
                return {"b": request(client, second, stops[1], field)}
            secondary = pool.submit(continue_if_needed)
            try:
                while not primary.done() and client.calls == previous and monotonic() < deadline:
                    sleep(.005)
                if monotonic() >= deadline:
                    raise TimeoutError("Mission deadline before transit")
                platform.move("b", stops[1]["goal"])
                permitted = (graph["capture-b"] == ("travel-b",) or positive(primary.result(), field))
                if permitted:
                    second = platform.take("b")
                    archived["b"] = second.sha
                    delivered.set_result(second)
                else:
                    delivered.set_result(None)
                platform.home()
                readings["a"] = primary.result()
                readings.update(secondary.result())
            finally:
                if not delivered.done():
                    delivered.set_result(None)
    return dict(archived=archived, readings=readings, returned=True)

