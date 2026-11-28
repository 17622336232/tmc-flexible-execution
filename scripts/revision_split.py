"""Blind metadata-only screen of the local public pool for a future independent test.

Only asset identifiers and pre-existing split membership are compared. No image,
object pose, task answer or evaluation label is exported or consulted.
"""
import gzip
import json
from hashlib import sha256
from pathlib import Path

from experiments.flexible_execution.public_schedule import TEST_SHA256
from experiments.flexible_execution.revision_audit import BASE, OUT, read, save, digest


TASKS = (
    ("two-front-or", ("DeskLamp", "FloorLamp", "Television"),
     "先用前方照片分别判断台灯和落地灯是否可见。若至少一种可见，用右侧照片判断电视是否可见。"
     "允许在等待时转向取证，结束前恢复朝向。报告每项已确定的结论及照片；看不清的项写无法判断。"),
    ("two-side-shared", ("Microwave", "Toaster", "CoffeeMachine"),
     "先判断前方是否看得到微波炉。如果看得到，使用同一张左侧照片分别核对烤面包机和咖啡机。"
     "允许在等待时转向取证，结束前恢复朝向。报告已触发的检查及照片；看不清的项写无法判断。"),
)


def assets(scene, target_types=None):
    ids = set()
    stack = [*scene["objects"], *scene.get("doors", ()), *scene.get("windows", ())]
    while stack:
        obj = stack.pop()
        if obj.get("assetId") and (target_types is None or obj.get("id", "").split("|")[0] in target_types):
            ids.add(obj["assetId"])
        stack.extend(obj.get("children") or ())
    return ids


def metadata(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        for index, line in enumerate(stream):
            scene = json.loads(line)
            # Coarse side/diagonal distances ignore translation, rotation, mirroring
            # and vertex order. Different arrangements can conservatively collide.
            import math
            from itertools import combinations
            shapes = sorted(sorted(round(math.hypot(a["x"]-b["x"], a["z"]-b["z"]) * 2)
                                   for a, b in combinations(room["floorPolygon"], 2)) for room in scene["rooms"])
            geometry = sha256(json.dumps(shapes).encode()).hexdigest()
            yield index, (assets(scene), geometry, [assets(scene, types) for _, types, _ in TASKS])


def main():
    root = Path("outputs/public-visual")
    test = root / "procthor/test.jsonl.gz"
    val = root / "procthor/val.jsonl.gz"
    assert digest(test) == TEST_SHA256
    seen_test = {c["official_house_index"] for c in read(BASE / "public-final/pilot_manifest.json")["cases"]}
    seen_val = {0, *(c["official_house_index"] for c in read(root / "validation-pilot-v1/pilot_manifest.json")["cases"])}
    # Additional archived smoke/calibration metadata; never open an unrun task.
    for top in ("validation", "closed-loop", "calibration", "smoke"):
        for p in root.glob(top+"*/**/*.json"):
            if p.stat().st_size > 2_000_000:
                continue
            d = read(p)
            if isinstance(d,dict) and type(d.get("house_index")) is int:
                seen_val.add(d["house_index"])
    exposed, seen_geometry = set(), set()
    for path, seen in [(test,seen_test),(val,seen_val)]:
        for index, (group, geometry, _) in metadata(path):
            if index in seen:
                exposed.update(group)
                seen_geometry.add(geometry)
    sets = dict(metadata(test))
    candidates = [i for i,(g, geometry, _) in sets.items()
                  if i not in seen_test and geometry not in seen_geometry and g and not exposed.intersection(g)]
    order = sorted(candidates, key=lambda i: sha256(f"revision-source-holdout-20261128/{i}".encode()).digest())
    reserved = []
    used = set(exposed)
    for i in order:
        if not sets[i][0].intersection(used):
            reserved.append(i)
            used.update(sets[i][0])
    # Assignment precedes asset eligibility. Absence, visibility and eventual outcomes
    # cannot be used as a success filter. Shared background catalog assets are permitted.
    drawn = []
    reserved_assets = set(exposed)
    geometries = set(seen_geometry)
    for index in sorted(sets, key=lambda i: sha256(f"task-source-reserve-v1/{i}".encode()).digest()):
        slot = int.from_bytes(sha256(f"task-form-v1/{index}".encode()).digest()[:2], "big") % len(TASKS)
        _, geometry, groups = sets[index]
        group = groups[slot]
        if index in seen_test or geometry in geometries or group & reserved_assets:
            continue
        reserved_assets.update(group)
        geometries.add(geometry)
        drawn.append(dict(house_index=index, geometry_group=geometry, structure=TASKS[slot][0],
                          instruction=TASKS[slot][2], split="reserve-no-render"))
        if len(drawn) == 32:
            break
    record = dict(role="asset-screened reserve only; no images opened, no test run", archive_sha256=digest(test),
                  exposed_test_houses=len(seen_test), known_validation_houses=len(seen_val), houses_scanned=len(sets),
                  exposed_asset_count=len(exposed), unseen_house_ids=len(sets.keys()-seen_test), eligible=len(candidates),
                  pairwise_asset_disjoint_reserve=reserved, definition="all movable object, door and window catalog asset IDs, including children; no task outcome labels",
                  limitations="Common background assets may recur in the separately listed task-source split. Polygon-distance grouping uses half-metre bins; it cannot certify all perceptual near-duplicates.",
                  task_source_reserve=len(drawn), reserve_role="capacity only, independent sample size not determined")
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT / "public_reserve.json",record)
    save(OUT / "public_reserve_manifest.json", dict(role="sealed candidates only; paired development gate closed",
         source_archive_sha256=digest(test), task_design=TASKS, rows=drawn))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
