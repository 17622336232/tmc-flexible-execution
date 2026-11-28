"""Audit already-sealed native actions, then independently check the cited camera pixels."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.flexible_execution.revision_audit import digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows
from research.flexible_scheduling.native_section import expected
from research.flexible_scheduling.ordinary_frontier import bind_reply, camera_goals, frontier, pose_matches
from scripts.continuing_crossworld_views import assert_snapshot, copied_renderer
from scripts.continuing_native_section_study import ROOT as OLD, jobs, register
from scripts.continuing_sensor_visibility import bodies, camera

OUT = Path("outputs/continuing/native-action-binding-dev01")


def seal_bindings():
    cfg, reg = register()
    sealed = read(OLD / "sealed.json")
    results = []
    for case, row, mode, folder in jobs(cfg, reg):
        log = folder / "model_calls.jsonl"
        if digest(log) != sealed[case["id"] + "/" + mode]:
            raise ValueError("Original decision changed before binding")
        call, = read_trace_rows(log)[0]
        frame = row["input"]["current_image_id"]
        photo = Path(row["source"])
        if call["input"] != row["input"] or call["image_sha256"] != [digest(photo)] or row["sha256"] != digest(photo):
            raise ValueError("Source or permission mismatch")
        geometry = call["input"]["geometry"]
        goals = camera_goals(case, geometry["position"], geometry["yaw"])
        roots = {k for k, spec in expected(case).items() if not spec["parents"] and spec["view"] == "front"}
        output = call.get("output") or {}
        facts, errors = bind_reply(case, output, {}, {frame: "front"}, roots)
        actual = output.get("action", {})
        matches = actual.get("type") == "move" and pose_matches(actual.get("goal"), goals["side"])
        values = {k: v for k, (v, _) in facts.items()}
        legal, skipped = frontier(case, values)
        results.append(dict(case=case["id"], method=mode, source=frame, image_sha256=digest(photo),
            log_sha256=digest(log), root_claims=values, root_coverage=set(facts)==roots, issues=errors,
            ordinary_move_to_authorized_side=matches, eligible=matches and not errors and set(facts)==roots,
            subsequently_legal=sorted(legal), skipped=sorted(skipped)))
    OUT.mkdir(parents=True, exist_ok=True)
    assert_snapshot(OUT / "eligibility.json", dict(source_seal_sha256=digest(OLD / "sealed.json"), rows=results))
    return cfg, results


def visual_instances(scene, renderer, meta, photo, devices):
    """Referee only. Each class is visible if one physical instance covers at least 36 pixels."""
    if digest(photo) != meta["sha256"]:
        raise ValueError("Source compression bytes changed")
    named = bodies(scene, renderer)
    replica, mask = camera(renderer, meta)
    with Image.open(photo) as image:
        error = np.abs(np.asarray(image.convert("RGB"), dtype=np.int16) - replica.astype(np.int16))
    alignment = float((error.max(axis=2) <= 18).mean())
    counts = Counter(dict(zip(*np.unique(mask, return_counts=True))))
    grouped = defaultdict(Counter)
    for (station, part), body in named.items():
        if "_station_" not in station:
            continue
        world, index = station.split("_station_")
        if world == "warehouse" and part.startswith("cargo_"):
            if part.startswith("cargo_tape_"):
                number = part[len("cargo_tape_"):]
            else:
                number = part[len("cargo_"):]
            if not number.isdigit():
                continue
            predicate = devices[int(index)-1]["color"] + "_carton"
            instance = station + "/" + number
        elif world == "campus":
            instance = station
            if part in {"vehicle", "vehicle_cabin", "windscreen"} or part.startswith("wheel_"):
                predicate = "vehicle"
            elif part in {"extinguisher","extinguisher_neck","extinguisher_handle","extinguisher_hose","extinguisher_label"}:
                predicate = "extinguisher"
            elif part in {"exit_obstruction", "crate_band"}:
                predicate = "obstruction_crate"
            else:
                continue
        else:
            continue
        grouped[predicate][instance] += int(counts[body])
    return alignment, {cat: dict(items) for cat, items in grouped.items()}


def grade():
    cfg, bindings = seal_bindings()
    rows = []
    for case in cfg["cases"]:
        ep = Path(case["source"])
        photo = ep / "private/robot/onboard_0001.jpg"
        with copied_renderer(ep) as (scene, renderer):
            rate, groups = visual_instances(scene, renderer, read(photo.with_suffix(".json")), photo,
                                           read(ep / "private/truth.json")["devices"])
        for item in (b for b in bindings if b["case"] == case["id"]):
            fields = []
            for key, value in item["root_claims"].items():
                category = expected(case)[key]["check"]
                pixels = groups.get(category, {})
                target = max(pixels.values(), default=0) >= 36
                fields.append(dict(id=key, claimed=value, predicate=category, pixels=pixels,
                                   target=target, correct=rate>=.90 and value==target))
            rows.append({**item, "alignment": rate, "fields": fields})
    unique = {(v["image_sha256"], f["id"]): f["correct"] for v in rows for f in v["fields"]}
    out = dict(eligibility_sha256=digest(OUT/"eligibility.json"), new_requests=0, registered=len(rows),
        bound=sum(r["eligible"] for r in rows), repeated_field_observations=sum(len(r["fields"]) for r in rows),
        correct_observations=sum(f["correct"] for r in rows for f in r["fields"]),
        distinct_fields=len(unique), distinct_correct=sum(unique.values()), rows=rows,
        scope="Independent post-seal grade of original first-frame claims on two archival JPEGs, not an online episode.")
    save(OUT/"summary.json", out)
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal","grade"))
    if parser.parse_args().mode=="seal":
        print("Sealed",len(seal_bindings()[1]))
    else:
        result=grade()
        print({k:v for k,v in result.items() if k not in ("rows","scope")})

