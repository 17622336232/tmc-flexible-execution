"""Reconstruct the observed whole-task critical path and revalue it without a new inference."""
from pathlib import Path
from statistics import median

from experiments.flexible_execution.revision_audit import digest, read, save
from experiments.flexible_execution.timeline import read_trace_rows
from scripts.continuing_room_break_even import previous_turns

BASE = Path("outputs/continuing/matched-event-rooms-dev01")
CASE = "crossed-screens-house-56"
OUT = BASE.parent / "online-cost-dev01"


def observed(arm):
    folder = BASE / "live" / CASE / arm
    result = read(folder / "result.json")
    trace = folder / "timeline.jsonl"
    model = folder / "model_calls.jsonl"
    start, finish = result["start_clock"], result["finish_clock"]
    cameras = {v["step"]:v for v in read_trace_rows(trace)[0] if v["event"]=="camera_observation"}
    events = read_trace_rows(trace)[0]
    depart, = [v["monotonic_seconds"] for v in events if v["event"]=="action_started"]
    frames = {v: digest(folder / (v + ".jpg")) for v in ("front", "side")}
    if any(frames[k] != cameras[k]["image_sha256"] for k in frames):
        raise ValueError("Camera changed")
    calls = []
    for c in read_trace_rows(model)[0]:
        src = c["input"]["image"]
        send, done = c["dispatched_monotonic_seconds"], c["return_monotonic_seconds"]
        if not (c["image_sha256"] == [frames[src]] and cameras[src]["monotonic_seconds"] <= send <= done):
            raise ValueError("Request and camera clock differ")
        calls.append(dict(site=src, keys=[q["id"] for q in c["input"]["questions"]],
                          send_s=send-start, reply_s=done-start, service_s=done-send,
                          output=c.get("output"), error=(c.get("error") or {}).get("code")))
    side = cameras["side"]["monotonic_seconds"]
    # Calls occupy a single server. Movement and shutter are only on its critical
    # path where there is no in-flight remote request.
    busy = sum(c["service_s"] for c in calls)
    overlap = sum(max(0, min(side-start,c["reply_s"])-max(depart-start,c["send_s"])) for c in calls)
    if not (start <= depart <= side <= finish and all(c["reply_s"] <= finish for c in calls)
            and all(a["reply_s"] <= b["send_s"] for a,b in zip(calls,calls[1:]))):
        raise ValueError("Inconsistent single-slot clocks")
    return dict(task_s=finish-start, evidence=frames, calls=calls, travel_and_capture_s=side-depart,
                physical_overlapped_s=overlap, physical_exposed_s=side-depart-overlap,
                model_on_path_s=busy, other_on_path_s=finish-start-busy-(side-depart-overlap),
                delivered=result.get("report") is not None, source_digests=[digest(trace),digest(model),digest(folder/"result.json")])


def main():
    rows = {k: observed(k) for k in ("joint","serial","strong-prefetch")}
    j, s = rows["joint"], rows["serial"]
    if j["evidence"] != s["evidence"] or not j["delivered"] or not s["delivered"]:
        raise ValueError("No exact successful pair")
    components = {k: j[k]-s[k] for k in ("model_on_path_s","physical_exposed_s","other_on_path_s")}
    if abs(sum(components.values()) - (j["task_s"]-s["task_s"])) > 1e-6:
        raise ValueError("Full critical path did not reconcile")
    clocks = previous_turns()
    prior = [v for v in clocks if v["house"] == 56 and v["turn"] == "RotateLeft"]
    cal = read(BASE/"calibration/summary.json")
    old = read(BASE/"causal_branch.json")
    single, pair = cal["single_s"], cal["pair_s"]
    estimates = {
        "old_transfer" : old["old_planner_outward_and_shutter_s"],
        "previous_same_site" : median(v["s"] for v in prior) + read(BASE/"live/registration.json")["costs"][CASE]["photo"],
        "observed_joint" : j["travel_and_capture_s"],
    }
    branch = []
    for profile, trip in estimates.items():
        branch.append(dict(cost_source=profile, trip_s=trip,
            immediately_ask_front_and_move_s=max(single, trip)+2*single,
            collect_then_ask_side_and_group_front_s=trip+single+pair,
            ask_front_then_collect_then_separate_s=trip+3*single,
            collect_then_group_roots_upper_service_for_win_s=max(single,trip)+single-trip,
            scope="Observed front negative, side positive and subsequent front detail negative. Fixed calibrated clocks. "
                  "Cross-source combined roots still need a later released front detail; its service time is not assumed measured."))
    out=dict(new_requests=0, case=CASE, actual=rows, joint_minus_serial_s=j["task_s"]-s["task_s"],
             components_s=components, prior_same_site=[{k:v[k] for k in ("s","sha256")} for v in prior],
             branch=branch, stronger_comparator=dict(delivered=rows["strong-prefetch"]["delivered"],
                    same_first_answer_class_and_source=rows["strong-prefetch"]["calls"][0]["keys"]==j["calls"][0]["keys"],
                    terminal_time_s=rows["strong-prefetch"]["task_s"]))
    OUT.mkdir(exist_ok=True)
    save(OUT/"clock_audit.json",out)
    print("Actual critical path components (joint minus serial)",components)
    print("Registered and site-specific branch calculations",[(b["cost_source"],b["trip_s"],
           b["immediately_ask_front_and_move_s"],b["collect_then_ask_side_and_group_front_s"],
           b["collect_then_group_roots_upper_service_for_win_s"]) for b in branch])


if __name__=="__main__":
    main()

