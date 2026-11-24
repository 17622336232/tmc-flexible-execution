"""Draw a timing figure from two archived RGB acquisition episodes."""

import argparse
import json
from pathlib import Path

from .live_audit import audit_live_connection


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def _interval(rows, kind, step):
    return tuple(next(row["monotonic_seconds"] for row in rows if row["step"] == step
                      and row["event"] == f"{kind}_{phase}") for phase in ("started", "finished"))


def _uav(root):
    robot = root / "private/robot"
    trace = _rows(robot / "timeline.jsonl")
    audit = audit_live_connection(root)
    material = next(row for row in audit["accepted_materials"] if row["motion_started_while_pending"]
                    and row["required_source_verified"] and row["used_by_full_controller"])
    step = int(material["image_id"].split("_")[1])
    binding = next(row["binding"] for row in trace if row["event"] == "query_submitted")
    model = next(row for row in _rows(robot / "model_calls.jsonl")
                 if row["input"].get("frame_id") == binding["frame_id"] and row.get("output", {}).get("outcome"))
    response = ((model["dispatched_monotonic_seconds"], model["return_monotonic_seconds"])
                if "dispatched_monotonic_seconds" in model else _interval(trace, "model_request", 3))
    shutter = next(row["monotonic_seconds"] for row in trace if row["event"] == "frozen_photo" and row["step"] == step)
    return response, _interval(trace, "action", step), shutter


def _room(root):
    trace = _rows(root / "timeline.jsonl")
    call = next(row for row in _rows(root / "model_calls.jsonl") if row["input"].get("frame") == "front")
    motion = next(row["step"] for row in trace if row["event"] == "action_started")
    shutter = next(row["monotonic_seconds"] for row in trace if row["event"] == "camera_observation"
                   and row["step"] == "side")
    return (call["dispatched_monotonic_seconds"], call["return_monotonic_seconds"]), _interval(trace, "action", motion), shutter


def draw(uav, room, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uav_timing = audit_live_connection(uav)["model_timing_source"]
    traces = (("a   UAV inspection", _uav(uav), "Frozen query" if uav_timing == "recorded_dispatch_and_model_return"
               else "Frozen query (request timeline)"),
              ("b   ProcTHOR validation room", _room(room), "Frozen query"))
    figure, axes = plt.subplots(nrows=2, figsize=(7.4, 4.1), sharex=True)
    for axis, (title, (reply, action, shutter), query_label) in zip(axes, traces):
        origin = reply[0]
        stop, begin, end, image = [time - origin for time in (reply[1], *action, shutter)]
        axis.barh(1, stop, left=0, height=.26, color="#335D83")
        axis.barh(0, end - begin, left=begin, height=.26, color="#268270")
        axis.axvline(stop, color="#335D83", linestyle="--", linewidth=.9)
        axis.plot(image, 0, "D", color="#A35727", markersize=6)
        axis.set_yticks([0, 1])
        axis.set_yticklabels(("Robot motion", query_label))
        axis.set_ylim(-.48, 1.55)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.text(stop, 1.25, "reply", color="#335D83", ha="center", fontsize=9)
        axis.text(image, -.36, "photo", color="#A35727", ha="center", fontsize=9)
        axis.grid(axis="x", alpha=.19)
        for boundary in ("top", "right", "left"):
            axis.spines[boundary].set_visible(False)
    axes[-1].set_xlabel("Seconds from frozen model request")
    figure.tight_layout(pad=1.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uav-episode", required=True, type=Path)
    parser.add_argument("--public-episode", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    draw(args.uav_episode, args.public_episode, args.output)


if __name__ == "__main__":
    main()
