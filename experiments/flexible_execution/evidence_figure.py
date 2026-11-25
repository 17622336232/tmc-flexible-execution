"""Draw timing, candidate-check and paired-result figures from visual evidence."""

import argparse
import json
from pathlib import Path

from .live_audit import _camera_frames, _verified_judgment, audit_live_connection


METHOD_LABELS = {"paired-serial": "Serial", "paired-async": "Async", "conditional-serial": "Serial",
                 "conditional-async": "Async", "conditional-a": "A", "conditional-prefetch": "Prefetch",
                 "conditional-planner": "Planner", "conditional-stream": "Stream",
                 "conditional-stream-buffered": "Buffered stream", "conditional-rank-ordinary": "Ordinary ranking",
                 "conditional-rank-b1": "B1", "conditional-rank-c": "C", "conditional-rank-heuristic": "Heuristic",
                 "conditional-rank-retrieval": "Retrieval"}
COHORT_LABELS = {"LEGACY_TASK_DEVELOPMENT_PAIR_SCHEDULE": "UAV development", "CONDITIONAL_RANKER_DEVELOPMENT": "Ranking development",
                 "CUSTOM_VISUAL_MATCHED_HELDOUT": "UAV held-out evaluation", "PUBLIC_VISUAL_MATCHED_DEVELOPMENT": "ProcTHOR validation",
                 "PUBLIC_VISUAL_MATCHED_HELDOUT": "ProcTHOR held-out evaluation"}


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
    _, model = _verified_judgment(trace, _rows(robot / "model_calls.jsonl"), _camera_frames(robot))
    if not model:
        raise ValueError("The plotted model reply must match the exact frozen query")
    response = ((model["dispatched_monotonic_seconds"], model["return_monotonic_seconds"])
                if "dispatched_monotonic_seconds" in model else _interval(trace, "model_request", 3))
    shutter = next(row["monotonic_seconds"] for row in trace if row["event"] == "frozen_photo" and row["step"] == step)
    return response, _interval(trace, "action", step), shutter


def _room(root):
    trace = _rows(root / "timeline.jsonl")
    call = next(row for row in _rows(root / "model_calls.jsonl")
                if row["input"].get("frame") == "front" and "output" in row)
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


def draw_ranking_replay(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(report.read_text(encoding="utf8"))
    if data["mode"] != "OFFLINE_FIXED_POOLS_FROM_ACTUAL_VISUAL_FRONTENDS" or not data["snapshots"]:
        raise ValueError("Candidate-check plots require replayed pools from recorded visual inputs")
    groups = {}
    for snapshot in data["snapshots"]:
        methods = snapshot["methods"]
        groups.setdefault(snapshot["group"], []).append((methods["conditional-rank-b1"], methods["conditional-rank-c"]))
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    labels = []
    matches = 0
    for index, rows in enumerate(groups.values()):
        labels.append(f"Layout {index + 1}\n{len(rows)} recorded pools")
        matches += sum(fixed["selected"] == adaptive["selected"] for fixed, adaptive in rows)
        for offset, method, color, key in ((-.17, "B1 · fixed budget", "#335D83", 0),
                                            (.17, "C · adaptive budget", "#268270", 1)):
            checks = sum(row[key]["checked"] for row in rows)
            axis.bar(index + offset, checks, width=.29, color=color, label=method if index == 0 else None)
            axis.annotate(str(checks), (index + offset, checks), xytext=(0, 3), textcoords="offset points",
                          ha="center", va="bottom", fontsize=10)
    axis.set_title("Offline replay of identical visual candidate pools", loc="left", fontweight="bold")
    axis.set_ylabel("Candidate checks")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels)
    axis.set_ylim(0, axis.get_ylim()[1] * 1.28)
    axis.legend(frameon=False, loc="upper left", ncol=2)
    for boundary in ("top", "right"):
        axis.spines[boundary].set_visible(False)
    axis.grid(axis="y", alpha=.18)
    axis.set_axisbelow(True)
    figure.text(.5, .02, f"Same execution decision in {matches}/{len(data['snapshots'])} replays. No new model requests.",
                ha="center", fontsize=9, color="#46505B")
    figure.tight_layout(rect=(0, .07, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def draw_paired_results(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(report.read_text(encoding="utf8"))
    pairs = list(data["matched_differences"].items())
    if not pairs:
        raise ValueError("A paired result figure needs at least one recorded comparison")
    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(8.6, max(3.1, 1.6 + .43 * len(pairs))))
    metrics = (("success_difference", "Completion difference (percentage points)", 100),
               ("restricted_seconds_difference", "Time difference (seconds)", 1))
    for axis, (field, caption, scale) in zip(axes, metrics):
        axis.axvline(0, color="#777F87", linewidth=.9, linestyle="--")
        for index, (_, comparison) in enumerate(pairs):
            measure = comparison[field]
            interval = measure["cluster_bootstrap_95"]
            if interval is not None:
                axis.hlines(index, interval[0] * scale, interval[1] * scale, color="#335D83", linewidth=2)
            axis.plot(measure["mean"] * scale, index, "o", color="#335D83", markersize=5)
        axis.set_xlabel(caption)
        axis.grid(axis="x", alpha=.17)
        axis.set_axisbelow(True)
        for boundary in ("top", "right", "left"):
            axis.spines[boundary].set_visible(False)
    axes[0].set_yticks(range(len(pairs)))
    axes[0].set_yticklabels([f"{METHOD_LABELS.get(method, method)}  (n={row['independent_groups']})"
                                for method, row in pairs])
    axes[0].set_ylim(len(pairs) - .5, -.5)
    cohort = COHORT_LABELS.get(data.get("role"), "Paired results")
    reference = METHOD_LABELS.get(data["reference"], data["reference"])
    figure.suptitle(f"{cohort} · compared with {reference}", fontweight="bold")
    figure.text(.5, .035, "Lines show 95% paired intervals. Failed runs contribute their registered time limit.",
                ha="center", fontsize=9, color="#46505B")
    figure.tight_layout(rect=(0, .09, 1, .93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--uav-episode", type=Path)
    source.add_argument("--fixed-pools", type=Path)
    source.add_argument("--paired-summary", type=Path)
    parser.add_argument("--public-episode", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.fixed_pools or args.paired_summary:
        if args.public_episode:
            parser.error("--public-episode is only used with --uav-episode")
        if args.fixed_pools:
            draw_ranking_replay(args.fixed_pools, args.output)
        else:
            draw_paired_results(args.paired_summary, args.output)
        return
    if not args.public_episode:
        parser.error("--uav-episode also requires --public-episode")
    draw(args.uav_episode, args.public_episode, args.output)


if __name__ == "__main__":
    main()
