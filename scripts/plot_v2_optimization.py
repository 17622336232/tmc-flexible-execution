"""Rebuild optimization diagnostics and the final paired effect figure."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"baseline": "#65758a", "revised": "#147b80"}
METHODS = ("conditional-a", "conditional-async", "conditional-prefetch", "conditional-planner", "conditional-stream")
NAMES = ("A", "ASYNC", "PREFETCH", "PLANNER", "STREAM")
LABELS = {"controller_decision": "Whole-task decisions", "controller_first_segment": "Action packets",
          "controller_next_segment": "Continuation packets", "frozen_judgment": "Frozen-photo judgment",
          "material_confirmation": "Material confirmation", "target_localization": "Target localization",
          "task_compile": "Joint task compilation", "whole_task_acceptance": "Whole-task acceptance"}


def _read(path):
    return json.loads(path.read_text(encoding="utf8"))


def _export(figure, path):
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _request_costs(data, output):
    rows = sorted(((role, data["role_totals"][role]) for role in LABELS), key=lambda item:
                  item[1]["reported_service_seconds_sum"])
    figure, axis = plt.subplots(figsize=(8.6, 4.5))
    seconds = [row["reported_service_seconds_sum"] for _, row in rows]
    axis.barh([LABELS[role] for role, _ in rows], seconds, color=COLORS["revised"])
    for i, (_, row) in enumerate(rows):
        axis.text(seconds[i] + 8, i, f"{row['dispatches']} sends", va="center", fontsize=9)
    axis.set_xlim(0, 1.22 * max(seconds))
    axis.set_xlabel("Sum of recorded request durations (s)")
    axis.set_title(f"Original v2 development  |  {data['dispatches']} actual sends", loc="left")
    axis.grid(axis="x", alpha=.18)
    axis.set_axisbelow(True)
    _export(figure, output / "original-v2-request-costs.png")


def _development(old, new, output):
    figure, axis = plt.subplots(figsize=(8.6, 4.4))
    for label, data, shift, color in (("Original v2", old, -.19, COLORS["baseline"]),
                                      ("Revised v2", new, .19, COLORS["revised"])):
        rows = {r["method"]: r for r in data["rows"] if r["case"] == "accel-dev-001"}
        for index, method in enumerate(METHODS):
            row = rows.get(method)
            if row is None:
                continue
            passed = row["state"] == "FINISHED_VALID"
            axis.bar(index + shift, row["seconds"] if passed else 480, .34, color=color,
                     hatch=None if passed else "////", edgecolor=color, alpha=.95,
                     label=label if method == next(item for item in METHODS if item in rows) else None)
    axis.set_xticks(range(len(NAMES)))
    axis.set_xticklabels(NAMES)
    axis.set_ylabel("Deadline-restricted complete-task cost (s)")
    axis.set_ylim(0, 520)
    axis.set_title("Development scene 001  |  recorded arms", loc="left")
    axis.grid(axis="y", alpha=.18)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower left")
    figure.text(.12, .008, "Hatched: unsuccessful by deadline. Missing: no recorded result. One scene.", fontsize=9)
    figure.subplots_adjust(bottom=.18)
    _export(figure, output / "scene-001-development.png")


def _paired(reports, output):
    rows = []
    for name, data in reports:
        comparison = data.get("matched_differences", {}).get("conditional-a")
        effect = comparison and comparison.get("restricted_seconds_difference")
        if effect and effect["cluster_bootstrap_95"]:
            rows.append((name, effect["mean"], *effect["cluster_bootstrap_95"], comparison["independent_groups"]))
    if not rows:
        return
    figure, axis = plt.subplots(figsize=(8.6, 2.7))
    for i, (name, point, low, high, groups) in enumerate(rows):
        axis.plot([low, high], [i, i], color=COLORS["revised"])
        axis.scatter(point, i, color=COLORS["revised"], s=35, zorder=3)
    axis.axvline(0, color="#65758a", ls="--", lw=1)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([f"{name}  (groups={groups})" for name, _, _, _, groups in rows])
    axis.set_xlabel("A minus PLANNER, restricted cost (s) and 95% interval")
    axis.set_title("Frozen online evaluation", loc="left")
    _export(figure, output / "formal-paired-effect.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    base = args.outputs / "protocol-v2-optimization"
    output = base / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _request_costs(_read(base / "audit/current_v2_timeline.json"), output)
    _development(_read(args.outputs / "acceleration-v2/development_results.json"),
                 _read(base / "development_results.json"), output)
    items = ((name, base / "reports" / directory / "summary.json") for name, directory in
             (("Custom confirmation", "custom-final"), ("Public validation", "public-final")))
    _paired([(name, _read(path)) for name, path in items if path.is_file()], output)


if __name__ == "__main__":
    main()
