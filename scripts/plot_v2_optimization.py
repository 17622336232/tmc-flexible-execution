"""Rebuild optimization diagnostics and the final evaluation figures."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


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
                     hatch=None if passed else "////", edgecolor=color if passed else "#39475a", alpha=.95,
                     label=label if method == next(item for item in METHODS if item in rows) else None)
    axis.set_xticks(range(len(NAMES)))
    axis.set_xticklabels(NAMES)
    axis.set_ylabel("Deadline-restricted complete-task cost (s)")
    axis.set_ylim(0, 520)
    axis.set_title("Development scene 001  |  recorded arms", loc="left")
    axis.grid(axis="y", alpha=.18)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2)
    figure.text(.12, .008, "Hatched: unsuccessful by deadline. Missing: no recorded result. One scene.", fontsize=9)
    figure.subplots_adjust(bottom=.25)
    _export(figure, output / "scene-001-development.png")


def _development_overview(data, catalog, output):
    cases = catalog["cases"]
    rows = {(row["case"], row["method"]): row for row in data["rows"]}
    ymax = 1.08 * max(case["episode_timeout_seconds"] for case in cases)
    figure, axes = plt.subplots(1, len(cases), sharey=True, squeeze=False, figsize=(12, 4.2))
    for axis, case in zip(axes[0], cases):
        methods = [method for method in METHODS if method in case["order"]]
        deadline = case["episode_timeout_seconds"]
        for index, method in enumerate(methods):
            row = rows.get((case["id"], method))
            if row is None:
                axis.text(index, 14, "pending", ha="center", fontsize=8, color=COLORS["baseline"])
                continue
            passed = row["state"] == "FINISHED_VALID"
            axis.bar(index, row["seconds"] if passed else deadline, width=.65,
                     color=COLORS["revised"] if passed else "#eef0f2", edgecolor=COLORS["baseline"],
                     hatch=None if passed else "////", linewidth=.6)
            if not passed:
                axis.scatter(index, row["seconds"], color="#bb6429", marker="D", s=24, zorder=3)
        axis.set_xticks(range(len(methods)))
        axis.set_xticklabels([NAMES[METHODS.index(method)] for method in methods], rotation=28, ha="right")
        axis.set_xlim(-.55, len(methods) - .45)
        axis.set_ylim(0, ymax)
        axis.set_title(f"{case['id'].split('-')[-1]}  {case['family'].replace('_', ' ')}", loc="left")
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    axes[0][0].set_ylabel("Deadline-restricted complete-task cost (s)")
    figure.suptitle(f"Revised v2 development   {data['recorded_episodes']}/{data['registered_episodes']} recorded",
                    x=.08, ha="left")
    figure.legend(handles=[Patch(facecolor=COLORS["revised"], label="Complete task"),
                          Patch(facecolor="#eef0f2", edgecolor=COLORS["baseline"], hatch="////",
                                label="Unsuccessful at deadline cost"),
                          Line2D([], [], linestyle="none", marker="D", color="#bb6429", markersize=5,
                                 label="Actual active task time on failure")],
                  loc="lower center", ncol=3, frameon=False)
    figure.subplots_adjust(bottom=.27, top=.81, wspace=.13)
    _export(figure, output / "development-overview.png")


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


def _formal_outcomes(reports, output):
    if not reports:
        return
    outcomes = (("FINISHED_VALID", "Complete task", "#147b80"),
                ("FINISHED_TASK_FAILURE", "Task failure", "#a05728"),
                ("TIMEOUT", "Deadline expired", "#65758a"))
    figure, axes = plt.subplots(1, len(reports), squeeze=False, figsize=(5.6 * len(reports), 3.7))
    for axis, (title, report) in zip(axes[0], reports):
        methods = [(label, report["methods"][method]) for method, label in zip(METHODS, NAMES)
                   if method in report["methods"]]
        for index, (label, result) in enumerate(methods):
            total = result["recorded"]
            if total == 0 or total != sum(result["statuses"].get(state, 0) for state, _, _ in outcomes):
                raise ValueError(f"Unexpected outcome totals for {title}, {label}")
            left = 0.
            for state, _, color in outcomes:
                count = result["statuses"].get(state, 0)
                share = count / total
                axis.barh(index, share, left=left, height=.6, color=color)
                if share >= .08:
                    axis.text(left + share / 2, index, str(count), color="white", ha="center", va="center")
                left += share
        axis.set_yticks(range(len(methods)))
        axis.set_yticklabels([f"{label} ({result['recorded']})" for label, result in methods])
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xticks([0, .25, .5, .75, 1])
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.set_xlabel("Share of recorded episodes")
        axis.set_title(title, loc="left")
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
    figure.suptitle("Independently scored task outcomes", x=.115, ha="left")
    figure.legend(handles=[Patch(facecolor=color, label=label) for _, label, color in outcomes],
                  loc="lower center", ncol=len(outcomes), frameon=False)
    figure.subplots_adjust(left=.145, bottom=.25, top=.78, wspace=.42)
    _export(figure, output / "formal-outcomes.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    base = args.outputs / "protocol-v2-optimization"
    output = base / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _request_costs(_read(base / "audit/current_v2_timeline.json"), output)
    revised = _read(base / "development_results.json")
    _development(_read(args.outputs / "acceleration-v2/development_results.json"), revised, output)
    _development_overview(revised, _read(base / "development/pilot_manifest.json"), output)
    items = ((name, base / "reports" / directory / "summary.json") for name, directory in
             (("Custom confirmation", "custom-final"), ("Public validation", "public-final")))
    reports = []
    for name, path in items:
        if path.is_file():
            report = _read(path)
            if report["recorded_episodes"] == report["planned_episodes"]:
                reports.append((name, report))
    _paired(reports, output)
    _formal_outcomes(reports, output)


if __name__ == "__main__":
    main()
