"""Rebuild paired development metrics from saved arm outcomes and event ledgers."""

import csv
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean

from experiments.task_attack.runtime import write_json

from .clean_coverage import audit_run
from .pilot_schedule import _read, _valid_preflight
from .timeline import summarize


RECORDED = {"FINISHED_VALID", "FINISHED_TASK_FAILURE", "TIMEOUT"}


def _arm_row(directory, case, method):
    arm = directory / case["id"] / method
    job = _read(arm / "job.json")
    row = {"case": case["id"], "group": case.get("group_id", case["id"]), "family": case["family"],
           "domain": case["domain"], "seed": case["seed"], "method": method,
           "deadline_seconds": case["episode_timeout_seconds"], "state": "PENDING"}
    if not job:
        return row
    row.update(state=job["state"], logical_id=job["id"], infrastructure_attempts=job["attempt"],
               actual_dispatches=job.get("request_budget", {}).get("reserved"),
               cumulative_task_seconds=job.get("charged_seconds"))
    if job["state"] not in RECORDED:
        return row
    output = job.get("artifact_dir")
    row.update(artifact_dir=output, success=False, restricted_seconds=case["episode_timeout_seconds"])
    if not output:
        return row
    root = Path(output)
    preflight, result = _read(root / "clean_preflight.json"), _read(root / "result.json")
    if preflight:
        _valid_preflight(preflight, case, method, root)
    for earned, name in (("probe_completed", "base_frontend_completed"),
                         ("conditional_query_bound", "conditional_query_bound"),
                         ("conditional_materials", "active_conditional_materials")):
        row[name] = (result or {}).get("robot_metrics", {}).get(earned)
    row["legacy_outcome"] = (result or {}).get("score", {}).get("outcome")
    row["failure_reason"] = (job.get("result") or {}).get("reason")
    row["legacy_wrong_accepted"] = (result or {}).get("score", {}).get("wrong_accepted")
    timing = root / "private/robot/timeline.jsonl"
    if timing.is_file():
        clock = summarize(timing)
        row.update(model_request_seconds=clock["model_request_seconds"],
                   model_action_overlap_seconds=clock["model_action_overlap_seconds"],
                   action_seconds=sum(interval["seconds"] for interval in clock["intervals"].get("action", [])),
                   search_seconds=sum(interval["seconds"] for interval in clock["intervals"].get("search", [])))
    if result and result.get("execution_completed") and (root / "clean_coverage.json").is_file():
        coverage = audit_run(root)
        saved = _read(root / "clean_coverage.json")
        if saved != coverage:
            raise ValueError("Stored coverage does not agree with raw evidence for " + row["logical_id"])
        row["complete_coverage"] = coverage["complete"]
        row["missing_objects"] = len(coverage["missing_fields_by_object"])
        row["required_objects"] = coverage["audited_objects"]
    if job["state"] == "FINISHED_VALID":
        seconds = row["cumulative_task_seconds"]
        if (not row.get("complete_coverage") or row["legacy_outcome"] != "correct"
                or not isinstance(seconds, (int, float)) or not isfinite(seconds)
                or not 0 < seconds <= row["deadline_seconds"]):
            raise ValueError("A saved success failed independent validation for " + row["logical_id"])
        row.update(success=True, restricted_seconds=seconds)
    return row


def _interval(values, seed, replicates):
    if len(values) < 2:
        return None
    generator = Random(seed)
    draws = sorted(mean(generator.choices(values, k=len(values))) for _ in range(replicates))
    return [draws[int(.025 * (replicates - 1))], draws[int(.975 * (replicates - 1))]]


def _pair_rows(rows, reference):
    cases = {}
    for row in rows:
        if row["state"] in RECORDED:
            cases.setdefault(row["case"], {})[row["method"]] = row
    reports = {}
    for arms in cases.values():
        if reference not in arms:
            continue
        target = arms[reference]
        for method, row in arms.items():
            if method != reference:
                difference = {"case": row["case"], "group": row["group"],
                              "success_difference": int(row["success"]) - int(target["success"]),
                              "restricted_seconds_difference": row["restricted_seconds"] - target["restricted_seconds"]}
                reports.setdefault(method, []).append(difference)
    result = {}
    for name, rows in reports.items():
        groups = {}
        for row in rows:
            groups.setdefault(row["group"], []).append(row)
        values = {key: [mean(row[key] for row in group) for group in groups.values()]
                  for key in ("success_difference", "restricted_seconds_difference")}
        result[name] = {"matched_cases": len(rows), "independent_groups": len(groups),
                        **{key: {"mean": mean(numbers), "cluster_bootstrap_95": _interval(numbers, 17, 2000)}
                           for key, numbers in values.items()}}
    return result


def rebuild(directory, output):
    catalog = _read(directory / "pilot_manifest.json")
    if not catalog or not catalog["cases"]:
        raise ValueError("A frozen nonempty pilot manifest is required")
    rows = [_arm_row(directory, case, method) for case in catalog["cases"] for method in case["order"]]
    groups = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    methods = {}
    for method, cases in groups.items():
        valid = [row for row in cases if row["state"] in RECORDED]
        methods[method] = {"planned": len(cases), "recorded": len(valid),
                           "successes": sum(row["success"] for row in valid),
                           "statuses": {state: sum(row["state"] == state for row in cases)
                                        for state in sorted({row["state"] for row in cases})},
                           "mean_restricted_seconds": mean(row["restricted_seconds"] for row in valid) if valid else None}
    reference = "conditional-async" if "conditional-async" in methods else "paired-async"
    result = {"mode": "NATURAL_LATENCY_DEVELOPMENT", "role": catalog["role"], "reference": reference,
              "planned_episodes": len(rows), "recorded_episodes": sum(r["state"] in RECORDED for r in rows),
              "methods": methods, "matched_differences": _pair_rows(rows, reference),
              "energy_measurement": None, "scientific_conclusion": "INCONCLUSIVE"}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf8") as handle:
        header = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", result)
    return result
