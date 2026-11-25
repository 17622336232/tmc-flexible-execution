"""Evaluate only actions and releases supported by registered observations."""

from heapq import heappop, heappush
from pathlib import Path
from statistics import mean
from time import process_time

from experiments.task_attack.runtime import write_json

from .learning_schedule import build_config
from .ranker_replay import replay
from .runner import read_json
from .timeline import read_trace_rows


def _trace(directory):
    """Reveal recorded query outcomes only at their actual application event."""
    observations = {}
    attempts, supported, unbound = 0, 0, 0
    for status in sorted(directory.glob("*/*/job.json")):
        job = read_json(status)
        path = Path(job["artifact_dir"]) / "private/robot/timeline.jsonl" if job.get("artifact_dir") else None
        if path is None or not path.is_file():
            continue
        events, _ = read_trace_rows(path, allow_truncated=True)
        queue, pending, results = [], {}, []
        usable = {"query_submitted", "query_applied", "query_rejected", "candidate_executed", "evidence_accepted"}
        for number, event in enumerate(events):
            if event["event"] in usable:
                heappush(queue, (event["monotonic_seconds"], number, event))
        while queue:
            time, _, event = heappop(queue)
            kind, binding = event["event"], event.get("binding") or {}
            key = binding.get("id")
            if kind == "query_submitted" and key:
                pending[key] = (time, binding)
                attempts += 1
            elif kind == "query_applied":
                if key not in pending or pending[key][1] != binding:
                    unbound += 1
                    continue
                started, _ = pending.pop(key)
                results.append({"query": key, "released_at": time, "elapsed_seconds": time - started,
                                "outcome": event["outcome"], "input_sha256": binding.get("input_hash"),
                                "source": "EXACT_RECORDED_QUERY"})
                supported += 1
        observations[job["id"]] = {"source": str(path), "released": results,
                                   "unsupported_unresolved_queries": len(pending),
                                   "unrecorded_action_or_view": "OUT_OF_SUPPORT"}
    return {"evidence_level": "TRACE_SUPPORTED", "virtual_clock": "recorded_monotonic_event_order",
            "attempted_queries": attempts, "exact_bound_releases": supported, "unbound_releases": unbound,
            "counterfactual_full_episodes": 0, "new_requests": 0, "source_episodes": observations}


def offline(output_root):
    root = output_root / "acceleration-v2"
    root.mkdir(parents=True, exist_ok=True)
    diagnostic = output_root / "ranker-diagnostic-v1"
    native = output_root / "natural-pilot/native-training-check-v2"
    configurations = {f"{size}-seed-{seed}-cap-{cap}": build_config(diagnostic, size, seed, cap, include_retrieval=True)
                      for size in (1000, 3000, 10000) for seed in (17, 29, 43) for cap in (4, 8)}
    cpu_start = process_time()
    fixed = replay(native, root / "offline-configuration-replay.json", configurations=configurations)
    grid = {}
    for key in fixed["snapshots"][0]["methods"]:
        rows = [row["methods"][key] for row in fixed["snapshots"]]
        grid[key] = {"checked": sum(row["checked"] for row in rows),
                     "selected": sum(bool(row["selected"]) for row in rows),
                     "mean_forecast_seconds": mean(row["predicted_finish_seconds"] for row in rows),
                     "cpu_seconds": sum(row["process_cpu_seconds"] for row in rows)}
    traces = _trace(native)
    write_json(root / "trace-replay.json", traces)
    symbol = read_json(diagnostic / "latest_summary.json")
    scale = {}
    for row in symbol["run_rows"]:
        if row["objective"] == "cost":
            cost = next(item["mean_restricted_seconds"] for item in row["fixed"] if item["cap"] == 4)
            scale.setdefault(row["training_states"], []).append(cost)
    symbolic = {size: mean(scores) for size, scores in scale.items() if len(scores) == 3}
    size = min(symbolic, key=lambda item: (symbolic[item], item))
    # Seed is fixed in advance. Compare the deployable B1 check cap on the same pools.
    cap = min((4, 8), key=lambda item: (
        grid[f"{size}-seed-17-cap-{item}/conditional-rank-b1"]["mean_forecast_seconds"], item))
    result = {"protocol_version": "protocol-v2", "evidence_level": "DECISION_REPLAY", "actual_new_model_requests": 0,
              "fixed_pools": fixed["source_snapshots"], "independent_sources": fixed["independent_case_groups"],
              "settings": len(configurations), "paired_policy_settings": len(grid), "grid": grid,
              "training_size": size, "training_seed": 17, "check_cap": cap,
              "symbolic_restricted_seconds": symbolic, "trace_bound_releases": traces["exact_bound_releases"],
              "cpu_seconds": process_time() - cpu_start, "full_closed_loop_claim_supported": False}
    write_json(root / "offline_selection.json", result)
    return result
