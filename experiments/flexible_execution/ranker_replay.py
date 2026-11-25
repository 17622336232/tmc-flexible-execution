"""Compare ordering from fixed candidate pools derived from real camera inputs."""

from pathlib import Path
from time import process_time

from experiments.task_attack.runtime import write_json

from .candidate_ranking import RankedPolicy, candidate_prefixes, select_ranked
from .executor import Snapshot
from .learning_schedule import ROLE, registered_config
from .live_audit import _camera_frames, _rows, _verified_judgment
from .live_planning import checked
from .runner import _hash, read_json
from .task_ir import Condition, QuerySpec, Requirement, Skill, TaskContract


def _condition(raw):
    return Condition(raw["query_id"], tuple(raw["outcomes"])) if raw else None


def _contract(raw):
    requirements = tuple(Requirement(**{**row, "when": _condition(row["when"])}) for row in raw["requirements"])
    skills = tuple(Skill(**{**row, "covers": tuple(row["covers"]), "permission": _condition(row["permission"])})
                   for row in raw["skills"])
    queries = tuple(QuerySpec(**{**row, "outcomes": tuple(row["outcomes"])}) for row in raw["queries"])
    return TaskContract(**{**raw, "requirements": requirements, "skills": skills, "queries": queries,
                           "travel": tuple(tuple(row) for row in raw["travel"])})


def replay(directory, output, *, configurations=None):
    from .control import TASK_TERMINAL

    catalog = read_json(directory / "pilot_manifest.json")
    if catalog["role"] != ROLE or catalog.get("data_role") != "INDEPENDENT_NATIVE_TRAINING_CHECK":
        raise ValueError("Fixed candidate replay requires the independent native training cohort")
    protocol = read_json(directory / "run_protocol.json")
    if protocol["catalog_sha256"] != _hash(catalog):
        raise ValueError("Native trajectory inputs differ from the runtime protocol")
    config = registered_config(directory, catalog)
    configurations = configurations or {"recorded": config}
    policies = {(label, name): RankedPolicy.load(row, name) for label, settings in configurations.items()
                for name, row in settings["policies"].items()}
    rows, skipped = [], []
    for case in catalog["cases"]:
        for method in case["order"]:
            status = directory / case["id"] / method / "job.json"
            if not status.is_file():
                continue
            job = read_json(status)
            if job["state"] not in TASK_TERMINAL or not job.get("artifact_dir"):
                continue
            if (job["case"] != case or job["method"] != method or job["model"] != protocol["model"]
                    or job["reasoning_effort"] != protocol["reasoning_effort"]):
                raise ValueError("A native trajectory does not match its registered trial")
            robot = Path(job["artifact_dir"]) / "private/robot"
            contract_path = robot / "conditional_contract.json"
            events = _rows(robot / "timeline.jsonl")
            applied, _ = _verified_judgment(events, _rows(robot / "model_calls.jsonl"), _camera_frames(robot))
            if not applied or not contract_path.is_file():
                skipped.append({"case": case["id"], "source_method": method, "reason": "no_bound_visual_frontend"})
                continue
            binding = applied["binding"]
            raw_contract = read_json(contract_path)["task"]
            task = _contract(raw_contract)
            state = Snapshot(binding["submitted_at"], task.start, {}, {}, {})
            pool = candidate_prefixes(task, state, 3, 256, float("inf"))
            methods = {}
            for (label, name), policy in policies.items():
                stats = {}

                def record(event, **fields):
                    if event == "candidate_search_summary":
                        stats.update(fields)

                started_cpu = process_time()
                decision = select_ranked(task, state, checked, record, ranker=policy.ranker,
                                         check_cap=policy.check_cap, budget_ms=1000, max_prepare=3,
                                         minimum_net_gain=policy.minimum_net_gain, fixed_prefixes=pool)
                identifier = name if label == "recorded" else label + "/" + name
                methods[identifier] = {**stats,
                    "process_cpu_seconds": process_time() - started_cpu,
                    "selected": [action.skill.id for action in decision.prepare] if decision else [],
                    "predicted_finish_seconds": decision.expected_finish if decision else stats["baseline_finish"]}
            if len({row["candidate_prefixes_sha256"] for row in methods.values()}) != 1:
                raise ValueError("The identical candidate pool could not be scored within every policy's wall budget")
            rows.append({"case": case["id"], "group": case["group_id"], "source_method": method,
                         "bound_original_query_sha256": binding["input_hash"], "contract_sha256": _hash(raw_contract),
                         "pending_at_seconds": state.at, "fixed_candidates": len(pool), "methods": methods})
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"mode": "OFFLINE_FIXED_POOLS_FROM_ACTUAL_VISUAL_FRONTENDS", "actual_model_requests": 0,
              "evidence_level": "DECISION_REPLAY",
              "new_vision_observations": 0, "source_snapshots": len(rows),
              "catalog_sha256": protocol["catalog_sha256"], "runtime_protocol_sha256": _hash(protocol),
              "independent_case_groups": len({row["group"] for row in rows}), "snapshots": rows, "skipped": skipped}
    write_json(output, result)
    return result
