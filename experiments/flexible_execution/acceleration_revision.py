"""Register the second v2 worker and migrate only the remaining budget."""

import fcntl
from hashlib import sha256
from pathlib import Path
from subprocess import check_output

from .control import REQUEST_LIMIT, TASK_TERMINAL, _runtime_inputs, control_status, plans
from .runner import _hash, read_json


REVISION = "evidence-handoff-r1"
PORTS = (11674, 11675)


def selected_root(output_root):
    revised = output_root / "protocol-v2-optimization"
    registration = revised / "execution_revision.json"
    if registration.is_file():
        saved = read_json(registration)
        catalog = read_json(revised / "development/pilot_manifest.json")
        if saved["development_catalog_sha256"] != _hash(catalog):
            raise ValueError("The selected execution revision does not match the registered development catalog")
        return revised
    return output_root / "acceleration-v2"


def _read_bytes(path):
    return path.read_bytes() if path.is_file() else b""


def _parent_budget(template, job, control):
    parent = Path(job.get("request_ledger") or control / "requests" / (template["id"] + ".jsonl"))
    data = _read_bytes(parent)
    requests = len(data.splitlines())
    if requests != (job.get("request_budget") or {}).get("reserved", 0):
        raise ValueError("Original send count does not match the migrated budget")
    return {"original_job_id": template["id"], "original_ledger": str(parent),
            "original_ledger_sha256": sha256(data).hexdigest(), "original_state": job["state"],
            "prior_requests": requests, "prior_attempts": job["attempt"],
            "prior_charged_seconds": job.get("charged_seconds", 0.)}


def initialize_budget(job, ledger):
    parent = job.get("budget_parent")
    if not parent or job["budget_class"] != "original_development":
        return
    data = _read_bytes(Path(parent["original_ledger"]))
    if sha256(data).hexdigest() != parent["original_ledger_sha256"]:
        raise ValueError("The original budget ledger changed after migration")
    if not ledger.is_file() and data:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        staged = ledger.with_suffix(".migration-tmp")
        staged.write_bytes(data)
        staged.replace(ledger)
    if not _read_bytes(ledger).startswith(data):
        raise ValueError("The migrated request ledger does not inherit the original dispatch prefix")


def _committed_inputs(checkout, catalog, auth_project):
    runtime = _runtime_inputs(checkout, catalog, auth_project, PORTS)
    revision = check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    for name, digest in runtime["source_sha256"].items():
        committed = check_output(["git", "show", revision + ":" + name], cwd=checkout)
        if sha256(committed).hexdigest() != digest:
            raise ValueError("Commit the new worker input before registering it: " + name)
    return {**runtime, "code_git_commit": revision}


def _verify_registration(output_root, checkout, auth_project):
    root = output_root / "protocol-v2-optimization"
    registry = read_json(root / "execution_revision.json")
    development = root / "development"
    catalog = read_json(development / "pilot_manifest.json")
    protocol = read_json(development / "run_protocol.json")
    if (registry["execution_revision"] != catalog["execution_revision"] or
            registry["execution_revision"] != protocol["execution_revision"] or
            registry["development_catalog_sha256"] != _hash(catalog) or
            registry["development_protocol_sha256"] != _hash(protocol) or
            protocol["catalog_sha256"] != _hash(catalog) or protocol["auth_project"] != str(auth_project)):
        raise ValueError("The selected execution revision is not the worker's frozen protocol")
    for name, digest in protocol["source_sha256"].items():
        contents = check_output(["git", "show", protocol["code_git_commit"] + ":" + name], cwd=checkout)
        if sha256(contents).hexdigest() != digest:
            raise ValueError("The selected worker archive differs from the preflight: " + name)
    for path, digest in protocol["environment_sha256"].items():
        if sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("The selected simulator environment differs from the preflight: " + path)
    for key, parent in catalog["budget_parents"].items():
        if sha256(_read_bytes(Path(parent["original_ledger"]))).hexdigest() != parent["original_ledger_sha256"]:
            raise ValueError("The original budget changed after the verified migration: " + key)
    return registry


def register(output_root, checkout, auth_project):
    from .acceleration import lock
    from .adapters.live_inspection import ACCEPT_TASK, READ_MATERIAL
    from .adapters.v2_execution import COMPILER
    from .gpt_vision import JUDGE, LOCATE

    root = output_root / "protocol-v2-optimization"
    original = output_root / "acceleration-v2"
    auth_project = auth_project.resolve()
    if (root / "execution_revision.json").is_file():
        return _verify_registration(output_root, checkout, auth_project)
    with (output_root / "control/.simulator.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "execution_revision.json").is_file():
            return _verify_registration(output_root, checkout, auth_project)
        for control in (original / "control", output_root / "control", root / "control"):
            if control_status(control)["active_jobs"]:
                raise ValueError("Stop the active simulator before registering a new execution revision")
        audit = read_json(root / "audit/current_v2_timeline.json")
        old = original / "development"
        catalog, protocol = [read_json(old / name) for name in ("pilot_manifest.json", "run_protocol.json")]
        memory, repair, inherited = {}, {}, {}
        for template in plans(old, protocol, catalog):
            path = Path(template["arm"]) / "job.json"
            state = read_json(path) if path.is_file() else template
            if state["id"] != template["id"] or state["state"] not in TASK_TERMINAL | {"PENDING", "CANCELLED_BY_USER"}:
                raise ValueError("Original dev job is not safely migratable: " + template["id"])
            key = "/".join((template["case"]["id"], template["method"]))
            parent = _parent_budget(template, state, original / "control")
            memory[key] = parent
            if state["state"] in TASK_TERMINAL:
                repair[key] = parent
            else:
                inherited[key] = parent
        original_count = sum(row["prior_requests"] for row in memory.values())
        first_ceiling = 288
        repair_ceiling = 144
        maximum_migration = sum(REQUEST_LIMIT - row["prior_requests"] for row in inherited.values())
        if (audit["actual_new_model_requests"] != 0 or audit["dispatches"] != original_count
                or audit["source_code_git_commit"] != protocol["code_git_commit"] or len(repair) > 6
                or original_count + maximum_migration > first_ceiling or len(repair) * REQUEST_LIMIT > repair_ceiling
                or not inherited or not repair):
            raise ValueError("The observed development and repair allocation does not match the original limits")
        revised = {**catalog, "execution_revision": REVISION, "budget_parents": memory,
                   "budget_classes": {**{key: "repair_round_1" for key in repair},
                                      **{key: "original_development" for key in inherited}},
                   "parent_catalog_sha256": _hash(catalog)}
        runtime = _committed_inputs(checkout, revised, auth_project)
        development = root / "development"
        registry = {"state": "REGISTERED", "execution_revision": REVISION, "protocol_version": "protocol-v2",
                    "code_git_commit": runtime["code_git_commit"], "source_sha256": _hash(runtime["source_sha256"]),
                    "prompt_sha256": {k: sha256(v.encode("utf8")).hexdigest() for k, v in {
                        "compiler": COMPILER, "locator": LOCATE, "judgment": JUDGE,
                        "material_reader": READ_MATERIAL, "whole_task_acceptance": ACCEPT_TASK}.items()},
                    "model": runtime["model"], "reasoning_effort": runtime["reasoning_effort"],
                    "development_directory": str(development), "development_catalog_sha256": _hash(revised),
                    "development_protocol_sha256": _hash(runtime), "parent_development_directory": str(old),
                    "parent_protocol_sha256": _hash(protocol), "budget_parents": memory,
                    "first_development": {"original_dispatches": original_count, "migrated_jobs": len(inherited),
                                          "new_dispatch_ceiling": maximum_migration, "total_limit": first_ceiling},
                    "repair_round_1": {"jobs": len(repair), "dispatch_ceiling": len(repair) * REQUEST_LIMIT,
                                       "reserved_limit": repair_ceiling},
                    "repair_round_2": {"registered_jobs": 0, "maximum_jobs": 6, "maximum_dispatches": 144},
                    "formal_freeze_will_use": {"development": str(development), "catalog_sha256": _hash(revised),
                        "holdout_ids": str(original / "blind_online_subset.json"),
                        "offline_selection": str(original / "offline_selection.json")},
                    "actual_new_model_requests": 0}
        lock(development / "calibrated_costs.json", read_json(old / "calibrated_costs.json"))
        lock(development / "pilot_manifest.json", revised)
        lock(development / "run_protocol.json", runtime)
        lock(old / "execution_migrated.json", {"successor": str(development), "revision": REVISION})
        lock(root / "execution_revision.json", registry)
        return _verify_registration(output_root, checkout, auth_project)
