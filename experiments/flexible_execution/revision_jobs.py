"""Resume the preregistered CPU-only development block, one fresh arm at a time."""
import argparse
import fcntl
import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path

from .revision_audit import OUT, read, save


def source_id():
    root = Path("experiments/flexible_execution")
    files = sorted(root.glob("*.py")) + sorted((root / "adapters").glob("*.py"))
    files += [Path("experiments/task_attack") / name for name in ("agent.py", "model.py", "records.py", "runtime.py")]
    return sha256(b"".join(str(p).encode() + b"\0" + p.read_bytes() for p in files)).hexdigest()


def jobs(config):
    for c in config["cases"]:
        for arm in c["order"]:
            directory = OUT / c["id"] / arm
            args = ["bash", "scripts/run_flexible_clean.sh", "--method", arm, "--backend", config["backend"],
                    "--family", c["family"], "--seed", str(c["seed"]), "--layout-seed", str(c["layout_seed"]),
                    "--source-profile", c["source_profile"], "--equipment-count", str(config["equipment_count"]),
                    "--episode-timeout", str(config["timeout_s"]), "--max-request-attempts", str(config["request_limit"]),
                    "--onboard-evidence-only", "--ros-port", "11774", "--gazebo-port", "11775",
                    "--chatgpt-project", "/home/tdszy/uav/Diff-VLA", "--output", str(directory)]
            if c.get("early_fields"):
                args += ["--early-fields", c["early_fields"]]
            if arm != "original-gpt":
                args += ["--cost-config", config["calibration"], "--protocol-v2", "--compile-task-program",
                         "--handoff-at-observation", "--execution-revision", "revision-development-1"]
            yield c, arm, directory, args


def observed(directory):
    result = directory / "result.json"
    coverage = directory / "clean_coverage.json"
    ledger = directory / "request_attempts.jsonl"
    sent = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    clock = directory / "job_clock.json"
    elapsed = read(clock).get("elapsed_seconds") if clock.exists() else None
    if result.exists() and coverage.exists() and elapsed is not None:
        d = read(result)
        return dict(state="scored", strict=read(coverage)["complete"], old=d["score"]["outcome"],
                    seconds=elapsed, sends=sent, result_sha256=sha256(result.read_bytes()).hexdigest())
    return dict(state="unfinished", seconds=elapsed, sends=sent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = read(Path("config/revision_development.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "development.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case, arm, directory, command in jobs(config):
            receipt = directory.parent / (arm + ".receipt.json")
            row = observed(directory)
            print(case["id"], arm, row, flush=True)
            if not args.run:
                continue
            if receipt.exists():
                recorded = read(receipt)
                if recorded["state"] == "endpoint-blocked":
                    raise SystemExit("Endpoint failure recorded; diagnose before dispatch")
                if recorded["state"] in ("scored", "closed-failure"):
                    continue
            if row["state"] == "scored":
                save(receipt, dict(**row, registered_case=case["id"], method=arm, provenance="existing completed arm"))
                continue
            if directory.exists():
                raise SystemExit("Unclosed existing arm. Inspect its worker and raw logs before recovery: " + str(directory))
            directory.parent.mkdir(parents=True, exist_ok=True)
            metadata = dict(case=case["id"], method=arm, controller_pid=os.getpid(), source_sha256=source_id(),
                            config_sha256=sha256(Path("config/revision_development.json").read_bytes()).hexdigest(),
                            command=command)
            save(receipt, dict(**metadata, state="starting"))
            with (directory.parent / (arm + ".console.log")).open("x") as log:
                status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            row = observed(directory)
            if row["state"] != "scored":
                calls = directory / "private/robot/model_calls.jsonl"
                replies = [json.loads(l) for l in calls.read_text().splitlines()] if calls.exists() else []
                if status != 0 and (not replies or not any("output" in c for c in replies)):
                    row["state"] = "endpoint-blocked"
                else:
                    row["state"] = "closed-failure"
            save(receipt, dict(**row, **metadata, exit_code=status))
            if row["state"] == "endpoint-blocked":
                raise SystemExit("Preflight or first request failed. Examine the recorded console and requests.")


if __name__ == "__main__":
    main()
