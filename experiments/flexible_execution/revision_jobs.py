"""Resume the preregistered CPU-only development block, one fresh arm at a time."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from hashlib import sha256
from pathlib import Path

from .revision_audit import OUT, read, save


def source_id():
    root = Path("experiments/flexible_execution")
    files = sorted(root.glob("*.py")) + sorted((root / "adapters").glob("*.py"))
    files += [Path("experiments/task_attack") / name for name in ("agent.py", "model.py", "records.py", "runtime.py")]
    return sha256(b"".join(str(p).encode() + b"\0" + p.read_bytes() for p in files)).hexdigest()


def jobs(config, out=OUT):
    for c in config["cases"]:
        for arm in c["order"]:
            directory = out / c["id"] / arm
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
                         "--handoff-at-observation", "--execution-revision", config.get("revision", "revision-development-1")]
                if config.get("bounded_text"):
                    args.append("--bounded-text-compiler")
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


def worker_active(directory):
    """The scheduler lock can outlive neither a crash nor an orphaned worker."""
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = entry.read_bytes().decode("utf8", errors="replace").split("\0")
            if "--output" not in argv or not any("flexible_execution.adapters.clean_run" in arg for arg in argv):
                continue
            target = Path(argv[argv.index("--output") + 1])
            if ((entry.parent / "cwd").resolve() / target).resolve() == directory.resolve():
                return True
        except (OSError, IndexError):
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--round3", action="store_true")
    args = parser.parse_args()
    config_path = Path("config/revision_development.json")
    config = read(config_path)
    out = OUT
    if args.round3:
        config_path = Path("config/revision_round3.json")
        study = read(config_path)
        case = next(c for c in config["cases"] if c["id"] == study["case"])
        assert all(case[k] == study[k] for k in ("seed", "layout_seed", "source_profile", "early_fields"))
        assert config["timeout_s"] == study["timeout_s"] and config["request_limit"] == study["request_limit"]
        config = dict(config, cases=[dict(case, order=study["online_order"])], bounded_text=True,
                      revision="revision-development-3")
        out = OUT / "round3"
        if args.run and not (out / "text_validation.json").is_file():
            raise SystemExit("Run the registered text validation before dispatch")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "development.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case, arm, directory, command in jobs(config, out):
            receipt = directory.parent / (arm + ".receipt.json")
            row = observed(directory)
            existing = read(receipt) if receipt.exists() else {}
            if existing.get("state") == "scored":
                if row["state"] != "scored" or any(existing.get(k) != row[k] for k in
                        ("result_sha256", "sends", "strict", "seconds")):
                    raise ValueError("Completed receipt differs from original evidence: " + str(directory))
            display = dict(row)
            if row["state"] == "unfinished":
                display["state"] = existing.get("state", "unregistered-start" if directory.exists() else "queued")
                if display["state"] == "starting" and worker_active(directory):
                    display["state"] = "running"
            print(case["id"], arm, display, flush=True)
            if not args.run:
                continue
            if existing:
                recorded = existing
                if recorded["state"] == "endpoint-blocked":
                    raise SystemExit("Endpoint failure recorded; diagnose before dispatch")
                if recorded["state"] in ("scored", "closed-failure", "interrupted-after-dispatch"):
                    continue
            if row["state"] == "scored":
                save(receipt, dict(**row, registered_case=case["id"], method=arm, provenance="existing completed arm"))
                continue
            if directory.exists():
                while worker_active(directory):
                    time.sleep(3)
                row = observed(directory)
                if row["state"] == "scored":
                    prior = read(receipt) if receipt.exists() else {}
                    save(receipt, {**prior, **row, "case":case["id"], "method":arm})
                    continue
                if row["sends"]:
                    prior = read(receipt) if receipt.exists() else {}
                    save(receipt, {**prior, **row, "state":"interrupted-after-dispatch",
                                   "case":case["id"], "method":arm})
                    continue
                archive = directory.with_name(directory.name + ".before-first-send")
                if archive.exists():
                    raise SystemExit("Repeated preflight interruption; preserved both raw directories")
                directory.rename(archive)
                console = directory.parent / (arm + ".console.log")
                if console.exists():
                    console.rename(archive / "console.log")
            directory.parent.mkdir(parents=True, exist_ok=True)
            metadata = dict(case=case["id"], method=arm, controller_pid=os.getpid(), source_sha256=source_id(),
                            config_sha256=sha256(config_path.read_bytes()).hexdigest(),
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
