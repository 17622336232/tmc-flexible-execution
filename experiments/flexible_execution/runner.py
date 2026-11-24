"""Atomic per-episode records, resumable diagnostics, and offline re-scoring."""

import csv
import fcntl
import json
import subprocess
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from statistics import mean

from experiments.task_attack.runtime import write_json

from .audit import capture
from .evaluate import score
from .fixtures import MODE, diagnostics
from .simulation import METHODS, Episode


MECHANISM_SOURCES = ("__init__.py", "evidence.py", "evaluate.py", "executor.py", "fixtures.py",
                     "queries.py", "search.py", "simulation.py", "task_ir.py")
MECHANISM_FIELDS = ("profile", "mode", "fixture_sha256", "case_ids", "methods", "limits", "evaluation")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def _hash(data):
    return sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                             allow_nan=False).encode("utf8")).hexdigest()


def manifest(checkout, defaults):
    cases = diagnostics()
    if len(cases) != defaults["data"]["local_smoke_instances"]:
        raise ValueError("The mechanism fixture count does not match the local protocol")
    source_directory = Path(__file__).resolve().parent
    sources = [source_directory / filename for filename in MECHANISM_SOURCES]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
                            stdout=subprocess.PIPE, text=True, timeout=5).stdout.strip()
    methods = defaults["methods"]
    return {
        "protocol_version": 2,
        "profile": defaults["execution_profile"],
        "mode": MODE,
        "code_git_commit": commit,
        "source_sha256": {str(path.relative_to(checkout)): sha256(path.read_bytes()).hexdigest()
                          for path in sorted(sources)},
        "fixture_sha256": _hash([asdict(case) for case in cases]),
        "case_ids": [case.id for case in cases],
        "methods": list(METHODS),
        "limits": {"max_candidates": methods["candidate_cap"],
                   "max_prepare": methods["preparation_length_cap"],
                   "budget_ms": methods["initial_online_compute_budget_ms"]},
        "evaluation": "Fixed assumption cases for code checks. No external model calls or measured flight/energy.",
    }


def compatible_protocol(recorded, current):
    return (recorded.get("protocol_version") in {1, 2}
            and all(recorded.get(field) == current[field] for field in MECHANISM_FIELDS)
            and all(recorded.get("source_sha256", {}).get(path) == digest
                    for path, digest in current["source_sha256"].items()))


def _locked_protocol(directory, current):
    recorded = read_json(directory / "protocol.json")
    if not compatible_protocol(recorded, current):
        raise ValueError("The locked mechanism inputs changed. Choose a new --run-id.")
    if recorded != current:
        write_json(directory / "protocol_compatibility.json", {
            "original_protocol_sha256": _hash(recorded),
            "original_code_git_commit": recorded["code_git_commit"],
            "checked_code_git_commit": current["code_git_commit"],
            "verified_behavioral_keys": list(MECHANISM_FIELDS),
            "verified_source_sha256": current["source_sha256"],
        })
    return recorded


def _record_name(directory, case_id, method):
    return directory / "episodes" / f"{case_id}-{method}.json"


def _write_lines(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def collect(directory, protocol):
    cases = {case.id: case for case in diagnostics()}
    rows = []
    for case_id in protocol["case_ids"]:
        for method in protocol["methods"]:
            path = _record_name(directory, case_id, method)
            if not path.exists():
                continue
            record = read_json(path)
            if (record["case_id"], record["method"], record["mode"], record["protocol_sha256"]) != (
                case_id, method, protocol["mode"], _hash(protocol)
            ):
                raise ValueError("Episode belongs to a different protocol: " + path.name)
            if score(cases[case_id], record["events"]) != record["score"]:
                raise ValueError("Episode score differs from the event ledger: " + path.name)
            rows.append(record)
    return rows


def _metrics(entries):
    return {
        "episodes": len(entries),
        "successes": sum(item["success"] for item in entries),
        "wrong_accepted": sum(item["wrong_accepted"] for item in entries),
        "mean_restricted_seconds": mean(item["restricted_completion_seconds"] for item in entries)
        if entries else None,
        "mean_blocked_seconds": mean(item["blocked_seconds"] for item in entries) if entries else None,
    }


def summarize(directory, protocol, records):
    fields = (
        "case_id", "mechanism", "legacy_family", "method", "mode", "success", "wrong_accepted",
        "completion_seconds", "restricted_completion_seconds", "blocked_seconds",
        "proposal_wall_seconds", "observations", "actions_started_with_pending_query",
        "rejected_materials",
    )
    scored = [{**record, **record["score"]} for record in records]
    with (directory / "diagnostic_results.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    _write_lines(directory / "episodes.jsonl",
                 ({key: value for key, value in row.items() if key != "events"} for row in records))
    _write_lines(directory / "events.jsonl", (
        {"case_id": row["case_id"], "method": row["method"], "mode": row["mode"], **event}
        for row in records for event in row["events"]
    ))
    _write_lines(directory / "failure_cases.jsonl", (
        {key: value for key, value in row.items() if key != "events"}
        for row in records if not row["score"]["success"]
    ))
    methods = {method: [] for method in protocol["methods"]}
    mechanisms = {}
    for row in scored:
        methods[row["method"]].append(row)
        by_method = mechanisms.setdefault(row["mechanism"], {})
        by_method.setdefault(row["method"], []).append(row)
    stats = {
        "mode": protocol["mode"],
        "planned_episodes": len(protocol["case_ids"]) * len(protocol["methods"]),
        "completed_episodes": len(records),
        "methods": {method: _metrics(entries) for method, entries in methods.items()},
        "by_mechanism": {
            group: {method: _metrics(entries) for method, entries in by_method.items()}
            for group, by_method in mechanisms.items()
        },
    }
    write_json(directory / "statistics.json", stats)
    _write_notes(directory, stats)
    return stats


def _write_notes(directory, stats):
    lines = [
        "# 本地启动验证", "",
        "执行状态为 IN_PROGRESS。以下计时来自可重复的小规模机制诊断，动作与等待使用预定时间。",
        "提案计算用本机实际耗时，纳入诊断时间轴。该表用于排错。", "",
        "| 方法 | 正确交付 | 错误交付 | 平均受限完成时间（诊断秒） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, result in stats["methods"].items():
        seconds = result["mean_restricted_seconds"]
        formatted = f"{seconds:.4f}" if seconds is not None else "—"
        lines.append(f"| {method} | {result['successes']}/{result['episodes']} | "
                     f"{result['wrong_accepted']} | {formatted} |")
    lines.extend(("", "每回合检查必需对象和材料、判读绑定、许可、报告和终态。失败计冻结截止。",
                  "原巡检的 CPU 仿真传感器测试若已运行，保存在 legacy_cpu_sensor/。",
                  "它与本表分别记录。", ""))
    native = directory / "native_async_cpu" / "async_navigation_probe.json"
    if native.exists():
        probe = read_json(native)
        lines.extend((
            "## 原导航与后台查询的仿真接通", "",
            f"CPU 仿真中的原导航实测位移为 {probe['distance_m']:.3f} 米。",
            "保留了移动前后的机载图，查询在受控释放前绑定原图，独立移动和新图采集均已完成。",
            "查询的释放时间由测试设定，视觉服务自然延迟另测。", "",
        ))
    (directory / "result_summary_cn.md").write_text("\n".join(lines), encoding="utf8")
    (directory / "limitations_and_blockers.md").write_text(
        "# 实验边界与待续阶段\n\n"
        "当前候选搜索核验的共同后续是移动至规定终态。包含可返访中间终点的通用规划、"
        "学习排序器、计算预算联合调度和流式基线继续开发。\n\n"
        "这批回合使用程序构造的任务、结果和观察。真实取证验收、自然视觉延迟、"
        "ProcTHOR 视觉闭环、完整模型训练及物理飞行另行验收。\n\n"
        "audit.json 记录当前 NVIDIA 驱动接口、磁盘、新缓存门槛及相机源码路径。"
        "付费预算沿用任务书的零额度设置。\n",
        encoding="utf8",
    )
    (directory / "reproduction.md").write_text(
        "# 复现\n\n在本实验副本内运行。\n\n"
        "```bash\n"
        f"python3 -m experiments.flexible_execution.cli run --profile local5060 --resume --run-id {directory.name}\n"
        f"python3 -m experiments.flexible_execution.cli reproduce --run-id {directory.name}\n"
        "```\n\n"
        "reproduce 核对锁定源码和配置，逐回合重算原始事件。不同运行的 CPU 提案时间可有波动。\n\n"
        "原导航与后台查询的仿真检查可用新的输出目录重跑。\n\n"
        "```bash\n"
        "bash scripts/run_flexible_cpu_smoke.sh --output outputs/native-async-recheck\n"
        "```\n",
        encoding="utf8",
    )


def run(directory, checkout, defaults, resume):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = directory / "protocol.json"
        current = manifest(checkout, defaults)
        if protocol_path.exists():
            if not resume:
                raise ValueError("This run already exists. Choose --resume or a new --run-id.")
            current = _locked_protocol(directory, current)
        else:
            write_json(protocol_path, current)
        capture(directory, checkout, defaults)
        completed = {(row["case_id"], row["method"]) for row in collect(directory, current)}
        protocol_hash = _hash(current)
        for case in diagnostics():
            for method in current["methods"]:
                if (case.id, method) in completed:
                    continue
                episode = Episode(case, method, current["limits"])
                try:
                    events = episode.run()
                except Exception as error:
                    episode.emit("execution_error", type=type(error).__name__, message=str(error)[:500])
                    events = episode.events
                result = score(case, events)
                write_json(_record_name(directory, case.id, method), {
                    "case_id": case.id, "legacy_family": case.legacy_family, "mechanism": case.mechanism,
                    "method": method, "mode": MODE, "protocol_sha256": protocol_hash,
                    "events": events, "score": result,
                })
        return summarize(directory, current, collect(directory, current))


def reproduce(directory, current):
    with (directory / ".run.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recorded = _locked_protocol(directory, current)
        rows = collect(directory, recorded)
        total = len(recorded["case_ids"]) * len(recorded["methods"])
        if len(rows) != total:
            raise ValueError("The run is still incomplete. Resume before reproducing its results.")
        return summarize(directory, recorded, rows)
