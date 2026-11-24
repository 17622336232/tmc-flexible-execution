"""Read-only source and runtime inventory for the local profile."""

import csv
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

from experiments.task_attack import FAMILIES
from experiments.task_attack.runtime import write_json
from experiments.task_attack.tasks import SCENES, make_task


def _command(command, cwd):
    if not shutil.which(command[0]):
        return None
    try:
        result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _structure(family, spec):
    if spec.get("mode") == "current_literal_labels":
        return {"scope": "all", "report": "literal_read"}
    if family == "thermal_minimum":
        return {"scope": "all", "aggregate": "argmin", "report_fields": 1}
    operators = {
        "predicate": "eq", "greater": "gt", "less_equal": "le",
        "unequal": "neq", "less_than_fields": "field_lt", "greater_than_fields": "field_gt",
    }
    filters = []
    for key, name in operators.items():
        formula = spec.get(key)
        if isinstance(formula, dict):
            filters.extend((name, type(value).__name__) for value in formula.values())
        elif formula:
            filters.append((name, "field"))
    return {"scope": "specified_object" if "target" in spec else "all",
            "and": sorted(filters), "report_fields": len(spec["required"])}


def inventory(directory):
    rows = []
    for family in FAMILIES:
        scene = SCENES[family]
        devices = [
            {"identity": "sample-A", "temperature": 34, "order_count": 2},
            {"identity": "sample-B", "temperature": 81, "order_count": 3},
        ]
        _, spec, _ = make_task(family, devices, random.Random(17))
        canonical = json.dumps(_structure(family, spec), sort_keys=True)
        rows.append({"family": family, "legacy_domain": scene,
                     "role": "literal_read_control" if family.endswith("_labels") else "development",
                     "structure_hash": sha256(canonical.encode("utf8")).hexdigest(),
                     "normalized_report_structure": canonical})
    with (directory / "task_inventory.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    main = [row for row in rows if row["role"] == "development"]
    return {
        "legacy_main_families": len(main),
        "legacy_literal_controls": len(rows) - len(main),
        "main_families_by_domain": dict(Counter(row["legacy_domain"] for row in main)),
        "distinct_legacy_report_structure_hashes": len({row["structure_hash"] for row in main}),
        "meaning": "Canonicalized legacy report-contract shapes. Layouts, full dependency graphs and episodes are different counts.",
    }


def capture(directory, checkout, config):
    directory.mkdir(parents=True, exist_ok=True)
    workspace = Path.home() / "uav"
    sources = [workspace / name for name in
               ("Diff-VLA", "Embodied-intelligent-behavior", "Diff-planner-baseline", "nv/Elastic-Tracker")]
    repositories = {}
    for path in [checkout] + sources:
        repositories[str(path)] = {
            "present": path.is_dir(),
            "branch": _command(["git", "rev-parse", "--abbrev-ref", "HEAD"], path) if path.is_dir() else None,
            "commit": _command(["git", "rev-parse", "HEAD"], path) if path.is_dir() else None,
        }
    legacy = sources[1]
    gpu = _command([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ], checkout)
    pci = _command(["lspci", "-nn"], checkout) or ""
    devices = [line for line in pci.splitlines()
               if "NVIDIA" in line and ("VGA" in line or "3D controller" in line)]
    available = shutil.disk_usage(directory).free / (1024 ** 3)
    minimum = config["resources"]["minimum_free_disk_gb"]
    limits = {
        "python": sys.version.split()[0],
        "legacy_python": _command([str(legacy / ".venv/bin/python3"), "--version"], checkout),
        "ros_noetic_setup_present": Path("/opt/ros/noetic/setup.bash").is_file(),
        "legacy_ros_build_present": (legacy / "sim_ws/devel/setup.bash").is_file(),
        "nvidia_pci": devices,
        "nvidia_smi": gpu,
        "disk_free_gib": round(available, 2),
        "new_cache_minimum_free_gib": minimum,
        "new_model_or_dataset_cache_allowed": available >= minimum,
        "paid_external_api_budget": config["resources"]["paid_external_api_budget"],
        "paid_compute_budget": config["resources"]["paid_compute_budget"],
        "tracker_source_present": sources[-1].is_dir(),
        "legacy_default_model_profile_present": (legacy / ".runtime/chatgpt/default.json").is_file(),
        "full_project_default_model_profile_present": (sources[0] / ".runtime/chatgpt/default.json").is_file(),
    }
    facts = {"repositories": repositories, "resources": limits, "inventory": inventory(directory)}
    write_json(directory / "audit.json", facts)
    summary = directory / "repo_audit.md"
    if not summary.exists():
        summary.write_text(
            "# 仓库审计\n\n"
            "当前入口使用独立 Git 副本。分支、提交、路径和环境实测见 audit.json。\n\n"
            "| 现有接口 | 行为与接入位置 |\n"
            "| --- | --- |\n"
            "| experiments/task_attack/tasks.py · make_task | 生成指令和私有评分条件 |\n"
            "| experiments/task_attack/scene.py · build_scene | 生成模拟场景和事实 |\n"
            "| experiments/task_attack/agent.py · run_agent | 同步等待一轮看图与动作决定 |\n"
            "| experiments/task_attack/runtime.py · observe / move / approach | 原相机与导航接口 |\n"
            "| experiments/task_attack/evaluate.py · evaluate | 核对报告事实，覆盖另由新评分核对 |\n"
            "| experiments/task_attack/batch.py · main | 配对批跑和记录 |\n\n"
            "task_inventory.csv 按题型和规范化报告结构盘点。合成回合和真实观测分别计数。\n",
            encoding="utf8",
        )
    return facts
