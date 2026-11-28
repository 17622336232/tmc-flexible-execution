#!/usr/bin/env bash
# Rebuild tables/figures from recorded evidence. Does not start a model or simulator.
set -euo pipefail
tmc_revision_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$tmc_revision_root"
tmc_revision_python="${TMC_PYTHON:-${VLA_LEGACY_ROOT:-$HOME/uav/Embodied-intelligent-behavior}/.venv/bin/python}"
if [[ ! -x "$tmc_revision_python" ]]; then
    tmc_revision_python=python3
fi
export PYTHONPATH="$tmc_revision_root/bridge:$tmc_revision_root/research/attack_learning:$tmc_revision_root${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

"$tmc_revision_python" -m experiments.flexible_execution.revision_audit
"$tmc_revision_python" scripts/revision_raw_cost.py
"$tmc_revision_python" scripts/revision_validation.py
"$tmc_revision_python" scripts/revision_design_coverage.py
"$tmc_revision_python" scripts/revision_camera.py
"$tmc_revision_python" scripts/revision_scoring.py
"$tmc_revision_python" scripts/revision_text_validation.py
"$tmc_revision_python" scripts/revision_split.py
"$tmc_revision_python" scripts/revision_examples.py
"$tmc_revision_python" scripts/revision_online_mechanism.py
"$tmc_revision_python" scripts/revision_report.py
"$tmc_revision_python" scripts/revision_provenance.py
"$tmc_revision_python" scripts/revision_integrity.py
"$tmc_revision_python" scripts/revision_draft_audit.py
"$tmc_revision_python" scripts/revision_decision_report.py
