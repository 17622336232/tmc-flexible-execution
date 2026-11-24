#!/usr/bin/env bash
set -eo pipefail

flexible_checkout="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
flexible_legacy_root="${VLA_LEGACY_ROOT:-$HOME/uav/Embodied-intelligent-behavior}"

flexible_args=("$@")
set --
source /opt/ros/noetic/setup.bash
source "$flexible_legacy_root/sim_ws/devel/setup.bash"
if [[ -f "$flexible_legacy_root/.venv/bin/activate" ]]; then
    source "$flexible_legacy_root/.venv/bin/activate"
fi
export PYTHONPATH="$flexible_checkout/bridge:$flexible_checkout:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VLA_MODE=gazebo
cd "$flexible_checkout"
exec python3 -m experiments.flexible_execution.adapters.clean_run "${flexible_args[@]}"
