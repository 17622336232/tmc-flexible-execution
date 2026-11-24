#!/usr/bin/env bash
# Source this file from a launch script, not from ~/.bashrc.
set -eo pipefail
vla_source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Catkin forwards positional arguments to _setup_util.py. In particular, a
# caller's --help must not turn the generated environment script into usage text.
vla_environment_arguments=("$@")
set --
if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo '未找到 ROS Noetic。请先按 README 安装依赖。' >&2
  return 2 2>/dev/null || exit 2
fi
source /opt/ros/noetic/setup.bash
if [[ -f "$vla_source_root/sim_ws/devel/setup.bash" ]]; then
  source "$vla_source_root/sim_ws/devel/setup.bash"
fi
if [[ -f "$vla_source_root/.venv/bin/activate" ]]; then
  source "$vla_source_root/.venv/bin/activate"
fi
export PYTHONPATH="$vla_source_root/research/attack_learning:$vla_source_root/bridge:$vla_source_root:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export VLA_MODE=gazebo
if [[ -f "$vla_source_root/config/local.env" ]]; then
  source "$vla_source_root/config/local.env"
fi
cd "$vla_source_root"
set -- "${vla_environment_arguments[@]}"
unset vla_environment_arguments
