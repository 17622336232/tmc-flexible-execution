#!/usr/bin/env bash
set -eo pipefail
vla_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo '需要 ROS Noetic，请先运行 scripts/check_environment.sh。' >&2
  exit 2
fi
source /opt/ros/noetic/setup.bash
set -u
mkdir -p "$vla_root/sim_ws/src"
# The real flight source stays untouched; only the required packages enter this workspace.
for vla_rel in Utils/quadrotor_msgs Utils/pose_utils Utils/uav_utils diff_planner/path_searching diff_planner/traj_utils diff_planner/traj_opt diff_planner/plan_manage; do
  vla_name="${vla_rel##*/}"
  vla_link="$vla_root/sim_ws/src/$vla_name"
  if [[ ! -e "$vla_link" && ! -L "$vla_link" ]]; then
    ln -s "../../Diff-planner/src/$vla_rel" "$vla_link"
  fi
done
# A tiny plan_env copy exposes local-map bounds consistently to every consumer.
# Keeping one shared header avoids class-definition differences across libraries.
if [[ ! -e "$vla_root/sim_ws/src/plan_env" && ! -L "$vla_root/sim_ws/src/plan_env" ]]; then
  ln -s ../overrides/plan_env "$vla_root/sim_ws/src/plan_env"
fi
if [[ ! -e "$vla_root/sim_ws/src/bridge_ros" ]]; then
  ln -s ../../bridge_ros "$vla_root/sim_ws/src/bridge_ros"
fi
cd "$vla_root/sim_ws"
# Explicit message generation handles incomplete dependency declarations upstream.
catkin_make -j"${VLA_BUILD_JOBS:-2}" -l"${VLA_BUILD_JOBS:-2}" -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 --pkg quadrotor_msgs traj_utils
catkin_make -j"${VLA_BUILD_JOBS:-2}" -l"${VLA_BUILD_JOBS:-2}" -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3
printf '\n仿真编译完成。运行 bash scripts/run_attack_showcase.sh --family cracks --view-only\n'
