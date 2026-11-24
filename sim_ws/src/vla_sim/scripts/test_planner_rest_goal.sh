#!/usr/bin/env bash
set -euo pipefail
vla_pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vla_rest_test_dir="$(mktemp -d /tmp/vla-planner-rest.XXXXXX)"
trap 'rm -rf -- "$vla_rest_test_dir"' EXIT
g++ -std=c++14 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"$vla_pkg/include" -isystem /usr/include/eigen3 \
  -isystem "$vla_pkg/../traj_opt/include" \
  "$vla_pkg/tests/planner_rest_goal_test.cpp" -o "$vla_rest_test_dir/planner_rest_goal"
"$vla_rest_test_dir/planner_rest_goal"
