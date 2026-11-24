#!/usr/bin/env bash
set -euo pipefail
vla_pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vla_yaw_test_dir="$(mktemp -d /tmp/vla-trajectory-yaw.XXXXXX)"
trap 'rm -rf -- "$vla_yaw_test_dir"' EXIT
g++ -std=c++14 -O1 -g -Wall -Wextra -Werror -pedantic \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I"$vla_pkg/include" "$vla_pkg/tests/trajectory_yaw_state_test.cpp" \
  -o "$vla_yaw_test_dir/trajectory_yaw_state"
"$vla_yaw_test_dir/trajectory_yaw_state"
