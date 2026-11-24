#!/usr/bin/env bash
set -euo pipefail
vla_pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vla_env="$(cd -- "$vla_pkg/../../overrides/plan_env" && pwd)"
vla_test_dir="$(mktemp -d /tmp/vla-sensor-geometry.XXXXXX)"
trap 'rm -rf -- "$vla_test_dir"' EXIT
g++ -std=c++14 -O2 -I/usr/include/eigen3 -I"$vla_env/include" "$vla_pkg/tests/raycast_geometry.cpp" "$vla_env/src/raycast.cpp" -o "$vla_test_dir/raycast"
g++ -std=c++14 -O2 -I/usr/include/eigen3 -I"$vla_env/include" "$vla_pkg/tests/known_free_envelope.cpp" -o "$vla_test_dir/envelope"
"$vla_test_dir/raycast"
"$vla_test_dir/envelope"
