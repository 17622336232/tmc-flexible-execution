#!/usr/bin/env bash
set -euo pipefail
vla_pkg="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="$(mktemp /tmp/vla-photo-route.XXXXXX)"
trap 'rm -f "$binary"' EXIT
g++ -std=c++14 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -I/usr/include/eigen3 -I"$vla_pkg/include" \
  -I"$vla_pkg/../traj_opt/include" \
  "$vla_pkg/tests/local_photo_route_test.cpp" -o "$binary"
"$binary"
