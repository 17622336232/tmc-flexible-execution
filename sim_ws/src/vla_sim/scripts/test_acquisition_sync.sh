#!/usr/bin/env bash
set -euo pipefail
vla_pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vla_sync_test_dir="$(mktemp -d /tmp/vla-acquisition-sync.XXXXXX)"
trap 'rm -rf -- "$vla_sync_test_dir"' EXIT
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pedantic \
  -I"$vla_pkg/include" -isystem /usr/include/eigen3 \
  "$vla_pkg/tests/acquisition_sync_test.cpp" -o "$vla_sync_test_dir/acquisition_sync"
"$vla_sync_test_dir/acquisition_sync"
