#!/usr/bin/env bash
set -euo pipefail
vla_pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vla_sampling_test_dir="$(mktemp -d /tmp/vla-indoor-sampling.XXXXXX)"
trap 'rm -rf -- "$vla_sampling_test_dir"' EXIT
g++ -std=c++14 -O2 -Wall -Wextra -Werror -pedantic \
  -I"$vla_pkg/include" "$vla_pkg/tests/indoor_sampling_test.cpp" \
  -o "$vla_sampling_test_dir/indoor_sampling"
"$vla_sampling_test_dir/indoor_sampling"
