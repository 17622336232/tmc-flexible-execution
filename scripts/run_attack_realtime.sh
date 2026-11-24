#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
exec python3 -m experiments.task_attack_realtime.run "$@"
