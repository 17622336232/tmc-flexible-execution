#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
if [[ $# -lt 1 ]]; then
  echo '用法：bash scripts/research.sh 模块名 [参数...]' >&2
  echo '例如：bash scripts/research.sh fit_isolated_waiting --help' >&2
  exit 2
fi
vla_research_module="$1"
shift
if [[ ! "$vla_research_module" =~ ^[a-zA-Z0-9_]+$ || ! -f "$vla_source_root/research/attack_learning/$vla_research_module.py" ]]; then
  echo '未知研究模块；见 research/attack_learning/。' >&2
  exit 2
fi
exec python3 -m "$vla_research_module" "$@"
