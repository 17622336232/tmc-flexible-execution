#!/usr/bin/env bash
set -eo pipefail
vla_setup_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo '请先安装 ROS Noetic，步骤见 README。' >&2
  exit 2
fi
if ! /usr/bin/python3 -m venv --system-site-packages "$vla_setup_root/.venv"; then
  echo '系统缺少 ensurepip；在项目目录内使用 virtualenv 建立环境，不修改系统 Python。'
  /usr/bin/python3 -m pip install --target "$vla_setup_root/.runtime/venv_bootstrap" 'virtualenv==20.26.6'
  PYTHONPATH="$vla_setup_root/.runtime/venv_bootstrap" /usr/bin/python3 -m virtualenv --system-site-packages "$vla_setup_root/.venv"
fi
"$vla_setup_root/.venv/bin/python" -m pip install 'pip==24.3.1' 'setuptools==75.3.2' 'wheel==0.45.1'
"$vla_setup_root/.venv/bin/python" -m pip install -r "$vla_setup_root/requirements.txt"
echo 'Python 环境已建在 .venv；运行脚本会自动使用它。'
