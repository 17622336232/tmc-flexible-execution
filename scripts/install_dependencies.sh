#!/usr/bin/env bash
# Run explicitly on the target Ubuntu computer. No flight hardware is accessed.
set -euo pipefail
source /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 20.04 ]]; then
  echo '本入口针对 Ubuntu 20.04 + ROS Noetic。其他系统请使用对应虚拟机/容器，见 README。' >&2
  exit 2
fi
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release build-essential cmake pkg-config git \
  python3-pip python3-venv python3-dev python3-numpy python3-scipy python3-opencv \
  python3-yaml python3-pil python3-empy python3-catkin-pkg python3-rospkg \
  libeigen3-dev libpcl-dev libarmadillo-dev libopencv-dev libgazebo11-dev \
  fonts-dejavu-core mesa-utils xvfb xauth
if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  vla_key_dir="$(mktemp -d)"
  trap 'rm -rf -- "$vla_key_dir"' EXIT
  curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o "$vla_key_dir/ros.key"
  gpg --batch --yes --dearmor -o "$vla_key_dir/ros-archive-keyring.gpg" "$vla_key_dir/ros.key"
  sudo install -m 0644 "$vla_key_dir/ros-archive-keyring.gpg" /usr/share/keyrings/ros-archive-keyring.gpg
  printf 'deb [arch=%s signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main\n' "$(dpkg --print-architecture)" |
    sudo tee /etc/apt/sources.list.d/ros-noetic.list >/dev/null
  sudo apt-get update
fi
sudo apt-get install -y ros-noetic-desktop-full ros-noetic-gazebo-ros-pkgs \
  ros-noetic-cv-bridge ros-noetic-tf ros-noetic-tf2-geometry-msgs \
  ros-noetic-mavros-msgs ros-noetic-pcl-ros ros-noetic-pcl-conversions \
  ros-noetic-cmake-modules ros-noetic-rqt-image-view
echo '系统依赖安装完毕。下一步：bash scripts/setup_python.sh'
