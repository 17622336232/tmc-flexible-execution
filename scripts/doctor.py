"""Read-only source-distribution checks. Never logs in or calls a model."""
import argparse
import importlib
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-build', action='store_true')
    parser.add_argument('--require-model', action='store_true')
    args = parser.parse_args()
    failed = []
    def check(label, okay, detail=''):
        print(('[正常] ' if okay else '[缺少] ')+label+('：'+detail if detail else ''))
        if not okay:
            failed.append(label)
    check('Python', sys.version_info[:2] == (3, 8), sys.version.split()[0])
    for binary in ('roslaunch', 'catkin_make', 'gz', 'g++', 'pkg-config'):
        check(binary, bool(shutil.which(binary)))
    for module in ('numpy', 'PIL', 'cv2', 'cryptography', 'scipy', 'sklearn', 'pybullet', 'rospy'):
        try:
            importlib.import_module(module)
            check(module, True)
        except ImportError:
            check(module, False)
    if args.require_build:
        for path in ('sim_ws/devel/setup.bash', 'sim_ws/devel/lib/vla_sim/vla_diff_planner_node',
                     'sim_ws/devel/lib/vla_sim/vla_cloud_bridge', 'sim_ws/devel/lib/libvla_acquisition_lidar.so'):
            check(path, (ROOT/path).is_file())
    if args.require_model:
        from experiments.task_attack_showcase.run import verify_candidate
        try:
            verify_candidate()
            check('本地导入攻击模型与当前编译', True)
        except Exception as error:
            check('本地导入攻击模型与当前编译', False, str(error))
    print('桌面 DISPLAY：'+('已设置' if os.environ.get('DISPLAY') else '未设置；运行时使用 --headless --no-rviz'))
    if failed:
        print('请先解决上述缺少项，步骤见 README。')
        return 1
    print('检查通过。本命令未调用模型、未启动仿真。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
