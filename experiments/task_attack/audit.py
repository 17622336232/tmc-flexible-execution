"""Version identity for frozen pilot runs; never hash credentials or settings."""
import hashlib
import json
import shutil
from pathlib import Path


def implementation_files():
    root=Path(__file__).resolve().parents[2]
    files=list((root/'experiments/task_attack').glob('*.py'))
    files += [root/'experiments/task_attack/cpu_backend.launch',
              root/'experiments/task_attack/requirements-cpu.txt']
    files += [root/p for p in (
        'bridge/uav_bridge/responses_transport.py',
        'bridge_ros/src/bridge_ros/ros_adapter.py',
        'scripts/run_inspection.sh',
        'sim_ws/src/vla_sim/launch/demo.launch',
        'sim_ws/src/vla_sim/launch/planner.launch',
        'sim_ws/src/vla_sim/config/planner.yaml',
        'sim_ws/src/vla_sim/scripts/flight_controller.py',
        'sim_ws/src/vla_sim/models/vla_uav/model_indoor.sdf')]
    for name in ('sim_ws/devel/lib/vla_sim/vla_diff_planner_node',
                 'sim_ws/devel/lib/vla_sim/vla_traj_server',
                 'sim_ws/devel/lib/vla_sim/vla_cloud_bridge',
                 'sim_ws/devel/lib/libvla_acquisition_lidar.so'):
        if (root/name).exists():files.append(root/name)
    return sorted(files)


def implementation_hash():
    root=Path(__file__).resolve().parents[2]
    digest=hashlib.sha256()
    for path in implementation_files():
        digest.update(str(path.relative_to(root)).encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def snapshot_implementation(destination):
    root=Path(__file__).resolve().parents[2]
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    entries=[]
    for source in implementation_files():
        relative=source.relative_to(root)
        binary=source.suffix=='.so' or 'devel' in relative.parts
        entries.append(dict(path=str(relative),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),copied=not binary))
        if not binary:
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(str(source),str(target))
    (destination/'snapshot_manifest.json').write_text(json.dumps(dict(implementation_hash=implementation_hash(),files=entries),indent=2))
