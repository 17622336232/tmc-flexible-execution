"""Check the common native camera path in two previously exposed ProcTHOR rooms."""
import argparse
from hashlib import sha256
import os
from pathlib import Path
from time import monotonic

from experiments.flexible_execution.adapters.native_room import NativeRoomBackend
from experiments.flexible_execution.public_visual import _house,open_controller
from experiments.flexible_execution.revision_audit import digest,read,save
from research.flexible_scheduling.ordinary_frontier import camera_goals,pose_matches
from scripts.continuing_room_event_calibration import launch_local,prepare


ROOT=Path("outputs/continuing/native-public-preflight-dev01")
SOURCE=Path(__file__).resolve()


def capture(runtime,case,output):
    house,world=_house(Path(runtime["archive"]),case["house"])
    controller=open_controller(house,Path(runtime["build_dir"]),"opengl",os.environ["DISPLAY"],segmentation=True)
    try:
        backend=NativeRoomBackend(controller)
        photo,pose=backend.observe()
        camera={**case,"angle_degrees":90}
        goals=camera_goals(camera,pose["position"],pose["yaw"])
        before=monotonic()
        moved=backend.move(goals["side"])
        turned=monotonic()
        side,geometry=backend.observe()
        acquired=monotonic()
        returned=backend.move(goals["front"])
        finished=monotonic()
        frames={name:sha256(img["data"]).hexdigest() for name,img in (("front",photo),("side",side))}
        checks={
            "side_pose":pose_matches({**moved["position"],"yaw":moved["yaw"]},goals["side"]),
            "camera_pose":pose_matches({**geometry["position"],"yaw":geometry["yaw"]},goals["side"]),
            "return_pose":pose_matches({**returned["position"],"yaw":returned["yaw"]},goals["front"]),
            "distinct_rgb":frames["front"]!=frames["side"],
            "fresh_exposure":side["ros_stamp"]>photo["ros_stamp"],
        }
        save(output,dict(house=case["house"],world=world,checks=checks,photos=frames,
                         turn_s=turned-before,photo_s=acquired-turned,return_s=finished-acquired,new_requests=0))
    finally:
        controller.stop()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case")
    args=parser.parse_args()
    cfg,runtime=prepare()
    cases=cfg["cases"]
    ROOT.mkdir(exist_ok=True)
    if args.case:
        case=next(c for c in cases if c["id"]==args.case)
        capture(runtime,case,ROOT/(case["id"]+".json"))
        return
    result={}
    for case in cases:
        key=case["id"]
        output=ROOT/(key+".json")
        if not output.exists():
            proc=launch_local(runtime,case,SOURCE,["--case",key],ROOT/(key+".log"))
            if proc.wait()!=0:
                raise RuntimeError("Local public camera preflight did not finish for "+key)
        result[key]=dict(data=read(output),sha256=digest(output))
    save(ROOT/"summary.json",dict(new_requests=0,rooms=result))
    print([(k,all(v["data"]["checks"].values())) for k,v in result.items()])


if __name__=="__main__":
    main()
