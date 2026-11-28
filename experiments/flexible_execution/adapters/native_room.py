"""Feed the public simulator's camera and measured turns to the ordinary task loop."""
from math import atan2, cos, dist, pi, radians, sin
from time import monotonic

from experiments.flexible_execution.public_audit import save_segmentation
from experiments.flexible_execution.public_task import MIN_OBJECT_PIXELS, _same_pose
from experiments.flexible_execution.public_visual import camera_jpeg
from experiments.flexible_execution.revision_audit import digest,read,save


def native_pose(event):
    agent=event.metadata["agent"]
    p=agent["position"]
    return dict(position=dict(x=p["x"],y=p["z"],z=p["y"]),yaw=-radians(agent["rotation"]["y"]))


class NativeRoomBackend:
    """Share the benchmark's in-place quarter-turn camera with all controllers."""
    def __init__(self, controller):
        self.controller=controller
        self.scene={"catalog":{"scene_conditions":{"dynamic":False,"state_version":"public-static-room"}}}
        self.execution_deadline_monotonic=None
        self.camera_events={}

    def check_time(self):
        deadline=self.execution_deadline_monotonic
        if deadline is not None and monotonic()>=deadline:
            raise TimeoutError("The public task deadline has expired")

    def observe(self):
        self.check_time()
        event=self.controller.step(action="Pass")
        if not event.metadata.get("lastActionSuccess"):
            raise RuntimeError("The public camera did not return a new simulator observation")
        captured=monotonic()
        pose=native_pose(event)
        # The independent scorer can later map the observation clock back to its simulator event.
        # The controller and scheduling pipeline receive only RGB and the actual camera pose.
        self.camera_events[captured]=event
        return {**pose,"ros_stamp":captured,"data":camera_jpeg(event)},pose

    def move(self, goal):
        self.check_time()
        current=self.controller.last_event
        before=native_pose(current)
        if dist([goal[k] for k in ("x","y","z")],list(before["position"].values()))>.02:
            raise ValueError("This public benchmark exposes only in-place camera rotation")
        raw=goal["yaw"]-before["yaw"]
        delta=atan2(sin(raw),cos(raw))
        if abs(delta)<.02:
            return before
        if abs(abs(delta)-pi/2)>.02:
            raise ValueError("The public benchmark exposes quarter turns")
        action,expected=("RotateLeft",-90) if delta>0 else ("RotateRight",90)
        event=self.controller.step(action=action)
        if not event.metadata.get("lastActionSuccess") or not _same_pose(current.metadata["agent"],
                                                                         event.metadata["agent"],expected):
            raise RuntimeError("The actual public camera pose differs from the admitted turn")
        self.check_time()
        return native_pose(event)

    def approach(self, *_):
        raise ValueError("The matched public inspection collects evidence by turning in place")

    def snapshot(self):
        return native_pose(self.controller.last_event)

    def archive_scoring(self, episode, predicates):
        """Archive the referee evidence only after the native controller has stopped."""
        robot=episode/"private/robot"
        photos={}
        for f in sorted(robot.glob("onboard_*.json")):
            meta=read(f)
            frame=self.camera_events.get(meta["ros_stamp"])
            if frame is None or meta["sha256"]!=digest(f.with_suffix(".jpg")):
                raise ValueError("The scored public observation is not the recorded RGB frame")
            photos[f.stem]=meta["sha256"]
            for category in sorted(set(predicates.values())):
                save_segmentation(episode,f.stem+"_"+category,frame,category,MIN_OBJECT_PIXELS)
        save(episode/"scoring_sources.json",photos)
