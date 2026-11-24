#!/usr/bin/env python3
"""Record the running Gazebo gallery's actual RGB camera stream to H.264 MP4.

This tool never starts Gazebo, pauses physics, moves the drone, or issues flight
commands. It injects an observer camera into an existing ROS/Gazebo world and
removes its own camera when finished. Duration refers to simulation time; ROS
image timestamps retain the timing even when Gazebo runs below real time.

Example (source ROS and the workspace first):
  ROS_MASTER_URI=http://127.0.0.1:11342 python3 record_demo.py --duration 25

Requires gazebo_ros, gazebo_plugins, sensor_msgs, and ffmpeg with libx264.
"""
import argparse
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=25.0, help="Video duration in simulation seconds (default: 25)")
    parser.add_argument("--wall-timeout", type=float, default=150.0, help="Maximum wall-clock capture time, including image startup")
    parser.add_argument("--output", type=Path, default=ROOT / "worlds" / "gallery_flight.mp4")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--topic", default="/vla/overview/camera/image_raw")
    parser.add_argument("--model-name", default="vla_overview_camera")
    parser.add_argument("--gazebo-namespace", default="/gazebo")
    parser.add_argument("--no-spawn", action="store_true", help="Capture an existing observer camera without creating/removing it")
    parser.add_argument("--keep-camera", action="store_true", help="Keep the injected observer after capture")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output video")
    parser.add_argument("--pose", type=float, nargs=6, default=[14, -18, 15, 0, .58, 2.23],
                        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    args = parser.parse_args()
    if args.duration <= 0 or args.wall_timeout <= 0 or not 1 <= args.fps <= 30:
        parser.error("duration/wall-timeout must be positive; fps must be 1..30")
    if args.width < 16 or args.height < 16 or args.width % 2 or args.height % 2:
        parser.error("width and height must be positive even integers >= 16")
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists; choose another path or explicitly use --overwrite")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required (Ubuntu: sudo apt install ffmpeg)")
    return args


def camera_xml(args):
    tree = ET.parse(str(ROOT / "models" / "overview_camera" / "model.sdf"))
    sensor = tree.find(".//sensor")
    sensor.find("update_rate").text = str(args.fps)
    sensor.find("camera/image/width").text = str(args.width)
    sensor.find("camera/image/height").text = str(args.height)
    sensor.find("plugin/updateRate").text = str(args.fps)
    # An absolute topic also works with a user-specified namespace/model name.
    sensor.find("plugin/imageTopicName").text = args.topic
    return ET.tostring(tree.getroot(), encoding="unicode")


def spawn_camera(args, rospy):
    from gazebo_msgs.srv import SpawnModel
    from geometry_msgs.msg import Pose
    service = args.gazebo_namespace.rstrip("/") + "/spawn_sdf_model"
    rospy.wait_for_service(service, timeout=15)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = args.pose[:3]
    r, p, y = [v / 2 for v in args.pose[3:]]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    pose.orientation.x = sr*cp*cy - cr*sp*sy
    pose.orientation.y = cr*sp*cy + sr*cp*sy
    pose.orientation.z = cr*cp*sy - sr*sp*cy
    pose.orientation.w = cr*cp*cy + sr*sp*sy
    response = rospy.ServiceProxy(service, SpawnModel)(args.model_name, camera_xml(args), "/vla/overview", pose, "world")
    if not response.success:
        raise RuntimeError("Cannot inject observer camera: " + response.status_message +
                           "; use --no-spawn to capture an existing observer")


def image_bytes(message, width, height):
    if (message.width, message.height) != (width, height):
        raise RuntimeError("Camera size %sx%s does not match requested %sx%s" %
                           (message.width, message.height, width, height))
    if message.encoding not in ("rgb8", "bgr8"):
        raise RuntimeError("Unsupported camera encoding: " + message.encoding)
    row_size = width * 3
    if message.step < row_size or len(message.data) < message.step * height:
        raise RuntimeError("Incomplete camera image buffer")
    if message.step == row_size:
        data = bytes(message.data)
    else:
        data = b"".join(message.data[y*message.step:y*message.step+row_size] for y in range(height))
    return data, "rgb24" if message.encoding == "rgb8" else "bgr24"


def record(args):
    import rospy
    from sensor_msgs.msg import Image
    from gazebo_msgs.srv import DeleteModel

    rospy.init_node("vla_demo_recorder", anonymous=True)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inbox = queue.Queue(maxsize=8)
    dropped = [0]
    injected = False
    process = None
    subscriber = None
    metadata = {"source": "Gazebo RGB camera sensor", "topic": args.topic,
                "requested_simulation_seconds": args.duration,
                "fps": args.fps, "width": args.width, "height": args.height,
                "pose_xyz_rpy": args.pose, "complete": False}
    frames = real_frames = repeated_frames = 0
    first_stamp = last_stamp = None
    previous_frame = None
    started = time.monotonic()
    deadline = started + args.wall_timeout
    wanted_frames = max(1, round(args.duration * args.fps))

    def on_image(message):
        try:
            inbox.put_nowait(message)
        except queue.Full:
            try:
                inbox.get_nowait()
            except queue.Empty:
                pass
            dropped[0] += 1
            try:
                inbox.put_nowait(message)
            except queue.Full:
                dropped[0] += 1

    try:
        if not args.no_spawn:
            spawn_camera(args, rospy)
            injected = True
        subscriber = rospy.Subscriber(args.topic, Image, on_image, queue_size=1,
                                      buff_size=args.width*args.height*3*2, tcp_nodelay=True)
        print("Observer ready; waiting for actual camera frames on " + args.topic, flush=True)
        next_progress = 5
        while not rospy.is_shutdown() and frames < wanted_frames:
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for simulation frames; world may be paused or rendering unavailable")
            try:
                message = inbox.get(timeout=.5)
            except queue.Empty:
                continue
            stamp = message.header.stamp.to_sec()
            if last_stamp is not None and stamp < last_stamp:
                raise RuntimeError("Simulation time moved backwards during capture")
            if last_stamp is not None and stamp == last_stamp:
                continue
            frame, pixel_format = image_bytes(message, args.width, args.height)
            if process is None:
                first_stamp = stamp
                command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if args.overwrite else "-n",
                           "-f", "rawvideo", "-pixel_format", pixel_format,
                           "-video_size", "%sx%s" % (args.width, args.height), "-framerate", str(args.fps),
                           "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
                           "-threads", "2", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
                process = subprocess.Popen(command, stdin=subprocess.PIPE)
                metadata["pixel_format"] = pixel_format
                print("Recording %.1f simulation seconds to %s" % (args.duration, args.output), flush=True)
            elif pixel_format != metadata["pixel_format"]:
                raise RuntimeError("Camera encoding changed during capture")
            index = max(0, int(round((stamp - first_stamp) * args.fps)))
            # Preserve the real timestamp spacing if an image was lost. Repeat
            # the previous observed frame; do not synthesize/interpolate motion.
            while frames < min(index, wanted_frames) and previous_frame is not None:
                process.stdin.write(previous_frame)
                frames += 1
                repeated_frames += 1
            if frames < wanted_frames and index >= frames:
                process.stdin.write(frame)
                frames += 1
            previous_frame = frame
            real_frames += 1
            last_stamp = stamp
            if frames / args.fps >= next_progress:
                print("Captured %.1f / %.1f simulation seconds" % (frames/args.fps, args.duration), flush=True)
                next_progress += 5
        if frames < wanted_frames:
            raise RuntimeError("Recording interrupted before requested duration")
        metadata["complete"] = True
    finally:
        if subscriber is not None:
            subscriber.unregister()
        if process is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                metadata["complete"] = False
            try:
                returncode = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                returncode = -9
            if returncode:
                metadata["complete"] = False
                metadata["encoder_exit_code"] = returncode
        if injected and not args.keep_camera:
            try:
                service = args.gazebo_namespace.rstrip("/") + "/delete_model"
                rospy.wait_for_service(service, timeout=3)
                result = rospy.ServiceProxy(service, DeleteModel)(args.model_name)
                if not result.success:
                    print("Observer cleanup: " + result.status_message, file=sys.stderr)
            except Exception as exc:
                print("Observer cleanup failed: %s" % exc, file=sys.stderr)
        metadata.update({"encoded_frames": frames, "received_frames": real_frames,
                         "repeated_frames": repeated_frames, "queue_dropped_frames": dropped[0],
                         "first_simulation_stamp": first_stamp, "last_simulation_stamp": last_stamp,
                         "encoded_seconds": frames/args.fps, "wall_seconds": time.monotonic()-started})
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not metadata["complete"]:
        raise RuntimeError("Video encoding failed; see capture metadata")
    print("Saved %s (%s frames, %s repeated); metadata: %s" %
          (args.output, frames, repeated_frames, args.output.with_suffix(".json")), flush=True)


def main():
    args = parse_args()
    try:
        record(args)
    except Exception as exc:
        print("Recording failed: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
