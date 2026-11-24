"""Render a pinned ProcTHOR validation house and verify a real camera action."""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from time import monotonic

from experiments.task_attack.runtime import write_json


PROCTHOR_REVISION = "439193522244720b86d8c81cde2e51e3a4d150cf"
VALIDATION_SHA256 = "d808540514e26b6726cd2790490e669b572eeb94febb5188a2f403591dd21721"
VALIDATION_BYTES = 5094681
THOR_COMMIT = "f0825767cd50d69f666c7f282e54abfe58f1e917"
VALIDATION_SOURCE = {"official_split": "val", "archive_bytes": VALIDATION_BYTES,
                     "archive_sha256": VALIDATION_SHA256}


def _house(archive, index, *, source=VALIDATION_SOURCE):
    data = archive.read_bytes()
    if len(data) != source["archive_bytes"] or hashlib.sha256(data).hexdigest() != source["archive_sha256"]:
        raise ValueError("The archive does not match the pinned official ProcTHOR split")
    with gzip.open(archive, "rt", encoding="utf8") as handle:
        for position, content in enumerate(handle):
            if position == index:
                return json.loads(content), hashlib.sha256(content.encode("utf8")).hexdigest()
    raise ValueError("House index is outside the official split")


def _platform(renderer):
    from ai2thor.platform import CloudRendering, Linux64
    return Linux64 if renderer == "opengl" else CloudRendering


def prepare_build(build_dir, renderer):
    from ai2thor.build import Build

    class HTTPSBuild(Build):
        def _base_url(self):
            return super()._base_url().replace("http://", "https://", 1)

    build = HTTPSBuild(_platform(renderer), THOR_COMMIT, False, str(build_dir.resolve() / "releases"))
    build.download()
    return {"commit": THOR_COMMIT, "source": build.url, "executable": build.executable_path}


def open_controller(house, build_dir, renderer, display, *, segmentation=False, timeout=90):
    from ai2thor.controller import Controller

    class ProjectController(Controller):
        @property
        def base_dir(self):
            return str(build_dir.resolve())

        def unity_command(self, width, height, headless):
            command = super().unity_command(width, height, headless)
            return command + ["-force-glcore"] if renderer == "opengl" else command

    return ProjectController(scene=house, platform=_platform(renderer), commit_id=THOR_COMMIT,
                             width=320, height=240, x_display=display, renderDepthImage=False,
                             renderInstanceSegmentation=segmentation, rotateStepDegrees=90,
                             server_start_timeout=timeout)


def camera_jpeg(event):
    from PIL import Image

    pixels = event.frame
    if not event.metadata["lastActionSuccess"] or pixels is None or pixels.shape != (240, 320, 3):
        raise RuntimeError("The Unity renderer did not return a complete RGB observation")
    if int(pixels.min()) >= int(pixels.max()):
        raise RuntimeError("The camera returned a single-color frame")
    payload = BytesIO()
    Image.fromarray(pixels).save(payload, format="JPEG", quality=90)
    return payload.getvalue()


def source_record(index, digest, renderer, mode, *, source=VALIDATION_SOURCE):
    return {"dataset": "ProcTHOR-10K", "revision": PROCTHOR_REVISION, **source, "house_index": index,
            "house_sha256": digest, "mode": mode, "ai2thor": "5.0.0",
            "platform": _platform(renderer).__name__, "renderer": renderer, "build_commit": THOR_COMMIT}


def smoke(archive, output, build_dir, index=0, renderer="vulkan", display=None):
    house, digest = _house(archive, index)
    started = monotonic()
    output.mkdir(parents=True, exist_ok=False)
    manifest = source_record(index, digest, renderer, "PUBLIC_VISUAL_VALIDATION_SMOKE")
    write_json(output / "source.json", manifest)
    controller = None
    try:
        controller = open_controller(house, build_dir, renderer, display)
        observations = []
        for step, event in enumerate((controller.last_event, controller.step(action="RotateRight")), 1):
            path = output / f"camera_{step}.jpg"
            path.write_bytes(camera_jpeg(event))
            pose = event.metadata["agent"]
            observations.append({"frame": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "position": pose["position"], "rotation": pose["rotation"]})
        if (observations[0]["sha256"] == observations[1]["sha256"]
                or observations[0]["rotation"] == observations[1]["rotation"]):
            raise RuntimeError("The action did not change the observed camera view")
        result = {"complete": True, "observations": observations, "wall_seconds": monotonic() - started}
    except Exception as error:
        result = {"complete": False, "error": {"type": type(error).__name__, "message": str(error)[:1500]},
                  "wall_seconds": monotonic() - started}
    finally:
        if controller is not None:
            controller.stop()
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--house-index", type=int, default=0)
    parser.add_argument("--prepare-build", action="store_true")
    parser.add_argument("--renderer", choices=("vulkan", "opengl"), default="vulkan")
    parser.add_argument("--display", help="Existing isolated X display for the OpenGL fallback")
    args = parser.parse_args()
    if args.prepare_build:
        print(json.dumps(prepare_build(args.build_dir, args.renderer), ensure_ascii=False))
        return 0
    if args.validation is None or args.output is None or args.house_index < 0:
        parser.error("Provide a validation archive, new output directory and nonnegative house index")
    display = args.display or os.environ.get("DISPLAY")
    if args.renderer == "opengl" and not display:
        parser.error("The OpenGL fallback requires an existing isolated X display")
    result = smoke(args.validation, args.output, args.build_dir, args.house_index, args.renderer, display)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
