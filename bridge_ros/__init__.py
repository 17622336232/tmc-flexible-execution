"""Source-tree import shim; implementations live in the standard catkin src tree."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent / "src" / "bridge_ros"))
