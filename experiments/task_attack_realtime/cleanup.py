"""Release captured, case-owned ROS core processes after its launcher ends."""
import os
from pathlib import Path
import signal
import time


def belongs_to_case(args,root):
    if not args:return False
    executable=any(a.endswith('/bin/rosmaster') for a in args[:2]) or args[0]=='/opt/ros/noetic/lib/rosout/rosout'
    if not executable:return False
    prefix=str(Path(root).resolve()/'private/roslog')+'/'
    return any(a.startswith('__log:='+prefix) for a in args)


def process_args(pid):
    try:return Path('/proc/%d/cmdline'%pid).read_bytes().decode(errors='replace').split('\0')[:-1]
    except (FileNotFoundError,PermissionError,ProcessLookupError):return []


def capture(root):
    return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdigit() and belongs_to_case(process_args(int(p.name)),root)]


def stop_captured(root,pids):
    remaining=list(pids)
    for signum,timeout in ((signal.SIGINT,3.),(signal.SIGTERM,1.)):
        remaining=[pid for pid in remaining if belongs_to_case(process_args(pid),root)]
        for pid in remaining:
            try:os.kill(pid,signum)
            except ProcessLookupError:pass
        end=time.monotonic()+timeout
        while remaining and time.monotonic()<end:
            remaining=[pid for pid in remaining if belongs_to_case(process_args(pid),root)]
            if remaining:time.sleep(.05)
    return remaining
