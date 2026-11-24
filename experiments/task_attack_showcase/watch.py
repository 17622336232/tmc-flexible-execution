"""Read-only terminal explanation of this demonstration's actual edit events.

This process cannot publish an edit, choose an action or feed the robot/model.
It exits with its parent and never changes the native episode files.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time


def describe_action(action_id):
    match = re.fullmatch(r'place_decal_shell_(cracked|smooth)_(\d+)', action_id)
    if match:
        appearance = '增加裂纹图案覆片' if match[1] == 'cracked' else '放置光滑覆片遮住原表面'
        return '从左数第%d台设备，正面门板%s；编号保持不变' % (int(match[2]), appearance)
    match = re.fullmatch(r'place_decal_display_(\d+)_(\d+)', action_id)
    if match:
        return '从左数第%d台设备，右侧温度显示覆为%d°C；正面裂纹和编号不变' % (int(match[2]), int(match[1]))
    return '执行模型选择的物理修改：' + action_id


def explain(event, takeoff_wall):
    if event.get('event') == 'physical_edit':
        elapsed = '起飞后约%.1f秒，' % max(0., event['wall'] - takeoff_wall) if takeoff_wall is not None else ''
        if event.get('service_success'):
            print('\n[攻击接口确认成功] ' + elapsed + describe_action(event.get('action_id', '未知动作')) + '。', flush=True)
            print('[观察提示] 覆片直接出现在模型选择的位置，没有机械臂贴覆动画。动作编号：' + event.get('action_id', '未知'), flush=True)
        else:
            print('\n[攻击接口未确认成功] ' + event.get('action_id', '未知动作') + '；不能据此认定画面已经变化。', flush=True)
    elif event.get('event') == 'episode_interrupted':
        print('[展示结束] 本轮被中断；是否已实施攻击以此前的攻击接口记录为准。', flush=True)


def run(flight, parent_pid=None, replay=False):
    path = flight / 'private/events.jsonl'
    takeoff_wall = None
    stream = None
    pending = ''
    try:
        while replay or os.getppid() == parent_pid:
            if stream is None and path.exists():
                stream = path.open()
            if stream is not None:
                pending += stream.read()
                lines = pending.split('\n')
                pending = lines.pop()
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get('event') == 'takeoff_complete':
                        takeoff_wall = event['wall']
                    explain(event, takeoff_wall)
            if replay:
                break
            time.sleep(.2)
    finally:
        if stream is not None:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flight', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if not args.replay and args.parent_pid is None:
        parser.error('Live display requires its owning parent PID')
    try:
        run(args.flight, args.parent_pid, args.replay)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
