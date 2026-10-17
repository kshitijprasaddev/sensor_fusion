#!/usr/bin/env python3
"""Record only the visible RViz X11 window, using FFmpeg from our Docker image."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--window-id', help='X11 window ID, for selecting between multiple RViz windows')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    if not os.environ.get('DISPLAY'):
        parser.error('Run this from a terminal in your graphical desktop session')
    window_id = args.window_id
    if not window_id:
        tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
        matches = re.findall(r'^\s*(0x[0-9a-f]+) "([^"]*)"', tree, re.MULTILINE)
        client_ids = set(re.findall(r'^\s*(0x[0-9a-f]+).*\("rviz2" "rviz2"\)', tree, re.MULTILINE))
        windows = [(identifier, title) for identifier, title in matches
                   if identifier in client_ids and (title.endswith(' - RViz') or title == 'RViz')]
        if len(windows) != 1:
            parser.error(f'Expected one open RViz window, found {len(windows)}. Start the demo or pass --window-id.')
        window_id = windows[0][0]
    try:
        window_id = str(int(window_id, 0))
    except ValueError:
        parser.error('Window ID must be a decimal or hexadecimal integer')
    repo = Path(__file__).resolve().parent.parent
    output = (args.output or repo / 'results' / f'rviz-{datetime.now():%Y%m%d-%H%M%S}.mp4').resolve()
    if output.exists():
        parser.error(f'Refusing to overwrite {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['docker', 'run', '--rm', '--init', '--network', 'none',
               '--user', f'{os.getuid()}:{os.getgid()}',
               '-e', f'DISPLAY={os.environ["DISPLAY"]}',
               '-v', '/tmp/.X11-unix:/tmp/.X11-unix:ro',
               '-v', f'{output.parent}:/recordings']
    authority = os.environ.get('XAUTHORITY')
    if authority and Path(authority).is_file():
        command += ['-e', 'XAUTHORITY=/tmp/host.xauth', '-v', f'{authority}:/tmp/host.xauth:ro']
    command += ['--entrypoint', 'ffmpeg', os.environ.get('FUSION_IMAGE', 'group8-sensor-fusion:humble'),
                '-hide_banner', '-nostdin', '-n', '-f', 'x11grab', '-framerate', '15',
                '-draw_mouse', '0', '-window_id', window_id, '-i', os.environ['DISPLAY'],
                '-t', str(args.seconds), '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '24',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', f'/recordings/{output.name}']
    print(f'Recording {args.seconds:g} seconds to {output}. Keep RViz visible and its size unchanged.', flush=True)
    subprocess.run(command, check=True)
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
