"""Render Pip's review previews from pip.blend.

    python render_previews.py --which catch|turntable|both [--res 640] [--samples 32]

Writes PNG frames to previews/frames_<name>/ (re-runs skip finished frames),
then encodes previews/pip_<name>.mp4 with ffmpeg and a contact sheet.
"""
import argparse
import os
import subprocess
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def setup(res, samples):
    s = bpy.context.scene
    s.render.resolution_x = s.render.resolution_y = res
    s.render.resolution_percentage = 100
    s.cycles.samples = samples
    s.render.image_settings.file_format = 'PNG'
    return s


def render_range(s, name, f0, f1, batch=0):
    out = os.path.join(ROOT, 'previews', f'frames_{name}')
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    done = 0
    for f in range(f0, f1 + 1):
        path = os.path.join(out, f'{name}_{f:04d}.png')
        if os.path.exists(path):
            continue
        if batch and done >= batch:
            return None
        done += 1
        s.frame_set(f)
        s.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f'[render] {name} frame {f}/{f1}  {time.time() - t0:.0f}s', flush=True)
    return out


def encode(name, frames_dir, fps=24):
    mp4 = os.path.join(ROOT, 'previews', f'pip_{name}.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-pattern_type', 'glob',
                    '-i', os.path.join(frames_dir, f'{name}_*.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '17', '-movflags', '+faststart', mp4], check=True)
    sheet = os.path.join(ROOT, 'previews', f'pip_{name}_contact.jpg')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-pattern_type', 'glob',
                    '-i', os.path.join(frames_dir, f'{name}_*.png'),
                    '-vf', 'select=not(mod(n\\,8)),scale=320:-1,tile=6x2', '-frames:v', '1', '-q:v', '3', sheet],
                   check=True)
    print('[render] wrote', mp4, sheet)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', default='both')
    ap.add_argument('--res', type=int, default=640)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--blend', default=os.path.join(HERE, 'pip.blend'))
    ap.add_argument('--batch', type=int, default=0, help='render at most N new frames, then stop')
    a = ap.parse_args(argv)
    if a.which in ('catch', 'both'):
        bpy.ops.wm.open_mainfile(filepath=a.blend)
        s = setup(a.res, a.samples)
        s.camera = bpy.data.objects['Cam_ThreeQuarter']
        d = render_range(s, 'catch', 1, 120, a.batch)
        if d:
            encode('catch', d)
    if a.which in ('turntable', 'both'):
        bpy.ops.wm.open_mainfile(filepath=a.blend)
        s = setup(a.res, a.samples)
        s.camera = bpy.data.objects['Cam_Turntable']
        rig = bpy.data.objects['Pip_Rig']
        die = bpy.data.objects['Gold_Die']
        rig.animation_data.action = bpy.data.actions['Pip_Turntable']
        die.animation_data.action = None           # keep the die in his paws
        die.constraints['Held by Pip'].influence = 1.0
        d = render_range(s, 'turntable', 1, 96, a.batch)
        if d:
            encode('turntable', d)


if __name__ == '__main__':
    main()
