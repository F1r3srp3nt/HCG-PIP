# Pip handoff: die-toss animation copy

- `original/Pip.blend`: the handoff file exactly as received (read-only, sha256 `b2f679aa…dd4a`), plus its `Pip-v1-reference.png`, `Pip-preview.png` and `README.txt`.
- `Pip_anim.blend`: **the animation copy**. Same model, materials and proportions. It is saved with Blender 4.3, so it still opens in 4.3.2 or newer.
- `animate_pip_handoff.py`: rebuilds `Pip_anim.blend` from the original:
  `python animate_pip_handoff.py -- original/Pip.blend Pip_anim.blend` (needs bpy 4.3 with numpy<2 and scipy).

## What changed in the copy (the rig only)
- `Root`: a new parent of `Chest`. The feet, toes and haunches now follow `Root`, so they stay planted.
- `Forearm.L/R`: run from the elbow (at the foot of each foreleg) to the paw. The paw balls follow them, so the elbows can give.
- `Eye.L/R`: children of `Head`. Iris, pupil and glints follow them, so the gaze can move independently of the head.
- `Camera.Preview`: a fixed camera at the reference-art angle. The original `Camera` is untouched.

## Actions in the copy
- `Pip_DieToss_5s` + `Die_DieToss_5s` (frames 1–120, 24 fps; these are active).
- `Pip_DieToss_original84` + `Die_DieToss_original84`: the handoff's original animation, kept for reference.

## Beats (frame numbers)
| Frames | What happens |
| --- | --- |
| 1 | Hug pose (the handoff rest pose) |
| 10–15 | Eyes drop to the die |
| 16–26 | Head follows |
| 30–38 | Dip |
| **45** | **Release** |
| 46–61 | The die rises just above nose height, off to his left, tumbling once; his eyes track it |
| **62** | **Catch** against his chest |
| 63–69 | Elbows and paws give |
| 67–87 | Tail swings and settles a beat later |
| 73–86 | Left ear perks |
| ~90 | Back in the original pose; holds to 120 |
