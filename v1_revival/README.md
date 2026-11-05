# Pip v1 revival (start of the four-pose acting pass)

- **Source:** `blender/pip.blend` from commit `cea169d`, the earliest Pip v1 Blender scene. It is left untouched.
- **Work copy:** `Pip_v1_work.blend`, built by `revive_v1.py` (Blender 5.2 bpy) using v1's own scripts in `scripts_v1/`, taken from `cea169d`:
  `python revive_v1.py -- ../blender/pip.blend Pip_v1_work.blend`
- **Kept exactly from v1:** body mesh, ears, eyes, face parts, scarf mesh, rig, and both v1 actions.
- **Changed in the copy:**
  - The fur is re-groomed, with a swept fringe, broad tufty cheeks and chunkier tufts.
  - The view transform is now Khronos PBR Neutral.
  - The gold die is 1.4× bigger and glossier, and the paws are widened to grip it.
  - New `Pip_RefPose` action and `Cam_Reference` camera.
- **Still:** `previews/pip_v1_reference_angle_still.png`. Animation waits for approval of this still.
