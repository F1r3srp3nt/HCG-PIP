# Pip v3: fresh scene built from the single ChatGPT reference

- **Design reference:** `reference/pip_v3_target.png`, the only design input.
- **Scene:** `pip_v3.blend` (Blender 5.2+). It is a new scene with a new mesh; no geometry from any earlier Pip is used.
- **Build:** `python pip3_build.py -- --strands 200000` (Blender 5.2 bpy, numpy, scipy, scikit-image, Pillow).
  - `pip3_design.py`: every shape is placed in the reference image's pixel space through an orthographic camera tilted 10°, so the render lines up with the art.
  - `textures/`: the eye and grin decals, painted from the art's eye and mouth shapes.
- **Approval still:** `previews/pip_v3_approval_still.png`.

**Not rigged or animated yet**; that waits for approval of the still.
