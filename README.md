# HCG-PIP
Our cute fox friend :) The first video game character from HCG <3

Pip is the arctic-fox mascot for HCG LifeOS. This repo has a rigged, furred 3D
prototype of Pip built in Blender, plus his animations.

## v2 (current)

| Path | What it is |
| --- | --- |
| `blender/pip_v2.blend` | **Editable v2 file**: re-proportioned Pip, the rig, and the *Near Miss* action. Needs Blender 5.2+. |
| `previews/pip_v2_near_miss.mp4` | The 5 s locked-camera performance |
| `previews/pip_v2_likeness_check.png` | Reference art vs. v2 from the same angle (grin, and the relaxed resting mouth) |
| `reference/pip_v2_sheet.png` | The character sheet v2 was matched to |

**Model and rig changes from v1:**
- **Head and ears:** bigger head; broad, rounded ears with fluffy white rims and darker blue insides; a sculpted crest of swept tufts and spiky cheek tufts.
- **Face:** larger amber eyes. Each upper lid carries a dark liner band with an outer flick, so the outline follows blinks and squints. Smaller nose, shorter muzzle.
- **Mouth:** a relaxed closed mouth (a fine smile line). The `grin` custom property on `Pip_Rig` (0–1) drives the `Grin` shape key on `Pip_Mouth`, which opens it into a grin with the tongue showing.
- **Colours:** darker navy satin scarf, teal tail tip with a zig-zag edge, and a stronger periwinkle wash in the fur. Uses the *Khronos PBR Neutral* view transform.
- **Die:** a bigger gold die (16 cm) held against the belly, with chubbier arms and toe lobes on the feet.

**Actions in `pip_v2.blend`:**
- `Pip_NearMiss` + `Die_NearMiss` (frames 1–120, 24 fps; camera `Cam_NearMiss` is locked). Key frames:
  - Release: frame 36.
  - Catch: frame 55, off-centre and tilted.
  - Near drop: frames 56–64 (`die_socket` slips toward his left paw, `hold_ctrl` dips so the elbows fold, the tail swings out).
  - Glance: frame 87.
  - Grin: from frame 94.
  - Feet stay planted (`foot_ik` never moves).
- `Pip_Likeness` and `Pip_Rest`: single-frame poses used for the likeness check.

Build and render v2:

```bash
cd blender
python build_pip.py                                   # -> pip_v2.blend
python render_previews.py --which likeness            # -> previews/pip_v2_likeness_check.png
python render_previews.py --which nearmiss            # -> previews/pip_v2_near_miss.mp4
```

The v1 file (`blender/pip.blend`, *Lucky Flick* and turntable) is kept for comparison. The scripts that built it are in commit `cea169d`.

## v1

## What's here

| Path | What it is |
| --- | --- |
| `blender/pip.blend` | **The editable Blender file**: character, rig, fur, scarf, gold die, lights, cameras, both actions. Needs Blender 5.2 or newer. |
| `previews/pip_catch.mp4` | "Lucky Flick", the 5-second catch from the fixed three-quarter camera |
| `previews/pip_turntable.mp4` | 4-second 360° turntable for checking his face and silhouette |
| `previews/*_contact.jpg` | Contact sheets of both previews |
| `blender/*.py` | The procedural build: running it rebuilds `pip.blend` from scratch |
| `reference/` | The 2D reference art the model is matched to |

## Inside `pip.blend`

**Collections:** `Pip` (geometry + rig), `Props` (the die), `Stage` (backdrop, lights, cameras).

**Geometry**
- `Pip_Body`: one skinned mesh for the body, head, limbs and tail, plus the two cupped ear shells. It carries the `pip_color` colour map (cream white, periwinkle markings, blue inner ears, cyan tail tip).
- `Pip_Fur`: about 160k groomed hair curves. The *Deform on Body* geometry-nodes modifier makes them ride the skinned body.
- Separate face parts, each parented to a bone: eyes, upper and lower lids, lash lines with the outer flick, nose and tongue. `Eye_Catchlight` only shows up as the highlight in his eyes.
- `Pip_Scarf`: navy fleece scarf skinned to the chest, neck and four scarf-tail bones.
- `Gold_Die`: a separate object with real recessed pips (opposite faces add up to 7).

**Rig (`Pip_Rig`)**, organised in bone collections:
- *Body*: `root › hips › spine › chest › neck › head`, plus `jaw`, arm chains (`upper_arm / forearm / paw`) and leg chains (`thigh / shin / foot`).
- *Face*: two-bone ears (`ear.01/02.L/R`), `eye.L/R` (they aim at `look_target`), `lid_up` and `lid_low` per eye (rotate X to blink or squint).
- *Tail*: `tail.01`–`tail.06`.  *Scarf*: `scarf_a/b.01-02`.
- *Controls*:
  - `hold_ctrl` carries both paw IK targets (`paw_ik.L/R`) and the `die_socket`, so moving it moves paws and die together.
  - `foot_ik.L/R` keep the feet planted.
  - `look_target` is where both eyes point.

**Holding and releasing the die:** the die has a *Copy Transforms* constraint called **"Held by Pip"** that targets `die_socket`.
- Influence **1**: the die sits in his paws.
- Influence **0**: the die follows its own keyframes, i.e. free flight.

Key that influence with constant interpolation to hand the die off at any frame.

## Actions
- `Pip_LuckyFlick` (frames 1–120, 24 fps): Pip notices the die, winds up and flicks it at frame 37. His eyes and ears follow it up and down. He catches it on frame 70, and his paws, chest and hips dip with the weight while his tail swings out to balance. Then he turns to the camera with smug half-lids, a cocked ear and a slow blink. The die tumbles once on the way and lands with the 6 showing.
  - The body, head, jaw, lids, ears, paws and die-holder are **sparse, hand-editable keys**.
  - The tail, ear tips and scarf tails are baked every frame from those curves, as overlapping action.
  - The die flight and the eye target are baked as well.
- `Die_LuckyFlick`: the die's flight and the influence switches.
- `Pip_Turntable`: a calm hero pose. Only `root` turns, 360° over frames 1–97.

## Rebuild / re-render

The scripts use Blender as a Python module (`pip install bpy==5.2.*`) plus numpy, scipy and scikit-image:

```bash
cd blender
python build_pip.py                    # -> pip.blend (~30 s)
python render_previews.py --which both # -> previews/*.mp4 (Cycles CPU)
```

Key tuning knobs:
- `pip_design.py`: proportions. `HS` and `HC1` set the head scale and seat; there are also the ear shape, tail curve, eye and mouth landmarks.
- `pip_colors.py`: fur colour, length and grooming.
- `pip_anim.py`: the acting keys.
