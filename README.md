# HCG-PIP
Our cute fox friend :) The first video game character from HCG <3

Pip is the arctic-fox mascot for HCG LifeOS. This project starts from the
**Pip Likeness Study**: the approved character art, rebuilt as a Blender scene.

## Files

| Path | What it is |
| --- | --- |
| `blender/Pip-Likeness.blend` | The Blender scene (saved with Blender 4.3; opens in 4.3 or newer). All textures are packed inside. |
| `reference/Pip-Approved-Character-Reference.png` | The approved character art: the design target |
| `reference/Pip-front.png`, `reference/Pip-turned.png`, `reference/Pip-comparison.jpg` | The study's own renders and comparison sheet |
| `renders/` | Fresh renders of both cameras in the scene |

## Inside `Pip-Likeness.blend`

**Cameras** (both orthographic):
- `01 • Artwork view`: lines up with the approved art.
- `02 • Rotated 30 degrees`: for checking the 3D shape.

**Parts.** Each part is a separate mesh painted from the approved art:
- Body, torso undercoat, head (sculpted face relief) and cranium
- Near and far ears, paws, haunches and feet
- Curled tail plume
- Scarf collar and trailing cloth
- Gold die

**Hierarchy.** The parts hang off a simple tree of empties:
- `Chest` is the parent of the body, haunches, feet, scarf, `Tail`, `Head`, `Paw.Near` and `Paw.Far`.
- `Head` is the parent of the face, cranium, `Ear.Near` and `Ear.Far`.
- The die hangs off its own `Prop` empty.

**Lights:** `Softbox`, `Fill`, `Rim`.

**Render settings:** Cycles, 640 × 640, transparent background, Standard view transform.
