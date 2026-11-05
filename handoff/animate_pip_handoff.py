"""Animate the handoff Pip.blend (the approved V1 model) - die toss, 5 s.

Run with Blender 4.3 (same version the handoff file was saved in), e.g.
    python animate_pip_handoff.py  ORIGINAL.blend  OUT_COPY.blend
The original file is only read. Everything is written to the copy.

Rig additions (the model, materials and proportions are untouched):
  * Root      - new parent of Chest; feet, toes and haunches move to it so they stay planted.
  * Forearm.L/R - elbow -> paw, child of Paw.L/R; the paw balls ride on it, so elbows can give.
  * Eye.L/R   - child of Head; iris, pupil and glints ride on it, so the gaze can move.
  * Camera.Preview - fixed camera at the reference-art angle (the original Camera is kept).
Actions:
  * 'Pip_DieToss_5s' (rig) and 'Die_DieToss_5s' (gold die control), frames 1-120 @ 24 fps.
  * The handoff's original 84-frame actions are kept, renamed '*_original84'.
"""
import math
import sys

import bpy
from mathutils import Matrix, Vector, Quaternion, Euler

SRC, DST = sys.argv[sys.argv.index('--') + 1:][:2] if '--' in sys.argv else sys.argv[1:3]

RIG = 'Pip • pose rig'
DIE = 'Gold die • prop control'
F0, F1 = 1, 120
RELEASE, CATCH = 45, 62
APEX_Z = 2.55          # die centre at the top of the toss: just above nose height (nose z 2.22)
APEX_X = 0.62          # rises off to his left (like the art), so it never hides his eyes
APEX_Y = -1.05         # and out in front of the muzzle


def mat3(pb):
    return pb.bone.matrix_local.to_3x3()


def world_rot(p, y, r):
    return (Matrix.Rotation(math.radians(y), 3, 'Z') @ Matrix.Rotation(math.radians(r), 3, 'Y')
            @ Matrix.Rotation(math.radians(p), 3, 'X'))


def key_rot(pb, f, prr):
    """Rotation in degrees about world axes (rest orientation): pitch X, yaw Z, roll Y."""
    Rb = mat3(pb)
    L = Rb.transposed() @ world_rot(*prr) @ Rb
    if pb.rotation_mode == 'QUATERNION':
        pb.rotation_quaternion = L.to_quaternion()
        pb.keyframe_insert('rotation_quaternion', frame=f, group=pb.name)
    else:
        pb.rotation_euler = L.to_euler(pb.rotation_mode, pb.rotation_euler)
        pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)


def key_loc(pb, f, xyz):
    pb.location = mat3(pb).transposed() @ Vector(xyz)
    pb.keyframe_insert('location', frame=f, group=pb.name)


def reparent(ob, rig, bone):
    mw = ob.matrix_world.copy()
    ob.parent = rig
    ob.parent_type = 'BONE'
    ob.parent_bone = bone
    ob.matrix_parent_inverse = Matrix.Identity(4)
    bpy.context.view_layer.update()
    ob.matrix_world = mw


# ----------------------------------------------------------------------------- open
bpy.ops.wm.open_mainfile(filepath=SRC, load_ui=False, use_scripts=False)
scene = bpy.context.scene
rig = bpy.data.objects[RIG]
die = bpy.data.objects[DIE]

# keep the original animation, but take it off so we rig in the rest (hug) pose
for ob, tag in ((rig, 'Pip'), (die, 'Die')):
    act = ob.animation_data.action
    act.name = f'{tag}_DieToss_original84'
    act.use_fake_user = True
    ob.animation_data.action = None
for pb in rig.pose.bones:
    pb.location = (0, 0, 0)
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.rotation_euler = (0, 0, 0)
die.location = (0.0, -0.715, 1.12)
die.rotation_euler = (0.0, -0.16, 0.0)
bpy.context.view_layer.update()
DIE_REST = die.matrix_world.copy()

# ----------------------------------------------------------------------------- rig additions
bpy.context.view_layer.objects.active = rig
for o in bpy.context.view_layer.objects:
    o.select_set(False)
rig.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
eb = rig.data.edit_bones
root = eb.new('Root')
root.head, root.tail, root.roll = (0, 0, 0), (0, 0.6, 0), 0
root.use_deform = False
eb['Chest'].parent = root
for s, sx in ((1, 'L'), (-1, 'R')):
    fa = eb.new(f'Forearm.{sx}')
    fa.head = (s * 0.47, -0.40, 0.95)          # elbow at the foot of the foreleg
    fa.tail = (s * 0.433, -0.69, 1.22)         # paw
    fa.roll = 0
    fa.parent = eb[f'Paw.{sx}']
    ey = eb.new(f'Eye.{sx}')
    ey.head = (s * 0.327, -0.60, 2.335)
    ey.tail = (s * 0.327, -0.78, 2.335)
    ey.roll = 0
    ey.parent = eb['Head']
bpy.ops.object.mode_set(mode='OBJECT')
for b in ('Root', 'Forearm.L', 'Forearm.R', 'Eye.L', 'Eye.R'):
    rig.pose.bones[b].rotation_mode = 'XYZ'
bpy.context.view_layer.update()

for o in list(bpy.data.objects):
    n = o.name
    if n.startswith(('Foot ', 'Toe ', 'Haunch ')):
        reparent(o, rig, 'Root')
    elif n in ('Paw L', 'Paw R'):
        reparent(o, rig, f'Forearm.{n[-1]}')
    elif n.startswith(('Iris ', 'Pupil ', 'Glint ')):
        reparent(o, rig, f'Eye.{n[-1]}')

# ----------------------------------------------------------------------------- camera
cam_old = bpy.data.objects['Camera']
cd = cam_old.data.copy()
cd.type = 'ORTHO'
cd.ortho_scale = 4.75
cam = bpy.data.objects.new('Camera.Preview', cd)
scene.collection.objects.link(cam)
tgt = Vector((-0.15, -0.35, 1.80))
az, el, dist = math.radians(22), math.radians(8.6), 9.3     # reference-art angle (Pip's right, slightly above)
cam.location = tgt + Vector((-math.sin(az) * math.cos(el) * dist, -math.cos(az) * math.cos(el) * dist,
                             math.sin(el) * dist))
cam.rotation_euler = (tgt - cam.location).to_track_quat('-Z', 'Y').to_euler()
scene.camera = cam

# ----------------------------------------------------------------------------- acting
pb = rig.pose.bones
rig.animation_data_create()
SHO = [(1, 0), (14, 0), (26, 2), (34, 13), (38, 15), (RELEASE, -38), (48, -30), (54, -19),
       (60, -14), (CATCH, -9), (65, 9), (70, -3), (78, 1), (88, 0), (120, 0)]
SPREAD = [(1, 0), (RELEASE, 0), (48, 9), (56, 11), (60, 6), (CATCH, 0), (120, 0)]
ELBOW = [(1, 0), (26, 2), (34, 9), (38, 10), (RELEASE, -8), (48, 0), (CATCH - 1, -2), (CATCH, 0),
         (65, 17), (69, 6), (75, -2), (84, 0), (120, 0)]
CHEST = [(1, (0, 0, 0)), (20, (1, 0, 0)), (34, (5, 0, 0)), (38, (6, 0, 0)), (RELEASE, (-4, 0, 0)),
         (52, (-3, 0, 0)), (CATCH, (0, 0, 0)), (65, (5, 0, 1)), (70, (-1, 0, 0)), (80, (0.5, 0, 0)),
         (90, (0, 0, 0)), (104, (0.8, 0, 0)), (120, (0, 0, 0))]
HEAD = [(1, (0, 0, 0)), (16, (0, 0, 0)), (26, (11, 0, 3)), (34, (13, 0, 3)), (RELEASE, (5, 0, 1)),
        (50, (-10, 0, -1)), (54, (-13, 0, -2)), (58, (-4, 0, 0)), (CATCH, (8, 0, 1)), (66, (11, 0, 2)),
        (74, (7, 0, 1)), (84, (2, 0, 0)), (94, (0, 0, 0)), (120, (0, 0, 0))]
TAIL = [(1, (0, 0, 0)), (34, (0, 0, -4)), (RELEASE, (0, 6, 5)), (54, (0, 2, 2)), (CATCH, (0, 0, 0)),
        (67, (0, -10, -9)), (73, (0, 7, 6)), (80, (0, -4, -3)), (87, (0, 1.5, 1)), (96, (0, 0, 0)),
        (120, (0, 0, 0))]
EAR_L = [(1, (0, 0, 0)), (34, (-4, 0, 3)), (RELEASE, (2, 0, -2)), (CATCH, (0, 0, 0)), (66, (3, 0, 2)),
         (71, (0, 0, 0)), (75, (-7, 0, -16)), (79, (-3, 0, -9)), (86, (-4, 0, -11)), (96, (-1, 0, -3)),
         (108, (0, 0, 0)), (120, (0, 0, 0))]          # perks up a beat after the catch
EAR_R = [(1, (0, 0, 0)), (34, (-4, 0, -3)), (RELEASE, (2, 0, 2)), (CATCH, (0, 0, 0)), (66, (4, 0, -3)),
         (72, (0, 0, 0)), (120, (0, 0, 0))]

for f, a in SHO:
    key_rot(pb['Paw.L'], f, (a, 0, 0))
    key_rot(pb['Paw.R'], f, (a, 0, 0))
# spread: yaw the shoulder outward after the release (keys merged on top of SHO pitch)
from scipy.interpolate import PchipInterpolator  # noqa: E402  (bpy 4.3 venv ships scipy)
import numpy as np  # noqa: E402


def curve(keys):
    fr = np.array([k[0] for k in keys], float)
    val = np.array([np.atleast_1d(k[1]) for k in keys], float)
    p = PchipInterpolator(fr, val, axis=0)
    return lambda t: p(np.clip(t, fr[0], fr[-1]))


sho, spr = curve(SHO), curve(SPREAD)
for f in sorted({k[0] for k in SHO} | {k[0] for k in SPREAD}):
    a, sp = float(sho(f)[0]), float(spr(f)[0])
    key_rot(pb['Paw.L'], f, (a, sp, 0))
    key_rot(pb['Paw.R'], f, (a, -sp, 0))
for f, a in ELBOW:
    key_rot(pb['Forearm.L'], f, (a, 0, 0))
    key_rot(pb['Forearm.R'], f, (a, 0, 0))
for name, keys in (('Chest', CHEST), ('Head', HEAD), ('Ear.L', EAR_L), ('Ear.R', EAR_R)):
    for f, v in keys:
        key_rot(pb[name], f, v)
tail_c = curve(TAIL)
for f in range(F0, F1 + 1, 2):                       # tail: a touch of idle sway on top
    p, y, r = tail_c(f)
    key_rot(pb['Tail'], f, (p, y + 1.5 * math.sin(f / 9.0), r))

# ----------------------------------------------------------------------------- the die
paw_l, paw_r = bpy.data.objects['Paw L'], bpy.data.objects['Paw R']
scene.frame_set(F0)
REST_L, REST_R = paw_l.matrix_world.copy(), paw_r.matrix_world.copy()
mid_rest = (REST_L.translation + REST_R.translation) / 2
off_rest = DIE_REST.translation - mid_rest
rot_rest = DIE_REST.to_quaternion()


def held(f):
    scene.frame_set(f)
    ml, mr = paw_l.matrix_world, paw_r.matrix_world
    mid = (ml.translation + mr.translation) / 2
    dq = ml.to_quaternion() @ REST_L.to_quaternion().inverted()
    return mid + dq @ off_rest, dq @ rot_rest


p0, q0 = held(RELEASE)
p1, q1 = held(CATCH)
T = CATCH - RELEASE
a_up = APEX_Z - p0.z
b_dn = APEX_Z - p1.z
t_up = T * math.sqrt(a_up) / (math.sqrt(a_up) + math.sqrt(b_dn))
g = 2 * a_up / t_up ** 2
vz = g * t_up
dq = q1 @ q0.inverted()
axis, ang = dq.to_axis_angle()
if axis.length < 1e-6:
    axis = Vector((1, 0, 0))
spin_axis = (axis + Vector((0.0, 0.0, 0.0))).normalized()
total = ang + 2 * math.pi                       # one full tumble, lands on the same face
die.animation_data_create()
die.rotation_mode = 'QUATERNION'
track = {}
for f in range(F0, F1 + 1):
    if RELEASE < f < CATCH:
        t = f - RELEASE
        k = t / T
        arc = math.sin(math.pi * k) ** 0.6
        y = p0.y + (p1.y - p0.y) * k + (APEX_Y - (p0.y + p1.y) / 2) * arc
        x = p0.x + (p1.x - p0.x) * k + APEX_X * arc
        loc = Vector((x, y, p0.z + vz * t - 0.5 * g * t * t))
        rot = Quaternion(spin_axis, total * k) @ q0
    else:
        loc, rot = held(f)
    die.location = loc
    die.rotation_quaternion = rot
    die.keyframe_insert('location', frame=f, group='Die')
    die.keyframe_insert('rotation_quaternion', frame=f, group='Die')
    track[f] = loc.copy()
for fc in die.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'

# ----------------------------------------------------------------------------- eyes
# Eyes lead: they drop to the die first (f10-15), the head follows (f16-26), they ride
# the die up and back down, stay on it after the catch, then come home to centre.
eye_c = {s: Vector((s * 0.327, -0.60, 2.335)) for s in (1, -1)}
EYE_X, EYE_Z = 0.034, 0.042


def gaze_offset(f, s):
    scene.frame_set(f)
    hb = rig.matrix_world @ pb['Head'].matrix @ pb['Head'].bone.matrix_local.inverted()
    tgt_local = hb.inverted() @ track[f]                 # die in the head's rest space
    d = tgt_local - eye_c[s]
    n = max(d.length, 1e-6)
    return Vector((max(-1, min(1, 2.2 * d.x / n)) * EYE_X, 0, max(-1, min(1, 2.2 * d.z / n)) * EYE_Z))


weights = []
for f in range(F0, F1 + 1):
    if f < 10:
        w = 0.0
    elif f < 15:
        k = (f - 10) / 5
        w = k * k * (3 - 2 * k)
    elif f <= 76:
        w = 1.0
    elif f < 88:
        k = (f - 76) / 12
        w = 1 - k * k * (3 - 2 * k)
    else:
        w = 0.0
    weights.append((f, w))
for f, w in weights:
    for s, sx in ((1, 'L'), (-1, 'R')):
        off = gaze_offset(f, s) * w if w > 0 else Vector((0, 0, 0))
        off.y = 0.012 * (abs(off.x) / EYE_X + abs(off.z) / EYE_Z) * 0.5   # hug the curved eye
        key_loc(pb[f'Eye.{sx}'], f, tuple(off))

act = rig.animation_data.action
act.name = 'Pip_DieToss_5s'
act.use_fake_user = True
die.animation_data.action.name = 'Die_DieToss_5s'
die.animation_data.action.use_fake_user = True
scene.frame_start, scene.frame_end = F0, F1
scene.render.fps = 24
scene.frame_set(F0)
bpy.ops.wm.save_as_mainfile(filepath=DST, compress=True)
print('[handoff] saved', DST, 'release', RELEASE, 'catch', CATCH, 'apex', APEX_Z)
