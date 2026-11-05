"""Pip's first move: 'Lucky Flick' (5 s @ 24 fps) plus a static turntable pose.

Primary body acting is stored as sparse, hand-editable keyframes. Overlapping
action (tail, ear tips, scarf tails), the eye/look target and the die's flight
are baked per frame from the primary curves so the catch has weight.

Rotation values below are degrees about *world* axes in the rest pose:
  pitch (+ = nod forward/down), yaw (+ = turn toward Pip's left / the camera),
  roll (+ = tilt the top toward Pip's left).
"""
import math
import numpy as np
import bpy
from mathutils import Matrix, Vector, Quaternion
from scipy.interpolate import PchipInterpolator

import pip_design as D

F_START, F_END = 1, 120
RELEASE = 37        # die leaves the paws after this frame
CATCH = 70          # die lands in the paws on this frame
APEX_Z = 0.565


# ------------------------------------------------------------------ helpers
def curve(keys):
    fr = np.array([k[0] for k in keys], float)
    val = np.array([np.atleast_1d(np.asarray(k[1], float)) for k in keys])
    if len(fr) == 1:
        return lambda t: val[0]
    p = PchipInterpolator(fr, val, axis=0)
    return lambda t: p(np.clip(t, fr[0], fr[-1]))


def rest_rot(pb):
    return pb.bone.matrix_local.to_3x3()


def world_rot(pitch, yaw, roll):
    return (Matrix.Rotation(math.radians(yaw), 3, 'Z') @ Matrix.Rotation(math.radians(roll), 3, 'Y')
            @ Matrix.Rotation(math.radians(pitch), 3, 'X'))


def key_rot(pb, f, prr):
    Rb = rest_rot(pb)
    L = Rb.transposed() @ world_rot(*prr) @ Rb
    pb.rotation_euler = L.to_euler('XYZ', pb.rotation_euler)
    pb.keyframe_insert('rotation_euler', frame=f, group=pb.name)


def key_loc(pb, f, xyz):
    pb.location = rest_rot(pb).transposed() @ Vector(xyz)
    pb.keyframe_insert('location', frame=f, group=pb.name)


def fcurves(idblock):
    ad = idblock.animation_data
    if not ad or not ad.action:
        return []
    try:
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        cb = action_get_channelbag_for_slot(ad.action, ad.action_slot)
        return list(cb.fcurves) if cb else []
    except ImportError:
        return list(ad.action.fcurves)


def set_interp(idblock, path_contains, mode):
    for fc in fcurves(idblock):
        if path_contains in fc.data_path:
            for kp in fc.keyframe_points:
                kp.interpolation = mode


# ------------------------------------------------------------- the acting
def mirror(keys, flip_yaw=True, flip_roll=True):
    return [(f, (p, -y if flip_yaw else y, -r if flip_roll else r)) for f, (p, y, r) in keys]


C = CATCH
ROT = {
    # whole-body line of action
    'spine': [(1, (0, 0, 0)), (12, (-2, 0, 0)), (20, (3, 0, 1)), (31, (7, 0, 0)), (37, (-6, 0, 0)),
              (54, (-6, 2, -1)), (C, (-2, 0, 0)), (C + 4, (9, -2, -3)), (C + 10, (-3, 0, 1)),
              (C + 16, (2, -2, 0)), (96, (-3, -4.0, -1.0)), (120, (-3, -4.0, -1.0))],
    'chest': [(1, (0, 0, 0)), (12, (-3, 0, 0)), (20, (5, 0, 2)), (31, (10, 0, 0)), (37, (-9, 0, 0)),
              (54, (-10, 3, -2)), (C, (-4, 0, 0)), (C + 4, (14, -3, -4)), (C + 10, (-4, 0, 2)),
              (C + 16, (3, -3, 0)), (96, (-6, -8.0, -3.0)), (120, (-5, -8.0, -3.0))],
    'neck': [(1, (0, 2.0, 1.0)), (12, (0, 1.0, 1.0)), (20, (3, 0, 2)), (31, (4, 0, 1)), (37, (0, 0, 0)),
             (46, (-4, 1, -1)), (54, (-6, 1, -1)), (62, (-3, 1, -1)), (C, (2, 0, 0)), (C + 4, (4, 0, 1)),
             (C + 10, (1, 0, 0)), (C + 18, (1, -4.0, -2.0)), (96, (-2, -6.0, -3.0)), (106, (-2, -6.0, -3.0)),
             (110, (1, -6.0, -3.0)), (114, (-2, -6.0, -3.0)), (120, (-2, -6.0, -3.0))],
    'head': [(1, (0, 5.0, 2.0)), (12, (-1, 3.0, 2.0)), (20, (8, 0, 5)), (27, (9, 0, 7)), (31, (9, 0, 3)),
             (37, (0, 0, 0)), (46, (-9, 1, -3)), (54, (-13, 1, -4)), (62, (-7, 1, -2)), (C, (3, 1, 0)),
             (C + 4, (8, 0, 2)), (C + 10, (2, 0, 0)), (C + 18, (2, -6.0, -3.0)), (96, (-4, -13.0, -7.0)),
             (106, (-5, -14.0, -8.0)), (110, (1, -14.0, -8.0)), (114, (-5, -14.0, -8.0)), (120, (-5, -14.0, -8.0))],
    'jaw': [(1, (0, 0, 0)), (14, (0, 0, 0)), (20, (4, 0, 0)), (31, (-4, 0, 0)), (37, (5, 0, 0)),
            (54, (9, 0, 0)), (C, (3, 0, 0)), (C + 3, (-5, 0, 0)), (C + 10, (6, 0, 0)), (C + 18, (4, 0, 0)),
            (96, (-5, 0, 0)), (120, (-5, 0, 0))],
    # the die in the paws
    'hold_ctrl': [(1, (0, 0, 0)), (12, (0, 0, 0)), (22, (-8, 0, 4)), (31, (12, 0, 0)), (36, (-16, 0, 0)),
                  (40, (-4, 0, 0)), (C, (0, 0, 0)), (C + 4, (8, 0, -3)), (C + 10, (-2, 0, 0)),
                  (C + 16, (0, 0, 0)), (96, (-6, -14.0, 0)), (120, (-6, -14.0, 0))],
    'paw_ik.L': [(1, (0, 0, 0)), (32, (10, 0, 0)), (36, (-28, 0, 0)), (42, (-8, 0, 0)), (C, (0, 0, 0)),
                 (120, (0, 0, 0))],
    'paw_ik.R': [(1, (0, 0, 0)), (32, (10, 0, 0)), (36, (-28, 0, 0)), (42, (-8, 0, 0)), (C, (0, 0, 0)),
                 (120, (0, 0, 0))],
    # ears: perk at the die, pin back for the flick, track it up, flop on the catch, cock for the smug
    'ear.01.L': [(1, (0, 0, 0)), (14, (0, 0, 0)), (20, (14, 0, -5)), (31, (-12, 0, 8)), (38, (-4, 0, 3)),
                 (48, (9, -6, -9)), (60, (8, -4, -8)), (C, (12, 0, -4)), (C + 4, (4, 0, 0)),
                 (C + 10, (-9, 0, 6)), (C + 18, (0, 0, 0)), (96, (7, 0, 3)), (120, (7, 0, 3))],
    'ear.01.R': [(1, (0, 0, 0)), (14, (0, 0, 0)), (20, (14, 0, 5)), (31, (-12, 0, -8)), (38, (-4, 0, -3)),
                 (48, (9, 6, 9)), (60, (8, 4, 8)), (C, (12, 0, 4)), (C + 4, (4, 0, 0)),
                 (C + 10, (-9, 0, -6)), (C + 18, (0, 0, 0)), (96, (-8, -6, -13)), (120, (-8, -6, -13))],
    # eyelids: blink, widen, focus, squeeze on impact, smug half-lids with a slow blink
    'lid_up.L': [(1, (0, 0, 0)), (7, (0, 0, 0)), (9, (70, 0, 0)), (11, (0, 0, 0)), (18, (-6, 0, 0)),
                 (29, (12, 0, 0)), (33, (14, 0, 0)), (37, (-4, 0, 0)), (C, (-5, 0, 0)), (C + 3, (52, 0, 0)),
                 (C + 8, (0, 0, 0)), (90, (0, 0, 0)), (97, (21, 0, 0)), (105, (21, 0, 0)), (108, (68, 0, 0)),
                 (111, (22, 0, 0)), (120, (21, 0, 0))],
    'lid_low.L': [(1, (0, 0, 0)), (7, (0, 0, 0)), (9, (-12, 0, 0)), (11, (0, 0, 0)), (C + 3, (-16, 0, 0)),
                  (C + 8, (0, 0, 0)), (90, (0, 0, 0)), (97, (-12, 0, 0)), (105, (-12, 0, 0)), (108, (-16, 0, 0)),
                  (111, (-12, 0, 0)), (120, (-12, 0, 0))],
}
ROT['lid_up.R'] = [(f, v) for f, v in ROT['lid_up.L']]
ROT['lid_low.R'] = [(f, v) for f, v in ROT['lid_low.L']]

LOC = {
    'hold_ctrl': [(1, (0, 0, 0)), (10, (0, 0, 0.004)), (22, (0, -0.004, 0.020)), (31, (0, 0.008, -0.012)),
                  (36, (0, -0.022, 0.088)), (40, (0, -0.024, 0.092)), (46, (-0.006, -0.03, 0.052)),
                  (62, (-0.010, -0.03, 0.060)), (C - 2, (-0.012, -0.03, 0.072)), (C, (-0.012, -0.03, 0.068)),
                  (C + 4, (-0.008, -0.020, -0.026)), (C + 10, (-0.004, -0.010, 0.012)),
                  (C + 16, (0.0, -0.006, -0.004)), (C + 22, (0.0, -0.004, 0.002)), (96, (-0.012, -0.016, 0.014)),
                  (120, (-0.012, -0.016, 0.014))],
    'paw_ik.L': [(1, (0, 0, 0)), (37, (0, 0, 0)), (42, (0.026, 0, -0.004)), (60, (0.022, 0, 0)),
                 (C - 3, (0.018, 0, 0)), (C, (0, 0, 0)), (120, (0, 0, 0))],
    'paw_ik.R': [(1, (0, 0, 0)), (37, (0, 0, 0)), (42, (-0.026, 0, -0.004)), (60, (-0.022, 0, 0)),
                 (C - 3, (-0.018, 0, 0)), (C, (0, 0, 0)), (120, (0, 0, 0))],
    'hips': [(1, (0, 0, 0)), (12, (0, 0, 0.002)), (31, (0, 0, -0.008)), (37, (0, 0, 0.008)), (46, (0, 0, 0.002)),
             (C, (0, 0, 0)), (C + 4, (0, -0.004, -0.013)), (C + 10, (0, 0, 0.004)), (C + 16, (0, 0, -0.002)),
             (96, (0, 0, 0)), (120, (0, 0, 0))],
}

# tail: deliberate balance swing (yaw), lift (pitch) - then overlapped down the chain
TAIL = [(1, (0, 0, 0)), (14, (0, 4, 0)), (31, (8, -6, 0)), (37, (-12, 14, 0)), (46, (-6, 8, 0)),
        (60, (-3, 2, 0)), (C, (0, -4, 0)), (C + 3, (-14, -24, 0)), (C + 9, (-6, 16, 0)),
        (C + 16, (-3, -8, 0)), (C + 24, (-2, 3, 0)), (100, (-2, 6, 0)), (120, (-2, 4, 0))]
EAR_FLOP = [(1, (0, 0, 0)), (C, (0, 0, 0)), (C + 3, (15, 0, 0)), (C + 7, (-9, 0, 0)), (C + 12, (4, 0, 0)),
            (C + 18, (0, 0, 0)), (120, (0, 0, 0))]


def ballistic(p0, p1, T, apex_z):
    a = apex_z - p0[2]
    b = apex_z - p1[2]
    t_up = T * math.sqrt(a) / (math.sqrt(a) + math.sqrt(b))
    g = 2 * a / t_up ** 2
    vz = g * t_up

    def pos(t):
        k = t / T
        xy = p0[:2] + (p1[:2] - p0[:2]) * k
        xy[1] -= 0.075 * math.sin(math.pi * k)       # arc out in front of the snout
        xy[0] -= 0.02 * math.sin(math.pi * k)
        return np.array([xy[0], xy[1], p0[2] + vz * t - 0.5 * g * t * t])
    return pos


def bone_world(rig, name):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = rig.evaluated_get(dg)
    return ev.matrix_world @ ev.pose.bones[name].matrix


def animate(ch, st):
    scene = bpy.context.scene
    rig, die, fur = ch['rig'], ch['die'], ch['fur']
    pb = rig.pose.bones
    fur_mod = fur.modifiers[0]
    fur_mod.show_viewport = False            # faster baking
    rig.animation_data_create()

    # ---------------- primary sparse keys
    for bone, keys in ROT.items():
        for f, v in keys:
            key_rot(pb[bone], f, v)
    for bone, keys in LOC.items():
        for f, v in keys:
            key_loc(pb[bone], f, v)
    # the die lands with its six facing out: rotate the socket on the catch frame
    sock = pb['die_socket']
    for f, ang in ((1, 0.0), (CATCH - 1, 0.0), (CATCH, -90.0)):
        key_rot(sock, f, (ang, 0, 0))
    set_interp(rig, 'pose.bones["die_socket"]', 'CONSTANT')

    # ---------------- overlapping action baked from the primary curves
    frames = np.arange(F_START, F_END + 1)
    tail_c = curve(TAIL)
    gains = [0.45, 0.65, 0.85, 1.0, 1.05, 1.0]
    for k in range(6):
        b = pb[f'tail.{k + 1:02d}']
        for f in frames:
            p, y, r = tail_c(f - 1.7 * k) * gains[k]
            y += 3.0 * math.sin(2 * math.pi * f / 46 - 0.55 * k)
            key_rot(b, int(f), (p, y, r))
    flop = curve(EAR_FLOP)
    for sx in 'LR':
        e1 = curve(ROT[f'ear.01.{sx}'])
        b = pb[f'ear.02.{sx}']
        for f in frames:
            p, y, r = 0.55 * e1(f - 2.5) + flop(f - 1.0)
            key_rot(b, int(f), (p, y, r))
    chest = curve(ROT['chest'])
    hold = curve(LOC['hold_ctrl'])
    for tag, ph in (('a', 0.0), ('b', 1.3)):
        for seg, lag, g in (('01', 2.0, 0.6), ('02', 4.0, 1.0)):
            b = pb[f'scarf_{tag}.{seg}']
            for f in frames[::2]:
                cp, cy, cr = chest(f - lag)
                hz = hold(f - lag)[2]
                flutter = 3.0 * math.sin(2 * math.pi * f / 30 + ph + lag)
                key_rot(b, int(f), (-0.6 * cp * g - 120 * hz * g + flutter, 0.3 * cy * g, 2.0 * flutter))

    # ---------------- die flight
    die.rotation_mode = 'QUATERNION'
    scene.frame_set(RELEASE)
    M0 = bone_world(rig, 'die_socket')
    scene.frame_set(CATCH)
    M1 = bone_world(rig, 'die_socket')
    p0, p1 = np.array(M0.translation), np.array(M1.translation)
    q0, q1 = M0.to_quaternion(), M1.to_quaternion()
    dq = q1 @ q0.inverted()
    axis, ang = dq.to_axis_angle()
    if axis.length < 1e-6:
        axis = Vector((1, 0, 0))
    total = ang - 2 * math.pi            # one extra full tumble on the way
    T = CATCH - RELEASE
    pos = ballistic(p0, p1, T, APEX_Z)
    die_track = {}
    for f in range(F_START, F_END + 1):
        if f <= RELEASE:
            loc, rot = Vector(p0), q0
        elif f >= CATCH:
            loc, rot = Vector(p1), q1
        else:
            t = f - RELEASE
            k = t / T
            loc = Vector(pos(t))
            rot = Quaternion(axis, total * k) @ q0
        if RELEASE - 1 <= f <= CATCH + 1 or f in (F_START, F_END):
            die.location = loc
            die.rotation_quaternion = rot
            die.keyframe_insert('location', frame=f)
            die.keyframe_insert('rotation_quaternion', frame=f)
        die_track[f] = np.array(loc)
    con = die.constraints['Held by Pip']
    for f, v in ((F_START, 1.0), (RELEASE, 1.0), (RELEASE + 1, 0.0), (CATCH - 1, 0.0), (CATCH, 1.0)):
        con.influence = v
        con.keyframe_insert('influence', frame=f)
    set_interp(die, 'influence', 'CONSTANT')
    set_interp(die, 'location', 'LINEAR')
    set_interp(die, 'rotation', 'LINEAR')

    # ---------------- eyes: look target follows the die, then finds the camera
    cam = st['cam3q'].matrix_world.translation
    lt = pb['look_target']
    rest = np.array(lt.bone.head_local)
    idle_look = np.array([0.14, -0.75, 0.36])
    for f in range(F_START, F_END + 1):
        if f < 12:
            tgt = idle_look
        elif f < 18:
            k = (f - 12) / 6
            k = k * k * (3 - 2 * k)
            tgt = idle_look * (1 - k) + die_track[18] * k
        elif f <= CATCH + 8:
            tgt = die_track[max(F_START, f - 1)]
            if f > CATCH:          # glance at the caught die in the paws
                scene.frame_set(f)
                tgt = np.array(bone_world(rig, 'die_socket').translation)
        elif f <= 94:
            k = (f - CATCH - 8) / (94 - CATCH - 8)
            k = k * k * (3 - 2 * k)
            scene.frame_set(CATCH + 8)
            a = np.array(bone_world(rig, 'die_socket').translation)
            tgt = a * (1 - k) + np.array(cam) * k
        else:
            tgt = np.array(cam)
        key_loc(lt, f, tuple(tgt - rest))

    # ---------------- finish
    act = rig.animation_data.action
    act.name = 'Pip_LuckyFlick'
    act.use_fake_user = True
    act['frame_range'] = (F_START, F_END)
    die.animation_data.action.name = 'Die_LuckyFlick'
    die.animation_data.action.use_fake_user = True
    turntable_action(rig, die)
    fur_mod.show_viewport = True
    scene.frame_start, scene.frame_end = F_START, F_END
    scene.frame_set(F_START)


def turntable_action(rig, die):
    """A calm hero pose on a 4 s, 360-degree turntable (root rotates)."""
    pb = rig.pose.bones
    main = rig.animation_data.action
    main_slot = rig.animation_data.action_slot
    rig.animation_data.action = None
    for b in pb:
        b.location = (0, 0, 0)
        b.rotation_euler = (0, 0, 0)
    poses = {'head': (-3, 0, 5), 'neck': (-1, 0, 2), 'chest': (-3, 0, 0), 'jaw': (2, 0, 0),
             'ear.01.L': (0, 0, 2), 'ear.01.R': (2, 0, -2), 'tail.03': (0, 4, 0), 'tail.04': (0, 4, 0)}
    for bone, v in poses.items():
        key_rot(pb[bone], 1, v)
    key_loc(pb['look_target'], 1, (0, -0.25, 0.0))
    for f, ang in ((1, 0.0), (97, 360.0)):      # root bone axes == world axes
        pb['root'].rotation_euler = (0, 0, math.radians(ang))
        pb['root'].keyframe_insert('rotation_euler', frame=f, group='root')
    set_interp(rig, 'pose.bones["root"]', 'LINEAR')
    tt = rig.animation_data.action
    tt.name = 'Pip_Turntable'
    tt.use_fake_user = True
    rig.animation_data.action = main
    if main_slot is not None:
        rig.animation_data.action_slot = main_slot
