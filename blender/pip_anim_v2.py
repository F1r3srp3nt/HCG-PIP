"""Pip v2 actions.

* Pip_Likeness  - one-frame pose matching the reference sheet (grin, die held).
* Pip_Rest      - one-frame relaxed pose (closed mouth) for comparison.
* Pip_NearMiss  - the 5 s locked-camera performance (see animate_near_miss).

Rotation values are degrees about *world* axes in the rest pose:
pitch (+ = nod forward/down), yaw (+ = turn toward Pip's left), roll (+ = tilt top toward Pip's left).
Locations are world-axis offsets in metres.
"""
import math
import numpy as np
import bpy

from pip_anim import curve, key_rot, key_loc, set_interp, bone_world, fcurves  # noqa: F401


def clear_pose(rig):
    for b in rig.pose.bones:
        b.location = (0, 0, 0)
        b.rotation_euler = (0, 0, 0)
        b.scale = (1, 1, 1)


GRIN = {'lid_low.L': (-10, 0, 0), 'lid_low.R': (-10, 0, 0)}


def new_action(rig, name):
    rig.animation_data_create()
    rig.animation_data.action = None
    clear_pose(rig)
    return name


def finish_action(rig, name):
    act = rig.animation_data.action
    act.name = name
    act.use_fake_user = True
    return act


def key_grin(rig, f, v):
    rig['grin'] = float(v)
    rig.keyframe_insert('["grin"]', frame=f, group='face')


def static_pose(rig, name, rot, loc, grin=0.0):
    new_action(rig, name)
    pb = rig.pose.bones
    for b, v in rot.items():
        key_rot(pb[b], 1, v)
    for b, v in loc.items():
        key_loc(pb[b], 1, v)
    key_loc(pb['look_target'], 1, (0, -0.25, 0))
    key_grin(rig, 1, grin)
    return finish_action(rig, name)


def likeness_actions(rig):
    pose = {'head': (-2, -4, -4), 'neck': (0, -2, -1), 'chest': (-2, 0, 0),
            'ear.01.L': (0, 0, 0), 'ear.01.R': (0, 0, 0), 'tail.03': (0, 4, 0)}
    a_rest = static_pose(rig, 'Pip_Rest', pose, {})
    a_like = static_pose(rig, 'Pip_Likeness', {**pose, **GRIN}, {}, grin=1.0)
    return a_like, a_rest


# ====================================================================== Near Miss
# 5 s @ 24 fps, camera locked. Beats:
#   1-28   Pip looks down at the gold die between his paws (blink at 12)
#   28-34  anticipation: paws dip, elbows bend, ears pin back
#   36     RELEASE: paws stop at the top of the flick, the die leaves both paws
#   37-54  the die tumbles just above his nose; eyes, head and ears track it
#   55     CATCH, a little off-centre and tilted; elbows fold under the weight
#   56-64  the near drop: the die slides toward his left paw, he leans after it,
#          the tail swings out to steady him, then he hauls it back to centre
#   66-86  hold the beat (relieved exhale, blink)
#   87     the GLANCE: eyes snap to camera, ears perk - he knows you saw
#   94-120 small, pleased grin
F0, F1 = 1, 120
RELEASE, CATCH = 36, 55
APEX_Z = 0.432          # die centre at the top of the toss: just above the nose
GLANCE = 87

NM_ROT = {
    'spine': [(1, (0, 0, 0)), (26, (4, 0, 0)), (31, (6, 0, 0)), (36, (-5, 0, 0)), (46, (-3, 0, 0)),
              (CATCH, (4, 0, 0)), (59, (9, 0, 5)), (64, (2, 0, -2)), (70, (1, 0, 1)), (80, (0, 0, 0)),
              (GLANCE, (0, 0, 0)), (94, (-1, -3, -1)), (120, (-1, -4, -1))],
    'chest': [(1, (2, 0, 0)), (26, (6, 0, 0)), (31, (9, 0, 0)), (36, (-8, 0, 0)), (46, (-6, 0, 0)),
              (CATCH, (6, 0, 0)), (59, (12, 0, 7)), (64, (3, 0, -3)), (70, (2, 0, 1)), (78, (3, 0, 0)),
              (82, (1, 0, 0)), (GLANCE, (1, 0, 0)), (94, (-2, -5, -2)), (120, (-2, -6, -2))],
    'neck': [(1, (3, 0, 0)), (31, (5, 0, 0)), (36, (0, 0, 0)), (44, (-5, 0, 0)), (50, (-2, 0, 0)),
             (CATCH, (5, 0, 0)), (60, (7, 0, 2)), (70, (5, 0, 0)), (GLANCE, (5, 0, 0)), (92, (0, -6, -3)),
             (120, (-1, -7, -3))],
    'head': [(1, (8, 0, 3)), (14, (9, 0, 4)), (31, (11, 0, 2)), (36, (4, 0, 0)), (42, (-8, 0, -2)),
             (47, (-11, 0, -3)), (52, (-2, 1, 0)), (CATCH, (8, 1, 2)), (60, (14, 2, 6)), (66, (10, 0, 2)),
             (72, (11, 0, 2)), (GLANCE - 1, (11, 0, 2)), (GLANCE + 2, (8, -4, 0)), (93, (0, -12, -5)),
             (100, (-2, -13, -7)), (104, (1, -13, -7)), (108, (-2, -13, -7)), (120, (-2, -13, -7))],
    'hold_ctrl': [(1, (0, 0, 0)), (20, (-2, 0, 0)), (30, (6, 0, 0)), (36, (-10, 0, 0)), (44, (-4, 0, 0)),
                  (CATCH, (0, 0, 0)), (59, (8, 0, 6)), (64, (2, 0, -2)), (70, (0, 0, 0)), (120, (0, 0, 0))],
    'die_socket': [(1, (0, 0, 0)), (CATCH - 1, (0, 0, 0)), (CATCH, (0, 0, 14)), (59, (6, 0, 24)),
                   (63, (2, 0, 10)), (70, (0, 0, 2)), (76, (0, 0, 0)), (120, (0, 0, 0))],
    'paw_ik.L': [(1, (0, 0, 0)), (30, (8, 0, 0)), (36, (-24, 0, 0)), (42, (-6, 0, 0)), (CATCH, (0, 0, 0)),
                 (59, (10, 0, 8)), (66, (0, 0, 0)), (120, (0, 0, 0))],
    'paw_ik.R': [(1, (0, 0, 0)), (30, (8, 0, 0)), (36, (-24, 0, 0)), (42, (-6, 0, 0)), (CATCH, (0, 0, 0)),
                 (59, (-8, 0, 10)), (66, (0, 0, 0)), (120, (0, 0, 0))],
    'ear.01.L': [(1, (4, 0, 0)), (20, (8, 0, -2)), (30, (-6, 0, 4)), (38, (-2, 0, -2)), (46, (6, 0, -6)),
                 (CATCH, (10, 0, 0)), (59, (2, 0, 7)), (66, (-4, 0, 3)), (74, (2, 0, 0)), (GLANCE, (2, 0, 0)),
                 (GLANCE + 2, (-5, 0, -6)), (96, (0, 0, 4)), (110, (0, 0, 4)), (113, (7, 0, 9)),
                 (116, (0, 0, 4)), (120, (0, 0, 4))],
    'ear.01.R': [(1, (4, 0, 0)), (20, (8, 0, 2)), (30, (-6, 0, -4)), (38, (-2, 0, 2)), (46, (6, 0, 6)),
                 (CATCH, (10, 0, 0)), (59, (2, 0, -7)), (66, (-4, 0, -3)), (74, (2, 0, 0)),
                 (GLANCE, (2, 0, 0)), (GLANCE + 2, (-5, 0, 6)), (96, (0, 0, -4)), (120, (0, 0, -4))],
    'lid_up.L': [(1, (2, 0, 0)), (10, (2, 0, 0)), (12, (70, 0, 0)), (14, (2, 0, 0)), (26, (4, 0, 0)),
                 (32, (8, 0, 0)), (36, (-4, 0, 0)), (46, (-6, 0, 0)), (54, (-2, 0, 0)), (56, (34, 0, 0)),
                 (58, (-8, 0, 0)), (62, (-6, 0, 0)), (70, (6, 0, 0)), (76, (12, 0, 0)), (79, (70, 0, 0)),
                 (82, (8, 0, 0)), (GLANCE - 1, (6, 0, 0)), (GLANCE + 1, (-8, 0, 0)), (92, (-4, 0, 0)),
                 (98, (12, 0, 0)), (104, (13, 0, 0)), (120, (13, 0, 0))],
    'lid_low.L': [(1, (0, 0, 0)), (10, (0, 0, 0)), (12, (-8, 0, 0)), (14, (0, 0, 0)), (54, (0, 0, 0)),
                  (56, (-14, 0, 0)), (58, (0, 0, 0)), (76, (0, 0, 0)), (79, (-10, 0, 0)), (82, (0, 0, 0)),
                  (92, (0, 0, 0)), (98, (-11, 0, 0)), (120, (-11, 0, 0))],
}
NM_ROT['lid_up.R'] = list(NM_ROT['lid_up.L'])
NM_ROT['lid_low.R'] = list(NM_ROT['lid_low.L'])

NM_LOC = {
    'hold_ctrl': [(1, (0, 0, 0)), (12, (0, 0, 0.004)), (20, (0, 0.004, -0.004)), (30, (0, 0.006, -0.024)),
                  (33, (0, -0.004, 0.010)), (RELEASE, (0, -0.016, 0.052)), (40, (0, -0.020, 0.036)),
                  (46, (0.004, -0.018, 0.028)), (51, (0.020, -0.018, 0.040)), (CATCH, (0.034, -0.015, 0.050)),
                  (59, (0.040, -0.012, 0.004)), (63, (0.030, -0.010, 0.018)), (68, (0.012, -0.006, 0.010)),
                  (74, (0.004, -0.004, 0.004)), (GLANCE, (0.004, -0.004, 0.004)), (100, (-0.004, -0.010, 0.010)),
                  (120, (-0.004, -0.010, 0.010))],
    'die_socket': [(1, (0, 0, 0)), (CATCH - 1, (0, 0, 0)), (CATCH, (0.006, 0, 0.004)), (59, (0.018, 0, -0.006)),
                   (63, (0.010, 0, -0.002)), (70, (0.002, 0, 0)), (76, (0, 0, 0)), (120, (0, 0, 0))],
    'paw_ik.L': [(1, (0, 0, 0)), (RELEASE, (0, 0, 0)), (39, (0.024, 0, -0.012)), (50, (0.016, 0, -0.004)),
                 (CATCH - 1, (0.004, 0, 0)), (CATCH, (0, 0, 0)), (59, (0.012, 0, -0.014)), (66, (0, 0, 0)),
                 (120, (0, 0, 0))],
    'paw_ik.R': [(1, (0, 0, 0)), (RELEASE, (0, 0, 0)), (39, (-0.024, 0, -0.012)), (50, (-0.016, 0, -0.004)),
                 (CATCH - 1, (-0.004, 0, 0)), (CATCH, (0, 0, 0)), (59, (0.010, -0.004, 0.014)), (66, (0, 0, 0)),
                 (120, (0, 0, 0))],
    'hips': [(1, (0, 0, 0)), (30, (0, 0, -0.006)), (RELEASE, (0, 0, 0.006)), (46, (0, 0, 0)), (CATCH, (0, 0, 0)),
             (59, (0.004, -0.004, -0.012)), (64, (0, 0, 0.003)), (70, (0, 0, 0)), (120, (0, 0, 0))],
}
NM_TAIL = [(1, (0, 0, 0)), (30, (4, -4, 0)), (RELEASE, (-4, 6, 0)), (46, (0, 2, 0)), (CATCH, (0, 0, 0)),
           (58, (10, 24, 0)), (62, (4, -14, 0)), (67, (2, 8, 0)), (73, (0, -4, 0)), (80, (0, 2, 0)),
           (88, (0, 0, 0)), (120, (0, 0, 0))]
NM_EAR_FLOP = [(1, (0, 0, 0)), (CATCH, (0, 0, 0)), (57, (9, 0, 0)), (61, (-5, 0, 0)), (65, (2, 0, 0)),
               (70, (0, 0, 0)), (120, (0, 0, 0))]
NM_GRIN = [(1, 0.0), (94, 0.0), (101, 0.62), (120, 0.66)]


def ballistic(p0, p1, T, apex_z):
    a = apex_z - p0[2]
    b = apex_z - p1[2]
    t_up = T * math.sqrt(a) / (math.sqrt(a) + math.sqrt(b))
    g = 2 * a / t_up ** 2
    vz = g * t_up

    def pos(t):
        k = t / T
        xy = p0[:2] + (p1[:2] - p0[:2]) * k
        xy[1] -= 0.045 * math.sin(math.pi * k)       # arcs out in front of his nose
        return np.array([xy[0], xy[1], p0[2] + vz * t - 0.5 * g * t * t])
    return pos


def animate_near_miss(ch, st):
    from mathutils import Vector, Quaternion
    scene = bpy.context.scene
    rig, die, fur = ch['rig'], ch['die'], ch['fur']
    pb = rig.pose.bones
    fur_mod = fur.modifiers[0]
    fur_mod.show_viewport = False
    new_action(rig, 'Pip_NearMiss')

    for bone, keys in NM_ROT.items():
        for f, v in keys:
            key_rot(pb[bone], f, v)
    for bone, keys in NM_LOC.items():
        for f, v in keys:
            key_loc(pb[bone], f, v)
    for f, v in NM_GRIN:
        key_grin(rig, f, v)
    # the die snaps onto the socket at the catch: no easing into the catch pose
    set_interp(rig, 'pose.bones["die_socket"]', 'BEZIER')

    frames = np.arange(F0, F1 + 1)
    tail_c = curve(NM_TAIL)
    gains = [0.45, 0.65, 0.85, 1.0, 1.05, 1.0]
    for k in range(6):
        b = pb[f'tail.{k + 1:02d}']
        for f in frames:
            p, y, r = tail_c(f - 1.7 * k) * gains[k]
            y += 2.5 * math.sin(2 * math.pi * f / 50 - 0.55 * k)
            key_rot(b, int(f), (p, y, r))
    flop = curve(NM_EAR_FLOP)
    for sx in 'LR':
        e1 = curve(NM_ROT[f'ear.01.{sx}'])
        b = pb[f'ear.02.{sx}']
        for f in frames:
            key_rot(b, int(f), tuple(0.55 * e1(f - 2.5) + flop(f - 1.0)))
    chest = curve(NM_ROT['chest'])
    hold = curve(NM_LOC['hold_ctrl'])
    for tag, ph in (('a', 0.0), ('b', 1.3)):
        for seg, lag, g in (('01', 2.0, 0.6), ('02', 4.0, 1.0)):
            b = pb[f'scarf_{tag}.{seg}']
            for f in frames[::2]:
                cp, cy, cr = chest(f - lag)
                hz = hold(f - lag)[2]
                flutter = 2.5 * math.sin(2 * math.pi * f / 30 + ph + lag)
                key_rot(b, int(f), (-0.6 * cp * g - 90 * hz * g + flutter, 0.3 * cy * g, 0.5 * cr * g + 2 * flutter))

    # ---------------- the die: held -> free flight -> caught
    die.animation_data_create()
    die.animation_data.action = None
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
    total = ang - 2 * math.pi            # one full tumble on the way up and down
    T = CATCH - RELEASE
    pos = ballistic(p0, p1, T, APEX_Z)
    track = {}
    for f in range(F0, F1 + 1):
        if f <= RELEASE or f >= CATCH:
            scene.frame_set(f)
            M = bone_world(rig, 'die_socket')
            loc, rot = M.translation.copy(), M.to_quaternion()
        else:
            t = f - RELEASE
            loc = Vector(pos(t))
            rot = Quaternion(axis, total * (t / T)) @ q0
        if RELEASE - 1 <= f <= CATCH + 1 or f in (F0, F1):
            die.location = loc
            die.rotation_quaternion = rot
            die.keyframe_insert('location', frame=f)
            die.keyframe_insert('rotation_quaternion', frame=f)
        track[f] = np.array(loc)
    con = die.constraints['Held by Pip']
    for f, v in ((F0, 1.0), (RELEASE, 1.0), (RELEASE + 1, 0.0), (CATCH - 1, 0.0), (CATCH, 1.0)):
        con.influence = v
        con.keyframe_insert('influence', frame=f)
    set_interp(die, 'influence', 'CONSTANT')
    set_interp(die, 'location', 'LINEAR')
    set_interp(die, 'rotation', 'LINEAR')
    da = die.animation_data.action
    da.name = 'Die_NearMiss'
    da.use_fake_user = True

    # ---------------- eyes: on the die, tracking it in the air, then on us
    cam = np.array(st['cam_nm'].matrix_world.translation)
    lt = pb['look_target']
    rest = np.array(lt.bone.head_local)
    for f in range(F0, F1 + 1):
        if f < GLANCE:
            src = track[max(F0, f - 1)] if RELEASE < f <= CATCH + 1 else track[f]
            tgt = src + np.array([0, 0, 0.02])
        elif f < GLANCE + 2:
            k = (f - GLANCE + 1) / 3
            tgt = track[f] * (1 - k) + cam * k
        else:
            tgt = cam
        key_loc(lt, f, tuple(tgt - rest))

    act = finish_action(rig, 'Pip_NearMiss')
    act['frame_range'] = (F0, F1)
    fur_mod.show_viewport = True
    scene.frame_start, scene.frame_end = F0, F1
    scene.camera = st['cam_nm']
    scene.frame_set(F0)
    return act
