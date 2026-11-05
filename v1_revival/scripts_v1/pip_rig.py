"""Pip's armature, skin weights and constraints."""
import math
import numpy as np
import bpy
from mathutils import Vector, Matrix

import pip_design as D
import pip_parts as PP


def V(*a):
    return Vector(a if len(a) == 3 else a[0])


def bone_table():
    T = []   # name, head, tail, parent, deform, collection

    def B(n, h, t, p=None, d=True, c='Body'):
        T.append((n, np.array(h, float), np.array(t, float), p, d, c))

    B('root', (0, 0, 0), (0, 0.16, 0), None, False, 'Controls')
    B('hips', (0, 0.020, 0.088), (0, 0.000, 0.140), 'root')
    B('spine', (0, 0.000, 0.140), (0, -0.028, 0.200), 'hips')
    B('chest', (0, -0.028, 0.200), (0, -0.035, 0.246), 'spine')
    B('neck', (0, -0.035, 0.246), (0, -0.040, 0.292), 'chest')
    B('head', (0, -0.040, 0.292), (0, -0.040, 0.480), 'neck')
    B('jaw', D.H((0, -0.092, 0.352)), D.H((0, -0.168, 0.336)), 'head', True, 'Face')
    for s, sx in ((1, 'L'), (-1, 'R')):
        base, A, S, N = D.ear_frame(s)
        mid = base + A * D.EAR_LEN * 0.42
        tip = base + A * D.EAR_LEN
        B(f'ear.01.{sx}', base, mid, 'head', True, 'Face')
        B(f'ear.02.{sx}', mid, tip, f'ear.01.{sx}', True, 'Face')
        ec = D.EYE_C[s]
        g = D.EYE_GAZE[s]
        B(f'eye.{sx}', ec, ec + g * 0.05, 'head', False, 'Face')
        B(f'lid_up.{sx}', ec, ec + g * 0.045, 'head', False, 'Face')
        B(f'lid_low.{sx}', ec, ec + g * 0.04, 'head', False, 'Face')
        sh, el, wr, tip_ = D.arm_joints(s)
        B(f'upper_arm.{sx}', sh, el, 'chest')
        B(f'forearm.{sx}', el, wr, f'upper_arm.{sx}')
        B(f'paw.{sx}', wr, tip_, f'forearm.{sx}')
        hp, kn, an, toe = D.leg_joints(s)
        B(f'thigh.{sx}', hp, kn, 'hips')
        B(f'shin.{sx}', kn, an, f'thigh.{sx}')
        B(f'foot.{sx}', an, toe, f'shin.{sx}')
        # controls
        B(f'paw_ik.{sx}', wr, tip_, 'hold_ctrl', False, 'Controls')
        B(f'foot_ik.{sx}', an, toe, 'root', False, 'Controls')
    for i in range(len(D.TAIL_PTS) - 1):
        B(f'tail.{i + 1:02d}', D.TAIL_PTS[i], D.TAIL_PTS[i + 1],
          'hips' if i == 0 else f'tail.{i:02d}', True, 'Tail')
    for tag, path in (('a', PP.SCARF_TAIL_A), ('b', PP.SCARF_TAIL_B)):
        B(f'scarf_{tag}.01', path[0], path[2], 'chest', True, 'Scarf')
        B(f'scarf_{tag}.02', path[2], path[4], f'scarf_{tag}.01', True, 'Scarf')
    B('hold_ctrl', D.DIE_C, D.DIE_C + np.array([0, 0.07, 0]), 'root', False, 'Controls')
    B('die_socket', D.DIE_C, D.DIE_C + np.array([0, 0.05, 0]), 'hold_ctrl', False, 'Controls')
    B('look_target', (0, -0.60, 0.40), (0, -0.55, 0.40), 'root', False, 'Controls')
    # hold_ctrl must exist before the paw IK bones reference it -> sort
    order = {n: i for i, (n, *_r) in enumerate(T)}
    done, out = set(), []
    while len(out) < len(T):
        for row in T:
            if row[0] in done:
                continue
            if row[3] is None or row[3] in done:
                out.append(row)
                done.add(row[0])
    return out


def widget_meshes(coll):
    def circle(name, r, n=32, axis='Y'):
        v = []
        for i in range(n):
            a = 2 * math.pi * i / n
            x, y = r * math.cos(a), r * math.sin(a)
            v.append((x, 0, y) if axis == 'Y' else (x, y, 0))
        e = [(i, (i + 1) % n) for i in range(n)]
        me = bpy.data.meshes.new(name)
        me.from_pydata(v, e, [])
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        return ob
    return {'circle': circle('WGT_circle', 1.0), 'ring': circle('WGT_ring', 1.0, axis='Z')}


def build_rig(coll, wgt_coll):
    arm = bpy.data.armatures.new('Pip_Rig')
    arm.display_type = 'OCTAHEDRAL'
    rig = bpy.data.objects.new('Pip_Rig', arm)
    coll.objects.link(rig)
    rig.show_in_front = True
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.edit_bones
    table = bone_table()
    for name, h, t, p, d, c in table:
        b = eb.new(name)
        b.head = V(*h)
        b.tail = V(*t)
        b.roll = 0.0
        b.use_deform = d
        if p:
            b.parent = eb[p]
            if (Vector(eb[p].tail) - b.head).length < 1e-5 and p not in ('hold_ctrl',):
                b.use_connect = True
    bpy.ops.object.mode_set(mode='OBJECT')
    # bone collections for a tidy rig
    cols = {}
    for name, h, t, p, d, c in table:
        if c not in cols:
            cols[c] = arm.collections.new(c)
        cols[c].assign(arm.bones[name])
    W = widget_meshes(wgt_coll)
    pb = rig.pose.bones
    for b in pb:
        b.rotation_mode = 'XYZ'
    for name in ('hold_ctrl', 'paw_ik.L', 'paw_ik.R', 'foot_ik.L', 'foot_ik.R', 'look_target', 'root'):
        pb[name].custom_shape = W['ring'] if name == 'root' else W['circle']
    pb['root'].custom_shape_scale_xyz = (2.0, 2.0, 2.0)
    pb['look_target'].custom_shape_scale_xyz = (0.4, 0.4, 0.4)
    pb['hold_ctrl'].custom_shape_scale_xyz = (1.2, 1.2, 1.2)
    for sx in 'LR':
        ik = pb[f'forearm.{sx}'].constraints.new('IK')
        ik.target, ik.subtarget, ik.chain_count = rig, f'paw_ik.{sx}', 2
        cr = pb[f'paw.{sx}'].constraints.new('COPY_ROTATION')
        cr.target, cr.subtarget = rig, f'paw_ik.{sx}'
        ik = pb[f'shin.{sx}'].constraints.new('IK')
        ik.target, ik.subtarget, ik.chain_count = rig, f'foot_ik.{sx}', 2
        cr = pb[f'foot.{sx}'].constraints.new('COPY_ROTATION')
        cr.target, cr.subtarget = rig, f'foot_ik.{sx}'
        dt = pb[f'eye.{sx}'].constraints.new('DAMPED_TRACK')
        dt.target, dt.subtarget, dt.track_axis = rig, 'look_target', 'TRACK_Y'
        for b in (f'eye.{sx}', f'lid_up.{sx}', f'lid_low.{sx}'):
            pb[b].lock_location = (True, True, True)
    # handy custom properties for animators
    rig['pip_version'] = 1
    return rig


# ---------------------------------------------------------------- weights
def compute_weights(co, an, is_ear, ear_side, ear_par, edges, n_body):
    prims = an['prims']
    dist = an['dist']
    tau = 0.0055
    dmin = dist.min(1, keepdims=True)
    memb = np.exp(-(dist - dmin) / tau)
    memb[memb < 0.02] = 0
    bones = sorted({p[1] for p in prims} | {f'ear.0{i}.{s}' for i in (1, 2) for s in 'LR'})
    bidx = {b: i for i, b in enumerate(bones)}
    Wt = np.zeros((len(co), len(bones)))
    for j, p in enumerate(prims):
        Wt[:, bidx[p[1]]] += memb[:, j]
    # torso: share the belly between hips / spine / chest by height
    z = co[:, 2]
    sp = bidx['spine']
    belly = Wt[:, sp].copy()
    lo = np.clip((0.15 - z) / 0.05, 0, 1)
    hi = np.clip((z - 0.20) / 0.05, 0, 1)
    Wt[:, sp] = belly * (1 - lo - hi).clip(0, 1)
    Wt[:, bidx['hips']] += belly * lo
    Wt[:, bidx['chest']] += belly * hi
    # ears
    for s, sx in ((1, 'L'), (-1, 'R')):
        m = is_ear & (ear_side == s)
        u = ear_par[m, 0]
        w2 = np.clip((u - 0.30) / 0.30, 0, 1)
        wh = np.clip((0.06 - u) / 0.12, 0, 1)
        Wt[m] = 0
        Wt[m, bidx[f'ear.02.{sx}']] = w2
        Wt[m, bidx[f'ear.01.{sx}']] = (1 - w2) * (1 - wh)
        Wt[m, bidx['head']] = wh
    Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-9)
    # smooth over the body surface (not the ear shells)
    import scipy.sparse as sps
    e = edges[(edges[:, 0] < n_body) & (edges[:, 1] < n_body)]
    n = len(co)
    A = sps.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                       shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    body = np.arange(n) < n_body
    for _ in range(12):
        avg = (A @ Wt) / np.maximum(deg, 1)[:, None]
        Wt[body] = 0.5 * Wt[body] + 0.5 * avg[body]
    # keep 4 strongest
    idx = np.argsort(-Wt, axis=1)[:, 4:]
    np.put_along_axis(Wt, idx, 0, axis=1)
    Wt[Wt < 0.01] = 0
    Wt /= np.maximum(Wt.sum(1, keepdims=True), 1e-9)
    return bones, Wt


def assign_weights(ob, bones, Wt):
    for j, b in enumerate(bones):
        vg = ob.vertex_groups.new(name=b)
        col = Wt[:, j]
        nz = np.nonzero(col)[0]
        # group identical weights to cut down API calls
        q = np.round(col[nz], 3)
        for val in np.unique(q):
            ids = nz[q == val].tolist()
            vg.add(ids, float(val), 'REPLACE')


def bone_parent(ob, rig, bone):
    mw = ob.matrix_world.copy()
    ob.parent = rig
    ob.parent_type = 'BONE'
    ob.parent_bone = bone
    bpy.context.view_layer.update()
    ob.matrix_world = mw
