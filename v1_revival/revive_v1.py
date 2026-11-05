"""Revive Pip v1 (commit cea169d, blender/pip.blend) toward the four-pose reference.

    python revive_v1.py  ../blender/pip.blend  Pip_v1_work.blend      (Blender 5.2 bpy)

The v1 body mesh, ears, eyes, face parts, scarf mesh and rig are kept exactly.
Only these change, in the copy:
  * fur re-groomed with v1's own colour/groom fields plus a swept fringe and
    broad, tufty cheeks (chunkier clumps), same strand count as v1
  * colour management: Khronos PBR Neutral (keeps the art's blues, teal and gold)
  * gold die scaled up to the reference proportion; paws widened to grip it
  * 'Pip_RefPose' action (the reference's top-left pose) + 'Cam_Reference'
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'scripts_v1'))     # v1's own modules, not today's

import numpy as np
import bpy
from mathutils import Vector, Matrix
from scipy.spatial import cKDTree

import pip_design as D
import pip_mesh as PM
import pip_colors as PC
import pip_fur as PF
from pip_anim import key_rot, key_loc

args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
SRC, DST = args[0], args[1]
DIE_SCALE = 1.40
N_STRANDS = 160000

bpy.ops.wm.open_mainfile(filepath=SRC)
scene = bpy.context.scene
body = bpy.data.objects['Pip_Body']
fur = bpy.data.objects['Pip_Fur']
rig = bpy.data.objects['Pip_Rig']
die = bpy.data.objects['Gold_Die']
scarf = bpy.data.objects['Pip_Scarf']

# ----------------------------------------------------------------- groom fields (rest mesh)
me = body.data
n_all = len(me.vertices)
n_ear = sum(len(D.ear_mesh(s, nu=96, nv=64)[0]) for s in (1, -1))
n_body = n_all - n_ear
co = np.empty(n_all * 3)
me.vertices.foreach_get('co', co)
co = co.reshape(-1, 3)
nrm = np.empty(n_all * 3)
me.vertices.foreach_get('normal', nrm)
nrm = nrm.reshape(-1, 3)
an = PM.vertex_analysis(co)
lab, is_ear = PC.classify(an, n_body)
ear_side = np.zeros(n_all, int)
ear_par = np.zeros((n_all, 3))
off = n_body
for s in (1, -1):
    ev, ef, par = D.ear_mesh(s, nu=96, nv=64)
    ear_side[off:off + len(ev)] = s
    ear_par[off:off + len(ev)] = par
    off += len(ev)
ear_par[:, 0] = np.clip(ear_par[:, 0], 0, 1)
F = PC.fields(co, nrm, an, lab, is_ear, ear_side, ear_par, np.random.default_rng(3))
sv = np.array([v.co[:] for v in scarf.data.vertices])
dsc, _ = cKDTree(sv).query(co)
F['density'][dsc < 0.010] = 0
F['length'] = np.where(dsc < 0.03, np.minimum(F['length'], 0.012 + 0.6 * np.maximum(dsc - 0.01, 0)), F['length'])
reach = np.ones(n_all)

up, back = np.array([0, 0, 1.0]), np.array([0, 1.0, 0])
hm = np.isin(lab, ['head', 'crown']) & ~is_ear
# swept fringe: long tufts from the crown sweeping over the forehead toward his right
zc = D.H([0, 0, 0.44])[2]
fr = hm & (co[:, 2] > zc) & (nrm[:, 1] < 0.55)
w = PC.sstep(zc, zc + 0.05, co[:, 2]) * (1 - PC.sstep(0.05, 0.10, np.abs(co[:, 0] + 0.01)))
F['length'][fr] = 0.034 + 0.062 * w[fr]
F['lift'][fr] = 0.74
F['clump'][fr] = 0.97
reach[fr] = 3.4
F['comb'][fr] = PC.tangent(np.tile(np.array([-0.85, -0.45, 0.65]), (fr.sum(), 1))
                           + np.random.default_rng(11).normal(0, 0.25, (fr.sum(), 3)), nrm[fr])
# broad, fluffy cheek ruff: cheek lobes plus the lower sides of the head
ck = (lab == 'cheek') | (hm & (np.abs(co[:, 0]) > 0.085) & (co[:, 2] < D.EYE_C[1][2] + 0.005)
                         & (co[:, 1] < -0.02))
ck &= ~is_ear
out = np.c_[np.sign(co[ck, 0]), np.zeros(ck.sum()), np.zeros(ck.sum())]
F['comb'][ck] = PC.tangent(out * 1.0 - up * 0.45 + back * 0.25, nrm[ck])
F['length'][ck] = np.maximum(F['length'][ck], 0.062)
F['lift'][ck] = 0.42
F['clump'][ck] = 0.95
reach[ck] = 2.6
front = ck & (nrm[:, 1] < -0.65)                 # keep the face readable
F['length'][front] = 0.026
# chest bib a little fuller, tail tufts chunkier
ch = np.isin(lab, ['chest', 'neck']) & ~is_ear
F['length'][ch] = 0.038
reach[ch] = 1.8
reach[(lab == 'tail') & ~is_ear] = 1.8


# ----------------------------------------------------------------- strands (v1 shape, chunky tufts)
def strands(R, rng, K=PF.K):
    unit = PC.unit
    n = len(R['co'])
    L = R['length'] * rng.uniform(0.8, 1.15, n)
    nrm_, comb = R['nrm'], R['comb']
    lift = np.clip(R['lift'] + rng.normal(0, 0.06, n), 0.05, 0.95)
    jitter = unit(rng.normal(0, 1, (n, 3)))
    jitter = unit(jitter - nrm_ * (jitter * nrm_).sum(1, keepdims=True))
    comb = unit(comb + jitter * 0.12)
    d0 = unit(nrm_ * lift[:, None] + comb * (1 - lift)[:, None])
    d1 = unit(nrm_ * (0.25 * lift)[:, None] + comb + np.array([0, 0, -0.15]))
    t = np.linspace(0, 1, K)
    P = np.zeros((n, K, 3))
    P[:, 0] = R['co'] - nrm_ * 0.0008
    for k in range(1, K):
        tk = (k - 0.5) / (K - 1)
        P[:, k] = P[:, k - 1] + unit(d0 * (1 - tk) + d1 * tk) * (L / (K - 1))[:, None]
    rch = R['reach']
    lvl = np.clip(np.round(rch * 2) / 2, 0.5, 4.0)
    cidx = np.arange(n)
    dist = np.zeros(n)
    for lv in np.unique(lvl):
        ids = np.nonzero(lvl == lv)[0]
        cs = 0.0045 * lv
        key = np.floor((R['co'][ids] + rng.uniform(0, cs, 3)) / cs).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        cent = ids[first]
        dd, ci = cKDTree(R['co'][cent]).query(R['co'][ids])
        cidx[ids] = cent[ci]
        dist[ids] = dd
    rr = (0.004 + 0.12 * L) * rch + 0.0045 * lvl
    amt = R['clump'] * np.clip(1 - dist / rr, 0, 1)
    pull = amt[:, None, None] * (t ** 1.4)[None, :, None]
    P = P + pull * (P[cidx] - P) * 0.92
    ph = rng.uniform(0, 6.28, n)
    wav = np.sin(t[None, :] * 3.0 + ph[:, None]) * (0.035 * L)[:, None] * t[None, :]
    return P + jitter[:, None, :] * wav[..., None]


rng = np.random.default_rng(7)
vattr = dict(co=co, nrm=nrm, color=F['color'], length=F['length'], lift=F['lift'], comb=F['comb'],
             clump=F['clump'], density=F['density'], reach=reach)
R = PF.sample_roots(me, vattr, N_STRANDS, rng)
P = strands(R, rng)
K = PF.K
cv = bpy.data.hair_curves.new('Pip_Fur_v1r')
cv.add_curves([K] * len(P))
cv.position_data.foreach_set('vector', P.reshape(-1).astype(np.float32))
rad = cv.attributes.new('radius', 'FLOAT', 'POINT')
t = np.linspace(0, 1, K)
rr = 0.00085 * (1 - t) ** 0.9 + 0.00012
scale = np.clip(R['length'] / 0.022, 0.7, 1.35) ** 0.5
rad.data.foreach_set('value', (rr[None, :] * scale[:, None]).reshape(-1).astype(np.float32))
col = cv.attributes.new('fur_color', 'FLOAT_COLOR', 'CURVE')
c4 = np.c_[R['color'] * rng.uniform(0.94, 1.04, (len(P), 1)), np.ones(len(P))]
col.data.foreach_set('color', np.clip(c4, 0, 1).reshape(-1).astype(np.float32))
suv = cv.attributes.new('surface_uv_coordinate', 'FLOAT2', 'CURVE')
suv.data.foreach_set('vector', R['uv'].reshape(-1).astype(np.float32))
cv.surface = body
cv.surface_uv_map = me.uv_layers.active.name
for m in fur.data.materials:
    cv.materials.append(m)
old = fur.data
fur.data = cv
old.use_fake_user = False
print('[v1r] fur regroomed', len(P))

# ----------------------------------------------------------------- colour management
scene.view_settings.view_transform = 'Khronos PBR Neutral'
scene.view_settings.look = 'None'
scene.view_settings.exposure = -0.15

# ----------------------------------------------------------------- bigger, glossier gold die
gm = bpy.data.materials.get('Gold_Die')
if gm:
    bsdf = next(n for n in gm.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = (0.86, 0.36, 0.008, 1)
    bsdf.inputs['Metallic'].default_value = 0.75
    bsdf.inputs['Coat Weight'].default_value = 0.55
dm = die.data
dm.transform(Matrix.Scale(DIE_SCALE, 4))
dm.update()

# ----------------------------------------------------------------- reference pose + camera
rig.animation_data.action = None
for b in rig.pose.bones:
    b.location = (0, 0, 0)
    b.rotation_euler = (0, 0, 0)
pb = rig.pose.bones
half = D.DIE_EDGE * DIE_SCALE / 2
spread = half - D.DIE_EDGE / 2 + 0.004
pose = {'head': (-3, -6, -6), 'neck': (-1, -3, -2), 'chest': (-3, -2, 0), 'jaw': (3, 0, 0),
        'ear.01.L': (0, 0, -3), 'ear.01.R': (2, 0, 3), 'tail.03': (0, 4, 0), 'tail.04': (0, 4, 0)}
for b, v in pose.items():
    key_rot(pb[b], 1, v)
key_loc(pb['hold_ctrl'], 1, (0.008, -0.022, 0.012))
key_loc(pb['paw_ik.L'], 1, (spread, 0.004, 0.012))
key_loc(pb['paw_ik.R'], 1, (-spread, 0.004, 0.006))
key_rot(pb['hold_ctrl'], 1, (-4, -10, 4))
cd = bpy.data.cameras.new('Cam_Reference')
cd.lens = 85
cam = bpy.data.objects.new('Cam_Reference', cd)
for c in bpy.data.collections:
    if c.name == 'Stage':
        c.objects.link(cam)
tgt = Vector((-0.03, -0.05, 0.30))
az, el, d = math.radians(28), math.radians(9), 1.95
cam.location = tgt + Vector((-math.sin(az) * math.cos(el) * d, -math.cos(az) * math.cos(el) * d, math.sin(el) * d))
cam.rotation_euler = (tgt - cam.location).to_track_quat('-Z', 'Y').to_euler()
# eyes to camera
key_loc(pb['look_target'], 1, tuple(np.array(cam.location) - np.array(pb['look_target'].bone.head_local)))
act = rig.animation_data.action
act.name = 'Pip_RefPose'
act.use_fake_user = True
if die.animation_data:
    die.animation_data.action = None
die.constraints['Held by Pip'].influence = 1.0
scene.camera = cam
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=DST, compress=True)
print('[v1r] saved', DST)
