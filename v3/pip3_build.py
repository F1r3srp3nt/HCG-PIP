"""Build Pip v3 as a fresh Blender scene from reference/pip_v3_target.png.

    python pip3_build.py [--strands 220000] [--out pip_v3.blend]       (Blender 5.2 bpy)

Nothing from the earlier Pip models is used: the shapes, colours and face come
from the v3 reference only (see pip3_design.py).
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler
from mathutils.bvhtree import BVHTree
from scipy.spatial import cKDTree
from skimage import measure
from PIL import Image, ImageDraw, ImageFilter

import pip3_design as D

REF = os.path.join(os.path.dirname(HERE), 'reference', 'pip_v3_target.png')


def log(*a):
    print('[pip3]', *a, flush=True)


def hexlin(h):
    h = h.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def tangent(v, n):
    return unit(v - n * (v * n).sum(-1, keepdims=True))


def to_img(P):
    q = np.asarray(P) - D.ORIGIN
    return np.stack([q @ D.RIGHT / D.K + D.IMG / 2, D.IMG / 2 - q @ D.UP / D.K], -1)


def mesh_from(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    me.validate()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return me


def link(ob, coll):
    coll.objects.link(ob)
    return ob


# ====================================================================== body mesh
def body_mesh(coll):
    res = 0.003
    us = np.arange(-70, D.IMG + 70, res / D.K)
    vs = np.arange(-60, D.IMG + 60, res / D.K)
    ds = np.arange(-0.36, 0.42, res)
    U, V, W = np.meshgrid(us, vs, ds, indexing='ij')
    P = D.cam_pt(U.ravel(), V.ravel(), W.ravel())
    d = np.empty(len(P))
    for i in range(0, len(P), 1_500_000):
        d[i:i + 1_500_000] = D.body_sdf(P[i:i + 1_500_000])
    v, f, _, _ = measure.marching_cubes(d.reshape(U.shape), 0.0)
    uu = np.interp(v[:, 0], np.arange(len(us)), us)
    vv = np.interp(v[:, 1], np.arange(len(vs)), vs)
    dd = np.interp(v[:, 2], np.arange(len(ds)), ds)
    verts = D.cam_pt(uu, vv, dd)
    tmp = link(bpy.data.objects.new('tmp', mesh_from('tmp', verts, f[:, ::-1])), coll)
    m = tmp.modifiers.new('r', 'REMESH')
    m.mode, m.voxel_size, m.adaptivity = 'VOXEL', 0.0032, 0.0
    s = tmp.modifiers.new('s', 'LAPLACIANSMOOTH')
    s.iterations, s.lambda_factor = 2, 0.4
    dg = bpy.context.evaluated_depsgraph_get()
    body = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    old = tmp.data
    bpy.data.objects.remove(tmp)
    bpy.data.meshes.remove(old)
    n_body = len(body.vertices)
    bm = bmesh.new()
    bm.from_mesh(body)
    ear_info = []
    for key in ('R', 'L'):
        ev, ef, par = D.ear_mesh(D.EARS[key])
        em = mesh_from('e', ev, ef)
        bm.from_mesh(em)
        bpy.data.meshes.remove(em)
        ear_info.append((key, len(ev), par))
    final = bpy.data.meshes.new('Pip3_Body')
    bm.to_mesh(final)
    bm.free()
    bpy.data.meshes.remove(body)
    for p in final.polygons:
        p.use_smooth = True
    return final, n_body, ear_info


# ====================================================================== paint + groom
CREAM = hexlin('#F1EEEA')
BLUE_L = hexlin('#BCC8E6')
BLUE = hexlin('#93A6D6')
BLUE_D = hexlin('#5E79C2')
EAR_IN = hexlin('#566DAE')
EAR_IN_L = hexlin('#93A6D8')
TEAL = hexlin('#2FB6CF')
TEAL_L = hexlin('#7EE0E8')
CYAN_TUFT = hexlin('#7FD3EA')


def stroke_mask(uv, pts, width):
    """Distance-based mask to an image-space polyline (pixels)."""
    best = np.full(len(uv), 1e9)
    pts = np.asarray(pts, float)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(((uv - a) @ ab) / (ab @ ab), 0, 1)
        dd = np.linalg.norm(uv - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, dd)
    return 1 - sstep(width * 0.5, width, best)


def fields(co, nrm, labels, is_ear, ear_par, ear_key, rng):
    N = len(co)
    uv = to_img(co)
    front = nrm @ (-D.FWD)                         # > 0: faces the camera
    lab = labels
    col = np.tile(CREAM, (N, 1))
    length = np.full(N, 0.011)
    lift = np.full(N, 0.30)
    clump = np.full(N, 0.70)
    reach = np.full(N, 1.5)
    dens = np.ones(N)
    up = np.array([0, 0, 1.0])
    comb = tangent(np.tile(-up + 0.3 * D.FWD, (N, 1)), nrm)

    def L(*names):
        m = np.isin(lab, names)
        return m & ~is_ear

    # ---- general blue-lavender shading on backs, undersides and sides
    away = sstep(-0.1, 0.8, -front)                    # facing away from camera
    under = sstep(0.1, 0.8, -nrm[:, 2])
    side = sstep(0.4, 0.95, np.abs(nrm @ D.RIGHT))
    body = ~is_ear & ~L('muzzle')
    tint = np.clip(0.75 * away + 0.65 * under + 0.5 * side + 0.22, 0, 0.92) * body
    col = col + (BLUE - col) * tint[:, None]

    # ---- head
    hm = L('head', 'crown', 'cheek', 'muzzle')
    comb[hm] = tangent(np.tile(-D.FWD * -1 + 0.4 * up, (hm.sum(), 1)), nrm[hm])
    length[hm] = 0.010
    lift[hm] = 0.18
    # crown + locks: long swept tufts toward screen-left / up
    cr = L('crown')
    sweep = unit(np.array([-0.8, 0, 0.6]))
    comb[cr] = tangent(np.tile(sweep, (cr.sum(), 1)) + rng.normal(0, 0.2, (cr.sum(), 3)), nrm[cr])
    length[cr] = 0.042
    lift[cr] = 0.45
    clump[cr] = 0.97
    reach[cr] = 3.6
    # blue on top of the head and between the crest tufts, a lighter blue on the temples
    top = sstep(0.3, 0.9, nrm[:, 2]) * hm
    col = col + (BLUE - col) * (0.65 * top)[:, None]
    temple_R = (1 - sstep(10, 40, np.hypot(np.maximum(uv[:, 0] - 285, 0) * 0.6, np.abs(uv[:, 1] - 205))))
    temple_L = (1 - sstep(10, 40, np.hypot(np.maximum(475 - uv[:, 0], 0) * 0.6, np.abs(uv[:, 1] - 195))))
    col = col + (BLUE - col) * (0.55 * np.maximum(temple_R, temple_L) * hm * (front > 0))[:, None]
    # painted markings in the art's image space (only on surfaces facing the camera)
    vis = (front > 0.05) & hm
    brows = np.maximum(stroke_mask(uv, [(298, 182), (318, 170), (343, 164), (362, 163)], 7.5),
                       stroke_mask(uv, [(432, 163), (446, 157), (462, 157), (470, 161)], 6.5))
    streaks = np.maximum.reduce([stroke_mask(uv, [(352, 186), (366, 168), (378, 150)], 7),
                                 stroke_mask(uv, [(394, 182), (402, 160), (410, 140)], 6),
                                 stroke_mask(uv, [(420, 178), (430, 162)], 5)])
    eye_ring = np.zeros(N)
    for key, e in D.EYES.items():
        x0, y0, x1, y1 = e['box']
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = np.hypot((uv[:, 0] - cx) / ((x1 - x0) / 2), (uv[:, 1] - cy) / ((y1 - y0) / 2))
        eye_ring = np.maximum(eye_ring, (1 - sstep(1.0, 1.45, r)) * sstep(0.8, 1.0, r) * (uv[:, 1] < cy + 8))
    mark = np.maximum.reduce([brows, 0.75 * streaks, 0.55 * eye_ring]) * vis
    col = col + (BLUE_D - col) * mark[:, None]
    # face: short smooth fur; cleared under the eye and mouth decals
    for key, e in D.EYES.items():
        x0, y0, x1, y1 = e['box']
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = np.hypot((uv[:, 0] - cx) / ((x1 - x0) / 2), (uv[:, 1] - cy) / ((y1 - y0) / 2))
        inside = (r < 1.08) & (front > 0) & hm
        dens[inside] = 0
        near = (r < 1.7) & (front > 0) & hm
        length[near] = np.minimum(length[near], 0.004 + 0.010 * np.clip(r[near] - 1.0, 0, 1))
        comb[near] = tangent(co[near] - D.cam_pt(cx, cy, -0.13), nrm[near])
    mx0, my0, mx1, my1 = D.MOUTH_BOX
    mr = np.hypot((uv[:, 0] - (mx0 + mx1) / 2) / ((mx1 - mx0) / 2), (uv[:, 1] - (my0 + my1) / 2) / ((my1 - my0) / 2))
    dens[(mr < 1.05) & (front > 0) & hm] = 0
    nearm = (mr < 1.6) & (front > 0) & hm
    length[nearm] = np.minimum(length[nearm], 0.005)
    nr_ = np.hypot(uv[:, 0] - D.NOSE[0], uv[:, 1] - D.NOSE[1])
    dens[(nr_ < 24) & (front > 0) & hm] = 0
    m = L('muzzle')
    length[m] = np.minimum(length[m], 0.006)
    lift[m] = 0.15
    # cheek ruff: big spiky tufts out to the sides and down
    ck = L('cheek') | (hm & ((uv[:, 0] < 285) | (uv[:, 0] > 500)) & (uv[:, 1] > 225))
    out = np.c_[np.sign(uv[ck, 0] - 390), np.zeros(ck.sum()), np.zeros(ck.sum())]
    comb[ck] = tangent(out * 0.8 - 0.75 * up, nrm[ck])
    length[ck] = 0.034
    lift[ck] = 0.35
    clump[ck] = 0.97
    reach[ck] = 3.4
    # chin / jowl ruff: fluffy, pointing down
    jw = hm & (uv[:, 1] > 285) & ~L('muzzle')
    comb[jw] = tangent(np.tile(-up, (jw.sum(), 1)), nrm[jw])
    length[jw] = np.maximum(length[jw], 0.016)
    clump[jw] = 0.9
    reach[jw] = 2.2

    # ---- chest / body / legs / feet / arms / paws
    m = L('chest', 'body')
    length[m] = 0.014
    clump[m] = 0.85
    reach[m] = 2.0
    m = L('leg')
    length[m] = 0.014
    comb[m] = tangent(np.tile(-up + 0.5 * D.FWD, (m.sum(), 1)), nrm[m])
    m = L('foot')
    length[m] = 0.007
    lift[m] = 0.2
    comb[m] = tangent(np.tile(-D.FWD, (m.sum(), 1)), nrm[m])
    col[m] = col[m] + (CREAM - col[m]) * 0.5
    m = L('arm')
    length[m] = 0.016
    clump[m] = 0.9
    reach[m] = 2.2
    comb[m] = tangent(np.tile(-up, (m.sum(), 1)), nrm[m])
    m = L('paw')
    length[m] = 0.008
    lift[m] = 0.2
    clump[m] = 0.5
    col[m] = CREAM

    # ---- tail: long plume, teal top with a jagged edge
    m = L('tail')
    tp = np.array([D.cam_pt(*p) for p, r in D.TAIL])
    tt = np.zeros(N)
    tan = np.zeros((N, 3))
    best = np.full(N, 1e9)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(tp, axis=0), axis=1))])
    cum /= cum[-1]
    for i in range(len(tp) - 1):
        ab = tp[i + 1] - tp[i]
        t = np.clip(((co - tp[i]) @ ab) / (ab @ ab), 0, 1)
        dd = np.linalg.norm(co - (tp[i] + t[:, None] * ab), axis=1)
        k = dd < best
        best[k] = dd[k]
        tt[k] = cum[i] + t[k] * (cum[i + 1] - cum[i])
        tan[k] = ab / np.linalg.norm(ab)
    comb[m] = tangent(tan[m] + 0.2 * nrm[m], nrm[m])
    length[m] = 0.026 + 0.020 * np.sin(np.pi * np.clip(tt[m], 0, 1)) ** 0.5
    lift[m] = 0.32
    clump[m] = 0.92
    reach[m] = 2.6
    # teal: image-space jagged boundary for the visible side, tail param for the back
    bu = np.array([40, 70, 95, 120, 150, 180, 215])
    bv = np.array([440, 420, 405, 380, 365, 345, 330])
    edge_v = np.interp(uv[:, 0], bu, bv) + 9 * np.sign(np.sin(uv[:, 0] / 9.0))
    teal_vis = sstep(-6, 6, edge_v - uv[:, 1])
    teal_back = sstep(0.55, 0.62, tt)
    tealw = np.where(front > 0, teal_vis, teal_back) * m
    tealc = TEAL + (TEAL_L - TEAL) * sstep(380, 290, uv[:, 1])[:, None]
    col = col + (tealc - col) * tealw[:, None]
    col = col + (BLUE_L - col) * (0.35 * (1 - tealw) * m)[:, None]

    # ---- ears: deep blue inside with a lighter centre and white fluffy rims
    for key in ('R', 'L'):
        em = is_ear & (ear_key == key)
        if not em.any():
            continue
        e = D.EARS[key]
        B, T, A, S, Nf, Ln = D.ear_frame(e)
        u, X, Y = ear_par[em, 0], ear_par[em, 1], ear_par[em, 2]
        rim = np.abs(X) > 0.80
        inner = (Y > -0.2) & ~rim
        c = np.tile(CREAM, (em.sum(), 1))
        c[~inner] = CREAM + (BLUE_L - CREAM) * (0.4 + 0.3 * u[~inner])[:, None]
        cen = sstep(0.8, 0.15, np.abs(X)) * sstep(0.1, 0.35, u) * (1 - sstep(0.7, 0.9, u))
        ic = EAR_IN + (EAR_IN_L - EAR_IN) * (0.55 * cen)[:, None]
        c[inner] = ic[inner]
        if key == 'L':          # cyan tuft low in his left ear (screen-right)
            spot = np.exp(-(((u - 0.32) / 0.12) ** 2 + ((X + 0.15) / 0.3) ** 2))
            c[inner] = c[inner] + (CYAN_TUFT - c[inner]) * (0.9 * spot[inner])[:, None]
        lng = np.full(em.sum(), 0.010)
        lft = np.full(em.sum(), 0.2)
        cmb = tangent(np.tile(A, (em.sum(), 1)), nrm[em])
        dn = np.full(em.sum(), 0.9)
        lng[rim] = 0.022
        lft[rim] = 0.5
        dn[rim] = 1.0
        cmb[rim] = tangent(A * 0.4 + S * np.sign(X[rim])[:, None], nrm[em][rim])
        lng[inner] = 0.006
        dn[inner] = 0.45
        fringe = inner & (np.abs(X) > 0.72)
        lng[fringe] = 0.010
        dn[fringe] = 0.9
        cmb[fringe] = tangent(A * 0.7 - S * np.sign(X[fringe])[:, None] * 0.5 + Nf * 0.3, nrm[em][fringe])
        c[fringe] = c[fringe] + (CREAM - c[fringe]) * sstep(0.72, 0.9, np.abs(X[fringe]))[:, None]
        buried = D.body_sdf(co[em]) < -0.002
        dn[buried] = 0
        col[em] = c
        length[em], lift[em], comb[em], dens[em] = lng, lft, cmb, dn
        clump[em] = 0.75
        reach[em] = 1.6
    return dict(color=col, length=length, lift=lift, comb=comb, clump=clump, reach=reach, density=dens)


# ====================================================================== fur strands
KPTS = 6


def sample_roots(me, vattr, n, rng):
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    tri_v = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get('vertices', tri_v)
    tri_v = tri_v.reshape(-1, 3)
    tri_l = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get('loops', tri_l)
    tri_l = tri_l.reshape(-1, 3)
    co = vattr['co']
    a, b, c = co[tri_v[:, 0]], co[tri_v[:, 1]], co[tri_v[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    w = area * vattr['density'][tri_v].min(1)
    w /= w.sum()
    tri = rng.choice(nt, n, p=w)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], -1)
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    out = {}
    for k, v in vattr.items():
        x = v[tri_v[tri]]
        out[k] = (x * bary[..., None]).sum(1) if x.ndim == 3 else (x * bary).sum(1)
    out['uv'] = (uv[tri_l[tri]] * bary[..., None]).sum(1)
    out['nrm'] = unit(out['nrm'])
    out['comb'] = unit(out['comb'])
    return out


def strands(R, rng):
    n = len(R['co'])
    L = R['length'] * rng.uniform(0.8, 1.15, n)
    nrm, comb = R['nrm'], R['comb']
    lift = np.clip(R['lift'] + rng.normal(0, 0.06, n), 0.05, 0.95)
    jit = unit(rng.normal(0, 1, (n, 3)))
    jit = unit(jit - nrm * (jit * nrm).sum(1, keepdims=True))
    comb = unit(comb + jit * 0.12)
    d0 = unit(nrm * lift[:, None] + comb * (1 - lift)[:, None])
    d1 = unit(nrm * (0.2 * lift)[:, None] + comb + np.array([0, 0, -0.12]))
    t = np.linspace(0, 1, KPTS)
    P = np.zeros((n, KPTS, 3))
    P[:, 0] = R['co'] - nrm * 0.0008
    for k in range(1, KPTS):
        tk = (k - 0.5) / (KPTS - 1)
        P[:, k] = P[:, k - 1] + unit(d0 * (1 - tk) + d1 * tk) * (L / (KPTS - 1))[:, None]
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
    pull = amt[:, None, None] * (t ** 1.3)[None, :, None]
    P = P + pull * (P[cidx] - P) * 0.93
    ph = rng.uniform(0, 6.28, n)
    wav = np.sin(t[None, :] * 3.0 + ph[:, None]) * (0.03 * L)[:, None] * t[None, :]
    return P + jit[:, None, :] * wav[..., None]


def build_fur(body, coll, mat, vattr, n):
    rng = np.random.default_rng(7)
    me = body.data
    R = sample_roots(me, vattr, n, rng)
    P = strands(R, rng)
    cv = bpy.data.hair_curves.new('Pip3_Fur')
    cv.add_curves([KPTS] * len(P))
    cv.position_data.foreach_set('vector', P.reshape(-1).astype(np.float32))
    rad = cv.attributes.new('radius', 'FLOAT', 'POINT')
    t = np.linspace(0, 1, KPTS)
    rr = 0.0010 * (1 - t) ** 0.9 + 0.00012
    sc = np.clip(R['length'] / 0.016, 0.7, 1.4) ** 0.5
    rad.data.foreach_set('value', (rr[None] * sc[:, None]).reshape(-1).astype(np.float32))
    col = cv.attributes.new('fur_color', 'FLOAT_COLOR', 'CURVE')
    c4 = np.c_[R['color'] * rng.uniform(0.95, 1.04, (len(P), 1)), np.ones(len(P))]
    col.data.foreach_set('color', np.clip(c4, 0, 1).reshape(-1).astype(np.float32))
    suv = cv.attributes.new('surface_uv_coordinate', 'FLOAT2', 'CURVE')
    suv.data.foreach_set('vector', R['uv'].reshape(-1).astype(np.float32))
    cv.surface = body
    cv.surface_uv_map = me.uv_layers.active.name
    cv.materials.append(mat)
    ob = link(bpy.data.objects.new('Pip3_Fur', cv), coll)
    ob.parent = body
    ng = bpy.data.node_groups.new('Pip3_FurDeform', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi, go = ng.nodes.new('NodeGroupInput'), ng.nodes.new('NodeGroupOutput')
    dn = ng.nodes.new('GeometryNodeDeformCurvesOnSurface')
    ng.links.new(gi.outputs[0], dn.inputs[0])
    ng.links.new(dn.outputs[0], go.inputs[0])
    ob.modifiers.new('Deform on Body', 'NODES').node_group = ng
    return ob


# ====================================================================== materials
def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    return m, nt, out


def rgba(h):
    c = hexlin(h)
    return (c[0], c[1], c[2], 1.0)


def principled(nt, out, color=None, rough=0.5, **kw):
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    if color is not None:
        b.inputs['Base Color'].default_value = rgba(color) if isinstance(color, str) else color
    b.inputs['Roughness'].default_value = rough
    for k, v in kw.items():
        b.inputs[k].default_value = v
    nt.links.new(b.outputs[0], out.inputs['Surface'])
    return b


def materials():
    M = {}
    m, nt, out = new_mat('Pip3_Skin')
    b = principled(nt, out, rough=0.9, **{'Specular IOR Level': 0.1})
    a = nt.nodes.new('ShaderNodeAttribute')
    a.attribute_name = 'pip_color'
    nt.links.new(a.outputs['Color'], b.inputs['Base Color'])
    M['skin'] = m

    m, nt, out = new_mat('Pip3_Fur')
    h = nt.nodes.new('ShaderNodeBsdfHairPrincipled')
    h.model, h.parametrization = 'CHIANG', 'COLOR'
    h.inputs['Roughness'].default_value = 0.45
    h.inputs['Radial Roughness'].default_value = 0.8
    h.inputs['Coat'].default_value = 0.2
    a = nt.nodes.new('ShaderNodeAttribute')
    a.attribute_name = 'fur_color'
    info = nt.nodes.new('ShaderNodeHairInfo')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.80, 0.82, 0.88, 1)
    ramp.color_ramp.elements[1].position = 0.5
    nt.links.new(info.outputs['Intercept'], ramp.inputs['Fac'])
    mul = nt.nodes.new('ShaderNodeMix')
    mul.data_type, mul.blend_type = 'RGBA', 'MULTIPLY'
    mul.inputs['Factor'].default_value = 1.0
    nt.links.new(a.outputs['Color'], mul.inputs[6])
    nt.links.new(ramp.outputs['Color'], mul.inputs[7])
    gam = nt.nodes.new('ShaderNodeGamma')
    gam.inputs['Gamma'].default_value = 1.0
    nt.links.new(mul.outputs[2], gam.inputs['Color'])
    nt.links.new(gam.outputs['Color'], h.inputs['Color'])
    dif = nt.nodes.new('ShaderNodeBsdfDiffuse')
    nt.links.new(gam.outputs['Color'], dif.inputs['Color'])
    mix = nt.nodes.new('ShaderNodeMixShader')
    mix.inputs['Fac'].default_value = 0.55
    nt.links.new(h.outputs[0], mix.inputs[1])
    nt.links.new(dif.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    M['fur'] = m

    m, nt, out = new_mat('Pip3_Nose')
    principled(nt, out, '#141220', 0.22, **{'Coat Weight': 0.8, 'Coat Roughness': 0.06})
    M['nose'] = m
    m, nt, out = new_mat('Pip3_Scarf')
    b = principled(nt, out, rough=0.55, **{'Sheen Weight': 0.45, 'Sheen Roughness': 0.3,
                                          'Sheen Tint': (0.45, 0.55, 1.0, 1)})
    nz = nt.nodes.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = 40
    rp = nt.nodes.new('ShaderNodeValToRGB')
    rp.color_ramp.elements[0].color = rgba('#0C1240')
    rp.color_ramp.elements[1].color = rgba('#1D2B7C')
    nt.links.new(nz.outputs['Fac'], rp.inputs['Fac'])
    nt.links.new(rp.outputs['Color'], b.inputs['Base Color'])
    M['scarf'] = m
    m, nt, out = new_mat('Pip3_Gold')
    b = principled(nt, out, '#FFC21A', 0.22, **{'Metallic': 0.5, 'Coat Weight': 0.85, 'Coat Roughness': 0.05})
    M['gold'] = m
    m, nt, out = new_mat('Pip3_Pips')
    principled(nt, out, '#3A1C06', 0.5, **{'Metallic': 0.4})
    M['pips'] = m
    return M


def decal_material(name, png):
    m, nt, out = new_mat(name)
    img = bpy.data.images.load(png)
    img.pack()
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.interpolation = 'Cubic'
    tex.extension = 'CLIP'
    b = principled(nt, out, rough=0.3, **{'Coat Weight': 0.6, 'Coat Roughness': 0.05})
    nt.links.new(tex.outputs['Color'], b.inputs['Base Color'])
    nt.links.new(tex.outputs['Alpha'], b.inputs['Alpha'])
    m.blend_method = 'HASHED' if hasattr(m, 'blend_method') else None
    return m


# ====================================================================== painted decals
SS = 4   # supersampling for crisp painted edges


def paint_eye(key, path):
    e = D.EYES[key]
    x0, y0, x1, y1 = e['box']
    W, H = (x1 - x0) * SS * 2, (y1 - y0) * SS * 2      # 2 texels per ref pixel, supersampled
    sx = W / (x1 - x0)
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def P(x, y):
        return ((x - x0) * sx, (y - y0) * sx)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) / 2 - 3, (y1 - y0) / 2 - 3
    fl = e['flick']
    # almond outline: rounder top, flatter bottom, a lifted outer corner
    top, bot = [], []
    for i in range(41):
        k = i / 40
        x = x0 + 3 + k * (2 * hw)
        s = (x - cx) / hw
        lift = 6 * max(0.0, s * fl) ** 2
        top.append(P(x, cy - hh * (1 - s * s) ** 0.55 - lift + 2))
        bot.append(P(x, cy + hh * 0.92 * (1 - s * s) ** 0.75 - lift * 0.3))
    shape = top + bot[::-1]
    d.polygon(shape, fill=(236, 236, 242, 255))
    # iris (gradient) and pupil
    ix, iy, irx, iry = e['iris']
    irx, iry = irx * 1.18, iry * 1.12
    iris = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    di = ImageDraw.Draw(iris)
    steps = 26
    for i in range(steps):
        k = 1 - i / steps
        c0 = np.array([70, 30, 6]) * (k ** 0.6) + np.array([228, 150, 40]) * (1 - k ** 0.6)
        di.ellipse([*P(ix - irx * k, iy - iry * k), *P(ix + irx * k, iy + iry * k)], fill=tuple(int(v) for v in c0) + (255,))
    di.ellipse([*P(ix - irx * 0.62, iy + iry * 0.05 - iry * 0.45), *P(ix + irx * 0.62, iy + iry * 0.9)],
               fill=(246, 178, 60, 110))
    di.ellipse([*P(ix - irx, iy - iry), *P(ix + irx, iy + iry)], outline=(60, 25, 5, 255), width=int(2 * sx))
    px_, py_, prx, pry = e['pupil']
    prx, pry = prx * 1.1, pry * 1.1
    di.ellipse([*P(px_ - prx, py_ - pry), *P(px_ + prx, py_ + pry)], fill=(16, 8, 6, 255))
    mask = Image.new('L', (W, H), 0)
    ImageDraw.Draw(mask).polygon(shape, fill=255)
    im.paste(iris, (0, 0), Image.composite(iris, Image.new('RGBA', (W, H)), mask).split()[3])
    # liner: thick top band with a flick, thin dark lower line
    d = ImageDraw.Draw(im)
    d.line(top, fill=(14, 10, 14, 255), width=int(7.0 * sx), joint='curve')
    end = top[-1] if fl > 0 else top[0]
    tip = (end[0] + fl * 9 * sx, end[1] - 7 * sx)
    d.polygon([(end[0], end[1] + 2.5 * sx), (end[0] - fl * 2 * sx, end[1] - 3 * sx), tip], fill=(14, 10, 14, 255))
    d.line(bot[int(len(bot) * (0.15 if fl < 0 else 0.0)): int(len(bot) * (1.0 if fl < 0 else 0.85))],
           fill=(70, 40, 40, 220), width=int(1.6 * sx), joint='curve')
    # highlights
    for hx, hy, hr in e['hi']:
        d.ellipse([*P(hx - hr, hy - hr), *P(hx + hr, hy + hr)], fill=(255, 255, 255, 255))
    im = im.resize((W // SS, H // SS), Image.LANCZOS)
    im.save(path)
    return (x0, y0, x1, y1)


def paint_mouth(path):
    x0, y0, x1, y1 = D.MOUTH_BOX
    W, H = (x1 - x0) * SS * 2, (y1 - y0) * SS * 2
    sx = W / (x1 - x0)
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def P(x, y):
        return ((x - x0) * sx, (y - y0) * sx)
    # upper lip line (corners up), open D-shaped grin below it
    lip = [P(366, 266), P(378, 274), P(392, 279), P(408, 280), P(424, 277), P(438, 272), P(447, 266)]
    open_ = lip + [P(441, 286), P(433, 306), P(414, 324), P(392, 322), P(376, 304), P(369, 285)]
    d.polygon(open_, fill=(92, 24, 34, 255))
    # tongue
    d.ellipse([*P(381, 292), *P(433, 334)], fill=(234, 122, 142, 255))
    d.line([P(407, 296), P(406, 318)], fill=(200, 90, 110, 255), width=int(1.3 * sx))
    mask = Image.new('L', (W, H), 0)
    ImageDraw.Draw(mask).polygon(open_, fill=255)
    a = np.array(im)
    a[..., 3] = np.minimum(a[..., 3], np.array(mask))
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    d.line(open_ + [open_[0]], fill=(28, 12, 18, 255), width=int(2.2 * sx), joint='curve')
    # little nose-to-lip line
    d.line([P(427, 268), P(424, 277)], fill=(40, 25, 35, 200), width=int(1.2 * sx))
    im = im.resize((W // SS, H // SS), Image.LANCZOS)
    im.save(path)


def decal_patch(name, box, body_obj, mat, coll, nx=40, ny=30, lift=0.0015):
    """A grid in the art's image space over `box`, shrink-wrapped onto the body along
    the camera ray, UVs = the painted image."""
    x0, y0, x1, y1 = box
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(body_obj, dg)
    verts, uvs = [], []
    for j in range(ny + 1):
        for i in range(nx + 1):
            u = x0 + (x1 - x0) * i / nx
            v = y0 + (y1 - y0) * j / ny
            o = Vector(D.cam_pt(u, v, -0.5))
            hit = tree.ray_cast(o, Vector(D.FWD))
            p = hit[0] if hit[0] is not None else o + Vector(D.FWD) * 0.5
            verts.append(tuple(p - Vector(D.FWD) * lift))
            uvs.append((i / nx, 1 - j / ny))
    faces = [(j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i)
             for j in range(ny) for i in range(nx)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uvl = me.uv_layers.new(name='UVMap')
    for poly in me.polygons:
        for li, vi in zip(poly.loop_indices, poly.vertices):
            uvl.data[li].uv = uvs[vi]
        poly.use_smooth = True
    me.materials.append(mat)
    return link(bpy.data.objects.new(name, me), coll)


# ====================================================================== props
def make_nose(body_obj, mat, coll):
    dg = bpy.context.evaluated_depsgraph_get()
    tree = BVHTree.FromObject(body_obj, dg)
    hit = tree.ray_cast(Vector(D.cam_pt(D.NOSE[0], D.NOSE[1] + 2, -0.5)), Vector(D.FWD))
    c = np.array(hit[0]) - D.FWD * 0.004
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=20, radius=1.0)
    for v in bm.verts:
        x, y, z = v.co
        w = 1.0 + 0.30 * z
        v.co = Vector((x * w, y, z))
    me = bpy.data.meshes.new('Pip3_Nose')
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat)
    ob = link(bpy.data.objects.new('Pip3_Nose', me), coll)
    fwd = Vector(D.unit(-D.FWD + 0.55 * D.FACING))
    rot = fwd.to_track_quat('-Y', 'Z')
    ob.matrix_world = Matrix.Translation(Vector(c)) @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(
        (D.px(16), 0.013, D.px(11), 1))
    return ob


def make_scarf(mat, coll):
    parts = []
    rc = D.SCARF_RING
    c = D.cam_pt(*rc['c'])
    nt_, npf = 96, 16
    rx, rz, rt = D.px(rc['rx']), rc['rz'], D.px(rc['tube'])
    verts = []
    tilt = D.px(rc['tilt_px'])
    for i in range(nt_):
        th = 2 * math.pi * i / nt_
        cen = c + np.array([math.sin(th) * rx, -math.cos(th) * rz, 0]) + np.array([0, 0, -tilt * math.cos(th)])
        radial = D.unit(np.array([math.sin(th) * rz, -math.cos(th) * rx, 0]))
        fold = 1 + 0.10 * math.sin(th * 7 + 0.7)
        for j in range(npf):
            ph = 2 * math.pi * j / npf
            verts.append(cen + rt * fold * (math.cos(ph) * radial * 1.1 + math.sin(ph) * np.array([0, 0, 1.0])))
    faces = []
    for i in range(nt_):
        for j in range(npf):
            a = i * npf + j
            faces.append((a, ((i + 1) % nt_) * npf + j, ((i + 1) % nt_) * npf + (j + 1) % npf, i * npf + (j + 1) % npf))
    parts.append((np.array(verts), faces))
    # knot
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=12, radius=1.0)
    kc = D.cam_pt(*D.SCARF_KNOT)
    kv = [kc + np.array([v.co.x * 0.034, v.co.y * 0.03, v.co.z * 0.032]) * (1 + 0.1 * math.sin(4 * v.co.x + 3 * v.co.z))
          for v in bm.verts]
    kf = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    parts.append((np.array(kv), kf))
    # tails: ribbons along image-space paths, wavy, with a notched end
    for path, widths in zip(D.SCARF_TAILS, D.SCARF_TAIL_W):
        P = np.array([D.cam_pt(*p) for p in path])
        tt = np.linspace(0, 1, 20)
        src = np.linspace(0, 1, len(P))
        Pd = np.stack([np.interp(tt, src, P[:, k]) for k in range(3)], -1)
        Wd = D.px(np.interp(tt, src, widths))
        nw, th = 8, 0.007
        vv = []
        for i in range(len(tt)):
            tg = D.unit(Pd[min(i + 1, len(tt) - 1)] - Pd[max(i - 1, 0)])
            side = D.unit(np.cross(tg, -D.FWD))
            nrm = np.cross(side, tg)
            for j in range(nw):
                x = (j / (nw - 1) - 0.5) * Wd[i]
                rip = 0.006 * math.sin(tt[i] * 10 + j * 0.8) * tt[i]
                for lay in (0.5, -0.5):
                    vv.append(Pd[i] + side * x + nrm * (lay * th + rip))
        ff = []
        Wn = nw * 2
        for i in range(len(tt) - 1):
            for j in range(nw - 1):
                a, b = i * Wn + j * 2, i * Wn + (j + 1) * 2
                c2, d2 = (i + 1) * Wn + (j + 1) * 2, (i + 1) * Wn + j * 2
                ff += [(a, b, c2, d2), (a + 1, d2 + 1, c2 + 1, b + 1)]
            a, d2 = i * Wn, (i + 1) * Wn
            ff.append((a, d2, d2 + 1, a + 1))
            a, d2 = i * Wn + (nw - 1) * 2, (i + 1) * Wn + (nw - 1) * 2
            ff.append((a, a + 1, d2 + 1, d2))
        for i in (0, len(tt) - 1):
            for j in range(nw - 1):
                a, b = i * Wn + j * 2, i * Wn + (j + 1) * 2
                ff.append((a, a + 1, b + 1, b) if i == 0 else (a, b, b + 1, a + 1))
        parts.append((np.array(vv), ff))
    bm = bmesh.new()
    for v, f in parts:
        me = mesh_from('tmp', v, f)
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new('Pip3_Scarf')
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mat)
    ob = link(bpy.data.objects.new('Pip3_Scarf', me), coll)
    sub = ob.modifiers.new('Subdivision', 'SUBSURF')
    sub.levels, sub.render_levels = 1, 2
    return ob


PIPS = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)], 4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
        5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)], 6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]}


def make_die(M, coll):
    e = D.DIE_EDGE
    h = e / 2
    R = Euler([math.radians(a) for a in D.DIE_EULER_DEG], 'ZYX').to_matrix()   # == intrinsic XYZ
    normals = [Vector(n) for n in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
    # the face looking most up shows 1, the camera-left face 3, the camera-right face 5
    world = {tuple(n): R @ n for n in normals}
    top = max(normals, key=lambda n: world[tuple(n)].z)
    rest = [n for n in normals if (n + top).length > 1e-6 and n != top]
    view = Vector(-D.FWD)
    vis = sorted(rest, key=lambda n: -world[tuple(n)].dot(view))[:2]
    left = min(vis, key=lambda n: world[tuple(n)].dot(Vector(D.RIGHT)))
    right = vis[0] if vis[1] == left else vis[1]
    num = {tuple(top): 1, tuple(-top): 6, tuple(left): 3, tuple(-left): 4, tuple(right): 5, tuple(-right): 2}
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=e)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=e * 0.17, segments=6, profile=0.5, affect='EDGES')
    me = bpy.data.meshes.new('Pip3_Die')
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M['gold'])
    die = link(bpy.data.objects.new('Gold_Die', me), coll)
    cut = bpy.data.collections.new('cutters')
    bpy.context.scene.collection.children.link(cut)
    pr, sp = e * 0.09, e * 0.25
    for nrm, n in num.items():
        nv = np.array(nrm, float)
        a = np.array([1.0, 0, 0]) if abs(nv[0]) < 0.5 else np.array([0, 1.0, 0])
        b = np.cross(nv, a)
        for i, j in PIPS[n]:
            cbm = bmesh.new()
            bmesh.ops.create_uvsphere(cbm, u_segments=24, v_segments=12, radius=pr)
            cme = bpy.data.meshes.new('c')
            cbm.to_mesh(cme)
            cbm.free()
            cme.materials.append(M['pips'])
            co = bpy.data.objects.new('c', cme)
            co.location = Vector(nv * (h + pr * 0.3) + a * i * sp + b * j * sp)
            cut.objects.link(co)
    mod = die.modifiers.new('pips', 'BOOLEAN')
    mod.operation, mod.operand_type, mod.collection = 'DIFFERENCE', 'COLLECTION', cut
    mod.solver, mod.material_mode = 'EXACT', 'TRANSFER'
    dg = bpy.context.evaluated_depsgraph_get()
    nm = bpy.data.meshes.new_from_object(die.evaluated_get(dg))
    die.modifiers.clear()
    old = die.data
    die.data = nm
    nm.name = 'Pip3_Die'
    bpy.data.meshes.remove(old)
    for o in list(cut.objects):
        m = o.data
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(m)
    bpy.data.collections.remove(cut)
    for p in nm.polygons:
        p.use_smooth = True
    die.matrix_world = Matrix.Translation(Vector(D.cam_pt(*D.DIE_C))) @ R.to_4x4()
    return die


# ====================================================================== stage
def stage(coll):
    scene = bpy.context.scene
    cd = bpy.data.cameras.new('Cam_Reference')
    cd.type = 'ORTHO'
    cd.ortho_scale = D.ORTHO
    cam = link(bpy.data.objects.new('Cam_Reference', cd), coll)
    loc = Vector(D.ORIGIN - D.FWD * 4.0)
    cam.location = loc
    cam.rotation_euler = Vector(D.FWD).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam

    def area(name, uvd, size, power, color):
        Lt = bpy.data.lights.new(name, 'AREA')
        Lt.shape, Lt.size, Lt.energy, Lt.color = 'DISK', size, power, color
        o = link(bpy.data.objects.new(name, Lt), coll)
        o.location = Vector(uvd)
        o.rotation_euler = (Vector(D.ORIGIN) - o.location).to_track_quat('-Z', 'Y').to_euler()
    # soft painterly lighting: big key from upper screen-left, fill from the right, cool rim
    area('Key', D.cam_pt(-450, -550, -1.5), 1.3, 95, (1.0, 0.97, 0.93))
    area('Fill', D.cam_pt(1150, 300, -1.2), 1.8, 22, (0.82, 0.88, 1.0))
    area('Front', D.cam_pt(313, 313, -2.2), 2.0, 8, (1.0, 1.0, 1.0))
    area('Rim', D.cam_pt(600, -200, 1.2), 1.0, 45, (0.85, 0.92, 1.0))
    area('Rim2', D.cam_pt(0, 100, 1.0), 1.0, 30, (0.85, 0.92, 1.0))
    w = bpy.data.worlds.new('Pip3_World')
    scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.85, 0.88, 0.95, 1)
    bg.inputs['Strength'].default_value = 0.12
    s = scene
    s.render.engine = 'CYCLES'
    s.cycles.device = 'CPU'
    s.cycles.samples = 64
    s.cycles.use_denoising = True
    s.cycles.max_bounces = 6
    s.cycles.transparent_max_bounces = 16
    s.render.resolution_x = s.render.resolution_y = D.IMG
    s.render.film_transparent = True
    s.view_settings.view_transform = 'Khronos PBR Neutral'
    s.view_settings.look = 'None'
    s.view_settings.exposure = -0.35
    return cam


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strands', type=int, default=220000)
    ap.add_argument('--out', default=os.path.join(HERE, 'pip_v3.blend'))
    a = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    C = bpy.data.collections.new('Pip_v3')
    scene.collection.children.link(C)
    S = bpy.data.collections.new('Stage_v3')
    scene.collection.children.link(S)
    M = materials()
    me, n_body, ear_info = body_mesh(C)
    body = link(bpy.data.objects.new('Pip3_Body', me), C)
    me.materials.append(M['skin'])
    log('body', len(me.vertices))
    n = len(me.vertices)
    co = np.array([v.co[:] for v in me.vertices])
    nrm = np.array([v.normal[:] for v in me.vertices])
    prims = D.primitives()
    dist = np.stack([fn(co) for (_, _, _, _, fn) in prims], -1)
    labels = np.array([p[2] for p in prims])[np.argmin(dist, 1)]
    is_ear = np.zeros(n, bool)
    is_ear[n_body:] = True
    ear_par = np.zeros((n, 3))
    ear_key = np.full(n, '', dtype=object)
    off = n_body
    for key, cnt, par in ear_info:
        ear_par[off:off + cnt] = par
        ear_key[off:off + cnt] = key
        off += cnt
    ear_par[:, 0] = np.clip(ear_par[:, 0], 0, 1)
    F = fields(co, nrm, labels.astype(object), is_ear, ear_par, ear_key, np.random.default_rng(3))
    ca = me.color_attributes.new('pip_color', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', np.c_[F['color'], np.ones(n)].reshape(-1).astype(np.float32))
    body.add_rest_position_attribute = True
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.003)
    bpy.ops.object.mode_set(mode='OBJECT')
    # face decals + nose
    tex_dir = os.path.join(HERE, 'textures')
    os.makedirs(tex_dir, exist_ok=True)
    for key in ('R', 'L'):
        p = os.path.join(tex_dir, f'eye_{key}.png')
        box = paint_eye(key, p)
        decal_patch(f'Pip3_Eye.{key}', box, body, decal_material(f'Pip3_Eye.{key}', p), C)
    p = os.path.join(tex_dir, 'mouth_grin.png')
    paint_mouth(p)
    decal_patch('Pip3_Mouth', D.MOUTH_BOX, body, decal_material('Pip3_Mouth', p), C)
    make_nose(body, M['nose'], C)
    scarf = make_scarf(M['scarf'], C)
    # no fur under the scarf
    sv = np.array([v.co[:] for v in scarf.data.vertices])
    dsc, _ = cKDTree(sv).query(co)
    F['density'][dsc < 0.012] = 0
    make_die(M, C)
    vattr = dict(co=co, nrm=nrm, color=F['color'], length=F['length'], lift=F['lift'], comb=F['comb'],
                 clump=F['clump'], reach=F['reach'], density=F['density'])
    build_fur(body, C, M['fur'], vattr, a.strands)
    log('fur', a.strands)
    stage(S)
    bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)
    log('saved', a.out)


if __name__ == '__main__':
    main()
