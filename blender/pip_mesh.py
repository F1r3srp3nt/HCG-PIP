"""Builds Pip's unified body mesh (SDF body + cupped ears, voxel-remeshed)."""
import numpy as np
import bpy
import bmesh
from mathutils.bvhtree import BVHTree
from skimage import measure

import pip_design as D

VOXEL = 0.0032


def mesh_from_arrays(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.update()
    return me


def sdf_body_mesh(res=0.003):
    lo = np.array([-0.34, -0.32, -0.02])
    hi = np.array([0.24, 0.30, 0.60])
    n = np.ceil((hi - lo) / res).astype(int) + 1
    xs, ys, zs = (np.linspace(lo[i], lo[i] + (n[i] - 1) * res, n[i]) for i in range(3))
    G = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), -1).reshape(-1, 3)
    d = np.empty(len(G))
    step = 1_500_000
    for i in range(0, len(G), step):
        d[i:i + step] = D.body_sdf(G[i:i + step])
    vol = d.reshape(n)
    v, f, _, _ = measure.marching_cubes(vol, 0.0, spacing=(res, res, res))
    v = v + lo
    f = f[:, ::-1]          # outward winding for Blender
    return v, f


def build_body_mesh():
    """Body = voxel-remeshed SDF torso/head/limbs/tail + two clean ear shells.

    The ears stay as their own (clean, parametric) shells inside the same mesh
    object: their bases are buried in the head so the join is invisible, and
    the thin rims stay smooth instead of being aliased by the voxel remesh.
    """
    v, f = sdf_body_mesh()
    tmp = bpy.data.objects.new('tmp_body', mesh_from_arrays('tmp_body', v, f))
    bpy.context.scene.collection.objects.link(tmp)
    mod = tmp.modifiers.new('remesh', 'REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = VOXEL
    mod.adaptivity = 0.0
    sm = tmp.modifiers.new('smooth', 'LAPLACIANSMOOTH')
    sm.iterations = 2
    sm.lambda_factor = 0.4
    dg = bpy.context.evaluated_depsgraph_get()
    body = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    old = tmp.data
    bpy.data.objects.remove(tmp)
    bpy.data.meshes.remove(old)
    bm = bmesh.new()
    bm.from_mesh(body)
    for s in (1, -1):
        ev, ef, _ = D.ear_mesh(s, nu=96, nv=64)
        em = mesh_from_arrays('tmp_ear', ev, ef)
        bm.from_mesh(em)
        bpy.data.meshes.remove(em)
    final = bpy.data.meshes.new('Pip_Body')
    bm.to_mesh(final)
    bm.free()
    bpy.data.meshes.remove(body)
    for p in final.polygons:
        p.use_smooth = True
    return final


def ear_param_lookup():
    """KD lookup tables: ear surface points -> (u, X, Y) params."""
    from scipy.spatial import cKDTree
    out = {}
    for s in (1, -1):
        us = np.linspace(0, 1, 160)
        th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        U, T = np.meshgrid(us, th, indexing='ij')
        X, Y = np.cos(T), np.sin(T)
        P = D.ear_point(s, U.ravel(), X.ravel(), Y.ravel())
        out[s] = (cKDTree(P), np.stack([U.ravel(), X.ravel(), Y.ravel()], -1), P)
    return out


def vertex_analysis(co):
    """Per-vertex part distances, labels, ear params and tail param."""
    prims = D.primitives()
    dist = np.stack([fn(co) for (_, _, _, _, fn) in prims], -1)
    names = [p[0] for p in prims]
    lookup = ear_param_lookup()
    ear_d, ear_par = [], []
    for s in (1, -1):
        tree, par, P = lookup[s]
        dd, idx = tree.query(co)
        ear_d.append(dd - 0.002)
        ear_par.append(par[idx])
    ear_d = np.stack(ear_d, -1)
    # tail parameter: projection along the chain
    seg_a, seg_b = D.TAIL_PTS[:-1], D.TAIL_PTS[1:]
    best = np.full(len(co), 1e9)
    tail_t = np.zeros(len(co))
    tail_tan = np.zeros((len(co), 3))
    L = np.linalg.norm(seg_b - seg_a, axis=1)
    cum = np.concatenate([[0], np.cumsum(L)]) / L.sum()
    for i in range(len(seg_a)):
        ab = seg_b[i] - seg_a[i]
        t = np.clip(((co - seg_a[i]) @ ab) / (ab @ ab), 0, 1)
        q = seg_a[i] + t[:, None] * ab
        dd = np.linalg.norm(co - q, axis=1)
        m = dd < best
        best[m] = dd[m]
        tail_t[m] = cum[i] + t[m] * (cum[i + 1] - cum[i])
        tail_tan[m] = ab / np.linalg.norm(ab)
    return dict(dist=dist, names=names, ear_d=ear_d, ear_par=ear_par,
                tail_t=tail_t, tail_tan=tail_tan, prims=prims)
