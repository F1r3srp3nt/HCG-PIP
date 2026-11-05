"""Groomed fur as a Curves (hair) object that rides the rigged body."""
import numpy as np
import bpy
from scipy.spatial import cKDTree

from pip_colors import unit

K = 6          # points per strand


def sample_roots(me, vattr, n, rng):
    """Area x density weighted root sampling on the rest mesh."""
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
    dens = vattr['density'][tri_v].min(1)
    w = area * dens
    w /= w.sum()
    tri = rng.choice(nt, n, p=w)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], -1)
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)

    def interp(arr):
        x = arr[tri_v[tri]]
        return (x * bary[..., None]).sum(1) if x.ndim == 3 else (x * bary).sum(1)
    out = {k: interp(v) for k, v in vattr.items()}
    out['uv'] = (uv[tri_l[tri]] * bary[..., None]).sum(1)
    out['nrm'] = unit(out['nrm'])
    out['comb'] = unit(out['comb'])
    return out


def strands(R, rng):
    n = len(R['co'])
    L = R['length'] * rng.uniform(0.75, 1.2, n)
    nrm, comb = R['nrm'], R['comb']
    lift = np.clip(R['lift'] + rng.normal(0, 0.08, n), 0.05, 0.95)
    jitter = unit(rng.normal(0, 1, (n, 3)))
    jitter = unit(jitter - nrm * (jitter * nrm).sum(1, keepdims=True))
    comb = unit(comb + jitter * 0.10)
    d0 = unit(nrm * lift[:, None] + comb * (1 - lift)[:, None])
    d1 = unit(nrm * (0.25 * lift)[:, None] + comb + np.array([0, 0, -0.18]))
    t = np.linspace(0, 1, K)
    P = np.zeros((n, K, 3))
    P[:, 0] = R['co'] - nrm * 0.0008          # root sits just under the skin
    for k in range(1, K):
        tk = (k - 0.5) / (K - 1)
        d = unit(d0 * (1 - tk) + d1 * tk)
        P[:, k] = P[:, k - 1] + d * (L / (K - 1))[:, None]
    # clumping: tips gather toward the nearest clump-centre strand
    # clump centres on a hash grid whose cell size follows the local tuft size,
    # so cheeks / crest / tail get a few chunky tufts and the body many small ones
    rch = R.get('reach', np.ones(n))
    lvl = np.clip(np.round(rch * 2) / 2, 0.5, 4.0)
    cidx = np.arange(n)
    dist = np.zeros(n)
    for lv in np.unique(lvl):
        ids = np.nonzero(lvl == lv)[0]
        cs = 0.0045 * lv
        key = np.floor((R['co'][ids] + rng.uniform(0, cs, 3)) / cs).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        cent = ids[first]
        tree = cKDTree(R['co'][cent])
        dd, ci = tree.query(R['co'][ids])
        cidx[ids] = cent[ci]
        dist[ids] = dd
    reach = (0.004 + 0.12 * L) * rch + 0.0045 * lvl
    amt = R['clump'] * np.clip(1 - dist / reach, 0, 1)
    tipw = t ** 1.4
    pull = amt[:, None, None] * tipw[None, :, None]
    P = P + pull * (P[cidx] - P) * 0.92
    # a gentle wave so strands are not dead straight
    ph = rng.uniform(0, 6.28, n)
    wav = np.sin(t[None, :] * 3.0 + ph[:, None]) * (0.035 * L)[:, None] * t[None, :]
    P = P + jitter[:, None, :] * wav[..., None]
    return P


def build_fur(body, coll, mat, vattr, n_strands, seed=7):
    rng = np.random.default_rng(seed)
    me = body.data
    R = sample_roots(me, vattr, n_strands, rng)
    P = strands(R, rng)
    n = len(P)
    cv = bpy.data.hair_curves.new('Pip_Fur')
    cv.add_curves([K] * n)
    cv.position_data.foreach_set('vector', P.reshape(-1).astype(np.float32))
    rad = cv.attributes.new('radius', 'FLOAT', 'POINT')
    t = np.linspace(0, 1, K)
    rr = (0.00105 * (1 - t) ** 0.9 + 0.00012)
    scale = np.clip(R['length'] / 0.022, 0.7, 1.35) ** 0.5
    rad.data.foreach_set('value', (rr[None, :] * scale[:, None]).reshape(-1).astype(np.float32))
    col = cv.attributes.new('fur_color', 'FLOAT_COLOR', 'CURVE')
    c4 = np.c_[R['color'] * rng.uniform(0.94, 1.04, (n, 1)), np.ones(n)]
    col.data.foreach_set('color', np.clip(c4, 0, 1).reshape(-1).astype(np.float32))
    suv = cv.attributes.new('surface_uv_coordinate', 'FLOAT2', 'CURVE')
    suv.data.foreach_set('vector', R['uv'].reshape(-1).astype(np.float32))
    cv.surface = body
    cv.surface_uv_map = me.uv_layers.active.name
    cv.materials.append(mat)
    ob = bpy.data.objects.new('Pip_Fur', cv)
    coll.objects.link(ob)
    ob.parent = body
    # deform with the skinned body
    ng = bpy.data.node_groups.new('Pip_FurDeform', 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    gi = ng.nodes.new('NodeGroupInput')
    go = ng.nodes.new('NodeGroupOutput')
    dn = ng.nodes.new('GeometryNodeDeformCurvesOnSurface')
    ng.links.new(gi.outputs[0], dn.inputs[0])
    ng.links.new(dn.outputs[0], go.inputs[0])
    mod = ob.modifiers.new('Deform on Body', 'NODES')
    mod.node_group = ng
    return ob
