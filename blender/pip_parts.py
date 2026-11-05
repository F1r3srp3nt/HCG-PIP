"""Face parts, scarf and the gold die."""
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix

import pip_design as D
from pip_mesh import mesh_from_arrays


def link(obj, coll):
    coll.objects.link(obj)
    return obj


def eye_frame(s):
    f = D.EYE_GAZE[s]
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return f, u, r


def uv_sphere(radius, segs=48, rings=32):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=rings, radius=radius)
    return bm


def tube(points, radii, segs=10, cap=True):
    """Simple swept tube through points with per-point radius."""
    P = np.asarray(points, float)
    verts, faces = [], []
    n = len(P)
    prev = None
    for i in range(n):
        t = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        if prev is None:
            a = np.cross(t, [0, 0, 1.0])
            if np.linalg.norm(a) < 1e-3:
                a = np.cross(t, [1.0, 0, 0])
        else:
            a = prev - t * (prev @ t)
        a /= np.linalg.norm(a)
        b = np.cross(t, a)
        prev = a
        for j in range(segs):
            ang = 2 * math.pi * j / segs
            verts.append(P[i] + radii[i] * (math.cos(ang) * a + math.sin(ang) * b))
    for i in range(n - 1):
        for j in range(segs):
            k = (j + 1) % segs
            faces.append((i * segs + j, i * segs + k, (i + 1) * segs + k, (i + 1) * segs + j))
    if cap:
        verts.append(P[0])
        verts.append(P[-1])
        c0, c1 = len(verts) - 2, len(verts) - 1
        for j in range(segs):
            k = (j + 1) % segs
            faces.append((c0, k, j))
            faces.append((c1, (n - 1) * segs + j, (n - 1) * segs + k))
    return np.array(verts), faces


# ------------------------------------------------------------------ eyes
def eye_matrix(s):
    """World matrix of an eye: local +Z = gaze, local +Y = up."""
    f, u, r = eye_frame(s)
    R = Matrix((tuple(-r), tuple(u), tuple(f))).transposed()
    return Matrix.Translation(Vector(D.EYE_C[s])) @ R.to_4x4()


def make_eye(s, mats, coll):
    bm = uv_sphere(D.EYE_R, 64, 40)
    me = bpy.data.meshes.new(f'Pip_Eye.{"L" if s > 0 else "R"}')
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(me.name, me)
    ob.matrix_world = eye_matrix(s)
    me.materials.append(mats['eye'])
    return link(ob, coll)


UP_LID_CENTER, UP_LID_SIDE = 44.0, 4.0      # degrees: arched upper lid edge
LOW_LID_CENTER, LOW_LID_SIDE = -50.0, -24.0


def lid_edge(az, upper=True):
    """Lid edge elevation (radians) as a function of azimuth: an arch."""
    c, sd = (UP_LID_CENTER, UP_LID_SIDE) if upper else (LOW_LID_CENTER, LOW_LID_SIDE)
    k = min(1.0, (az / math.radians(95)) ** 2)
    return math.radians(c + (sd - c) * k)


def liner_width(az, s, upper=True):
    """Angular width (radians) of the dark liner band hugging the eye under the lid edge."""
    outer = np.clip(az / math.radians(90), 0, 1)       # +az = outer corner (see lid_mesh)
    if upper:
        return math.radians(6.0 + 11.0 * outer ** 1.2)
    return math.radians(1.0 + 3.5 * outer)


def lid_mesh(s, upper=True):
    """A spherical lid shell with an arched cut edge. Under the edge a thin dark
    liner band wraps tight to the eyeball, so the eye outline (like the art's
    bold lash line) rides with the lid through blinks and squints.
    Rotating the shell about the eye's side axis (lid bone local X) opens/closes it."""
    f, u, r = eye_frame(s)
    r = -r * s                       # +r now points to the outer corner for both eyes
    R0 = D.EYE_R * 1.06
    RL = D.EYE_R * 1.006
    th = 0.0030
    nel, naz = 28, 56
    verts, faces = [], []
    sign = 1 if upper else -1
    azs = np.linspace(-math.radians(100), math.radians(100), naz)
    edge_w = []
    for layer in (0, 1):
        for i in range(nel):
            k = i / (nel - 1)
            for j, az in enumerate(azs):
                e0 = lid_edge(az, upper)
                lw = liner_width(az, s, upper) * (0.0 if layer == 0 else 1.0)
                start = e0 - sign * liner_width(az, s, upper)
                el = start + (sign * math.radians(118) - start) * k
                inside = (el - e0) * sign < 0          # in the liner band
                if inside:
                    rad = RL + (0.0012 if layer == 0 else 0.0)
                else:
                    blend = min(1.0, abs(el - e0) / math.radians(3))
                    rad = RL + (R0 - RL) * blend + (th if layer == 0 else 0.0) * blend + (0.0012 if layer == 0 else 0) * (1 - blend)
                d = math.cos(el) * (math.cos(az) * f + math.sin(az) * r) + math.sin(el) * u
                verts.append(D.EYE_C[s] + rad * d)
                if inside:
                    edge_w.append(1.0)
                else:
                    edge_w.append(1.0 - min(1.0, abs(el - e0) / math.radians(6)))
    nn = nel * naz
    for i in range(nel - 1):
        for j in range(naz - 1):
            a, b, c, d_ = i * naz + j, i * naz + j + 1, (i + 1) * naz + j + 1, (i + 1) * naz + j
            faces.append((a, b, c, d_))
            faces.append((nn + a, nn + d_, nn + c, nn + b))
    for j in range(naz - 1):
        a, b = j, j + 1
        faces.append((a, nn + a, nn + b, b))
    return np.array(verts), faces, np.array(edge_w)


def make_lids(s, mats, coll):
    out = {}
    sx = 'L' if s > 0 else 'R'
    for upper in (True, False):
        v, fc, ew = lid_mesh(s, upper)
        name = f'Pip_Lid{"Up" if upper else "Low"}.{sx}'
        me = mesh_from_arrays(name, v, fc)
        for p in me.polygons:
            p.use_smooth = True
        at = me.attributes.new('lash', 'FLOAT', 'POINT')
        at.data.foreach_set('value', ew.astype(np.float32) * (1.0 if upper else 0.85))
        me.materials.append(mats['lid'])
        ob = link(bpy.data.objects.new(name, me), coll)
        out['up' if upper else 'low'] = ob
    # bold upper lash line + the little flick at the outer corner
    f, u, r = eye_frame(s)
    pts, rad = [], []
    for az in np.linspace(-math.radians(68), math.radians(86), 40):
        el = lid_edge(az, True) + math.radians(3.5)
        d = math.cos(el) * (math.cos(az) * f - math.sin(az) * r * s) + math.sin(el) * u
        pts.append(D.EYE_C[s] + D.EYE_R * 1.12 * d)
        k = (az + math.radians(68)) / math.radians(154)
        rad.append(0.0026 + 0.0042 * min(1.0, k * 1.4) ** 0.8)
    # flick: continue outward and up from the outer corner
    last = pts[-1]
    outward = -r * s
    for i, k in enumerate((0.3, 0.6, 0.9)):
        pts.append(last + outward * 0.017 * k + u * 0.011 * k ** 1.4 + f * -0.004 * k)
        rad.append(0.0050 * (1 - k) + 0.0006)
    v, fc = tube(pts, rad, 8)
    # thin lower lash on the outer half of the eye
    lp, lr = [], []
    for az in np.linspace(math.radians(10), math.radians(80), 14):
        el = lid_edge(az, False) - math.radians(1.0)
        d = math.cos(el) * (math.cos(az) * f - math.sin(az) * r * s) + math.sin(el) * u
        lp.append(D.EYE_C[s] + D.EYE_R * 1.10 * d)
        k = (az - math.radians(10)) / math.radians(70)
        lr.append(0.0007 + 0.0011 * math.sin(math.pi * min(1.0, k)) ** 0.5)
    v2_, f2_ = tube(lp, lr, 6)
    fc = list(fc) + [tuple(i + len(v) for i in face) for face in f2_]
    v = np.concatenate([v, v2_])
    me = mesh_from_arrays(f'Pip_Lash.{sx}', v, fc)
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mats['lash'])
    out['lash'] = link(bpy.data.objects.new(me.name, me), coll)
    return out


def make_nose(mats, coll):
    bm = uv_sphere(1.0, 32, 20)
    for v in bm.verts:
        x, y, z = v.co
        # rounded inverted triangle, a touch wider on top
        w = 0.0170 * D.HS * (1.0 + 0.34 * z)
        v.co = Vector((x * w, y * 0.0122 * D.HS, z * 0.0112 * D.HS))
    me = bpy.data.meshes.new('Pip_Nose')
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mats['nose'])
    ob = bpy.data.objects.new('Pip_Nose', me)
    ob.location = Vector(D.NOSE_C)
    ob.rotation_euler = (math.radians(-12), 0, 0)
    return link(ob, coll)


def make_tongue(mats, coll):
    bm = uv_sphere(1.0, 32, 16)
    for v in bm.verts:
        x, y, z = v.co
        v.co = Vector((x * 0.019, y * 0.016, z * 0.0050 + 0.003 * (x * x)))
    me = bpy.data.meshes.new('Pip_Tongue')
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mats['tongue'])
    ob = bpy.data.objects.new('Pip_Tongue', me)
    # rests just under the closed mouth; shows when the jaw drops into a grin
    ob.location = Vector((0.0, D.MOUTH_C[1] + 0.015, D.mouth_z(0.0) - D.MOUTH_GAP - 0.0052))
    ob.rotation_euler = (math.radians(-6), 0, 0)
    return link(ob, coll)


def _surface_y(x, z, y0=-0.32, y1=-0.12):
    """Front surface of the body SDF along +Y at (x, z) (bisection)."""
    lo, hi = np.full_like(x, y0), np.full_like(x, y1)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        inside = D.body_sdf(np.stack([x, mid, z], -1)) < 0
        hi = np.where(inside, mid, hi)
        lo = np.where(inside, lo, mid)
    return 0.5 * (lo + hi)


def make_mouth(mats, coll, body=None, nx=41, nz=14):
    """Open-grin mouth fitted to the muzzle. Basis = relaxed closed mouth (a thin
    dark lip line on the smile curve); shape key 'Grin' = open D-shape with the
    tongue painted in. Driven by the rig's 'grin' property."""
    us = np.linspace(-1, 1, nx)
    vs = np.linspace(0, 1, nz)
    U, Vv = np.meshgrid(us, vs, indexing='ij')
    X = U * D.MOUTH_HALF_W
    top = D.mouth_z(X)
    taper = np.clip(1 - U ** 2, 0, 1) ** 0.5
    Zc = top - 0.0019 * np.minimum(Vv / 0.25, 1.0) * (0.35 + 0.65 * taper)   # rows past 0.25 fold away
    Zo = top + (D.mouth_open_bottom(X) - top) * Vv

    if body is not None:          # fit to the real (remeshed) skin with ray casts
        from mathutils.bvhtree import BVHTree
        dg = bpy.context.evaluated_depsgraph_get()
        tree = BVHTree.FromObject(body, dg)

        def surf(x, z):
            out = np.empty(x.size)
            for k, (xx, zz) in enumerate(zip(x.ravel(), z.ravel())):
                hit = tree.ray_cast(Vector((xx, -0.45, zz)), Vector((0, 1, 0)))
                out[k] = hit[0].y if hit[0] is not None else -0.2
            return out.reshape(x.shape)
    else:
        def surf(x, z):
            return _surface_y(x.ravel(), z.ravel()).reshape(x.shape)
    off = 0.0016
    Yo = surf(X, Zo) - off
    Yc = surf(X, Zc) - off
    verts_c = np.stack([X, Yc, Zc], -1).reshape(-1, 3)
    verts_o = np.stack([X, Yo, Zo], -1).reshape(-1, 3)
    faces = []
    for i in range(nx - 1):
        for j in range(nz - 1):
            a = i * nz + j
            faces.append((a, a + nz, a + nz + 1, a + 1))
    me = bpy.data.meshes.new('Pip_Mouth')
    me.from_pydata([tuple(v) for v in verts_c], [], faces)
    for p in me.polygons:
        p.use_smooth = True
    # paint: dark mouth, pink tongue low in the middle, darker rim
    col = np.zeros((len(verts_c), 4))
    tongue = np.exp(-((U / 0.55) ** 2 + ((Vv - 0.85) / 0.30) ** 2)).ravel()
    rim = np.clip(1 - np.minimum(Vv, 1 - np.abs(U)) / 0.12, 0, 1).ravel()
    base = np.array([0.075, 0.010, 0.016])
    pink = np.array([0.80, 0.22, 0.28])
    c = base[None] + (pink - base)[None] * np.clip(tongue * 1.4, 0, 1)[:, None]
    c = c * (1 - 0.6 * rim)[:, None]
    col[:, :3] = c
    col[:, 3] = 1
    ca = me.color_attributes.new('mcol', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', col.reshape(-1).astype(np.float32))
    ob = bpy.data.objects.new('Pip_Mouth', me)
    link(ob, coll)
    ob.shape_key_add(name='Closed')
    sk = ob.shape_key_add(name='Grin')
    sk.data.foreach_set('co', verts_o.reshape(-1).astype(np.float32))
    sk.slider_max = 1.0
    me.materials.append(mats['mouth'])
    return ob


# ------------------------------------------------------------------ scarf
SCARF_C = np.array([0.0, -0.036, 0.226])
SCARF_TILT = math.radians(8)        # front lower than back


def scarf_ring_point(theta, phi, rx=0.088, ry=0.080, rt=0.029):
    """theta around neck (0 = front), phi around the tube."""
    c, s = math.cos(theta), math.sin(theta)
    center = np.array([s * rx, -c * ry, 0.0])
    radial = np.array([s, -c, 0.0])
    radial /= np.linalg.norm(radial)
    # puffy, slightly squashed tube with soft folds
    fold = 1 + 0.10 * math.sin(theta * 7 + 1.3) * (0.5 + 0.5 * math.cos(phi))
    p = center + rt * fold * (math.cos(phi) * radial * 1.05 + math.sin(phi) * np.array([0, 0, 1.25]))
    # tilt
    ct, st = math.cos(SCARF_TILT), math.sin(SCARF_TILT)
    y, z = p[1], p[2]
    p = np.array([p[0], y * ct - z * st, y * st + z * ct])
    return SCARF_C + p


def ribbon(path, widths, normal_hint, twist=0.0, th=0.0075, nw=7):
    P = np.asarray(path, float)
    n = len(P)
    verts = []
    for i in range(n):
        t = P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        nh = np.asarray(normal_hint, float)
        side = np.cross(t, nh)
        side /= np.linalg.norm(side)
        nrm = np.cross(side, t)
        k = i / (n - 1)
        ang = twist * k
        side, nrm = side * math.cos(ang) + nrm * math.sin(ang), -side * math.sin(ang) + nrm * math.cos(ang)
        for j in range(nw):
            x = (j / (nw - 1) - 0.5) * widths[i]
            ripple = 0.004 * math.sin(k * 9 + j * 0.9) * k
            for layer in (0.5, -0.5):
                verts.append(P[i] + side * x + nrm * (layer * th + ripple))
    faces = []
    W = nw * 2
    for i in range(n - 1):
        for j in range(nw - 1):
            a = i * W + j * 2
            b = i * W + (j + 1) * 2
            c = (i + 1) * W + (j + 1) * 2
            d = (i + 1) * W + j * 2
            faces.append((a, b, c, d))
            faces.append((a + 1, d + 1, c + 1, b + 1))
        # long edges
        a, d = i * W, (i + 1) * W
        faces.append((a, d, d + 1, a + 1))
        a, d = i * W + (nw - 1) * 2, (i + 1) * W + (nw - 1) * 2
        faces.append((a, a + 1, d + 1, d))
    for i in (0, n - 1):
        for j in range(nw - 1):
            a = i * W + j * 2
            b = i * W + (j + 1) * 2
            faces.append((a, a + 1, b + 1, b) if i == 0 else (a, b, b + 1, a + 1))
    return np.array(verts), faces


_SZ = np.array([0, -0.004, -0.056])
SCARF_TAIL_A = np.array([[-0.052, -0.110, 0.262], [-0.090, -0.118, 0.238], [-0.118, -0.110, 0.208],
                         [-0.140, -0.096, 0.178], [-0.158, -0.080, 0.150]]) + _SZ
SCARF_TAIL_B = np.array([[-0.058, -0.104, 0.274], [-0.100, -0.094, 0.280], [-0.140, -0.068, 0.272],
                         [-0.174, -0.036, 0.258], [-0.202, -0.004, 0.242]]) + _SZ
SCARF_KNOT = np.array([-0.052, -0.104, 0.268]) + _SZ


def make_scarf(mats, coll):
    parts = []
    nt, npf = 96, 18
    verts = []
    for i in range(nt):
        th = 2 * math.pi * i / nt
        for j in range(npf):
            verts.append(scarf_ring_point(th, 2 * math.pi * j / npf))
    faces = []
    for i in range(nt):
        for j in range(npf):
            a = i * npf + j
            b = i * npf + (j + 1) % npf
            c = ((i + 1) % nt) * npf + (j + 1) % npf
            d = ((i + 1) % nt) * npf + j
            faces.append((a, d, c, b))
    parts.append((np.array(verts), faces, 'ring'))
    # knot: a squashed lumpy ball
    bm = uv_sphere(1.0, 24, 14)
    kv = []
    for v in bm.verts:
        x, y, z = v.co
        lump = 1 + 0.12 * math.sin(4 * x + 3 * z) * math.cos(3 * y)
        kv.append(SCARF_KNOT + np.array([x * 0.032, y * 0.025, z * 0.030]) * lump)
    kf = [tuple(v.index for v in f.verts) for f in bm.faces]
    bm.free()
    parts.append((np.array(kv), kf, 'knot'))
    for path, widths, hint, tw, name in (
            (SCARF_TAIL_A, [0.056, 0.066, 0.074, 0.078, 0.074], (0.3, -1, 0.3), 0.5, 'tail_a'),
            (SCARF_TAIL_B, [0.050, 0.060, 0.068, 0.072, 0.068], (0.2, -0.5, 1), -0.6, 'tail_b')):
        # densify the path
        tt = np.linspace(0, 1, 18)
        src = np.linspace(0, 1, len(path))
        P = np.stack([np.interp(tt, src, path[:, k]) for k in range(3)], -1)
        Wd = np.interp(tt, src, widths)
        v, f = ribbon(P, Wd, hint, tw)
        parts.append((v, f, name))
    bm = bmesh.new()
    part_ids = []
    for pid, (v, f, name) in enumerate(parts):
        me = mesh_from_arrays('tmp', v, f)
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
        part_ids += [pid] * len(v)
    me = bpy.data.meshes.new('Pip_Scarf')
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mats['scarf'])
    ob = link(bpy.data.objects.new('Pip_Scarf', me), coll)
    ob['part_names'] = [p[2] for p in parts]
    sub = ob.modifiers.new('Subdivision', 'SUBSURF')
    sub.levels = 1
    sub.render_levels = 2
    return ob, np.array(part_ids)


# ------------------------------------------------------------------ die
PIP_LAYOUT = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)],
              4: [(-1, -1), (1, -1), (-1, 1), (1, 1)], 5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)],
              6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]}
# face normal -> number (opposite faces sum to 7)
DIE_FACES = {(0, 0, 1): 1, (0, 0, -1): 6, (0, -1, 0): 3, (0, 1, 0): 4, (1, 0, 0): 5, (-1, 0, 0): 2}


def make_die(mats, coll):
    e = D.DIE_EDGE
    h = e / 2
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=e)
    bmesh.ops.bevel(bm, geom=list(bm.edges), offset=e * 0.16, segments=6, profile=0.5, affect='EDGES')
    bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=2, use_grid_fill=True)
    me = bpy.data.meshes.new('Gold_Die')
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mats['gold'])
    die = bpy.data.objects.new('Gold_Die', me)
    link(die, coll)
    # pips via boolean difference with spheres
    cutters = bpy.data.collections.new('die_cutters')
    bpy.context.scene.collection.children.link(cutters)
    pr = e * 0.085
    spacing = e * 0.25
    centers = []
    for nrm, num in DIE_FACES.items():
        n = np.array(nrm, float)
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.5 else np.array([0, 1.0, 0])
        b = np.cross(n, a)
        for (i, j) in PIP_LAYOUT[num]:
            c = n * (h + pr * 0.35) + a * i * spacing + b * j * spacing
            centers.append(c)
            cbm = uv_sphere(pr, 24, 12)
            cme = bpy.data.meshes.new('cut')
            cbm.to_mesh(cme)
            cbm.free()
            cme.materials.append(mats['pip'])
            co = bpy.data.objects.new('cut', cme)
            co.location = Vector(c)
            cutters.objects.link(co)
    mod = die.modifiers.new('pips', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.operand_type = 'COLLECTION'
    mod.collection = cutters
    mod.solver = 'EXACT'
    mod.material_mode = 'TRANSFER'
    dg = bpy.context.evaluated_depsgraph_get()
    newme = bpy.data.meshes.new_from_object(die.evaluated_get(dg))
    die.modifiers.clear()
    old = die.data
    die.data = newme
    newme.name = 'Gold_Die'
    bpy.data.meshes.remove(old)
    for o in list(cutters.objects):
        m = o.data
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(m)
    bpy.data.collections.remove(cutters)
    for p in newme.polygons:
        p.use_smooth = True
    die.location = Vector(D.DIE_C)
    return die
