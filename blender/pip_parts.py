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


UP_LID_CENTER, UP_LID_SIDE = 44.0, 8.0      # degrees: arched upper lid edge
LOW_LID_CENTER, LOW_LID_SIDE = -52.0, -30.0


def lid_edge(az, upper=True):
    """Lid edge elevation (radians) as a function of azimuth: an arch."""
    c, sd = (UP_LID_CENTER, UP_LID_SIDE) if upper else (LOW_LID_CENTER, LOW_LID_SIDE)
    k = min(1.0, (az / math.radians(95)) ** 2)
    return math.radians(c + (sd - c) * k)


def lid_mesh(s, upper=True):
    """A spherical lid shell with an arched cut edge. Rotating it about the
    eye's side axis (the lid bone's local X) opens and closes it."""
    f, u, r = eye_frame(s)
    R0 = D.EYE_R * 1.07
    th = 0.0028
    nel, naz = 22, 56
    verts, faces = [], []
    sign = 1 if upper else -1
    azs = np.linspace(-math.radians(100), math.radians(100), naz)
    edge_w = []
    for layer, rad in enumerate((R0 + th, R0)):
        for i in range(nel):
            k = i / (nel - 1)
            for j, az in enumerate(azs):
                e0 = lid_edge(az, upper)
                el = e0 + (sign * math.radians(118) - e0) * k
                d = math.cos(el) * (math.cos(az) * f + math.sin(az) * r) + math.sin(el) * u
                verts.append(D.EYE_C[s] + rad * d)
                edge_w.append(1.0 - min(1.0, abs(el - e0) / math.radians(9)))
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
        at.data.foreach_set('value', ew.astype(np.float32) * (1.0 if upper else 0.55))
        me.materials.append(mats['lid'])
        ob = link(bpy.data.objects.new(name, me), coll)
        out['up' if upper else 'low'] = ob
    # bold upper lash line + the little flick at the outer corner
    f, u, r = eye_frame(s)
    pts, rad = [], []
    for az in np.linspace(-math.radians(55), math.radians(84), 34):
        el = lid_edge(az, True) + math.radians(1.5)
        d = math.cos(el) * (math.cos(az) * f - math.sin(az) * r * s) + math.sin(el) * u
        pts.append(D.EYE_C[s] + D.EYE_R * 1.12 * d)
        k = (az + math.radians(55)) / math.radians(139)
        rad.append(0.0011 + 0.0034 * min(1.0, k * 1.5) ** 0.8)
    # flick: continue outward and up from the outer corner
    last = pts[-1]
    outward = -r * s
    for i, k in enumerate((0.3, 0.6, 0.9)):
        pts.append(last + outward * 0.013 * k + u * 0.008 * k ** 1.5 + f * -0.003 * k)
        rad.append(0.0040 * (1 - k) + 0.0005)
    v, fc = tube(pts, rad, 8)
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
        w = 0.0175 * D.HS * (1.0 + 0.32 * z)
        v.co = Vector((x * w, y * 0.0125 * D.HS, z * 0.0115 * D.HS))
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
        v.co = Vector((x * 0.022, y * 0.018, z * 0.006 + 0.004 * (x * x)))
    me = bpy.data.meshes.new('Pip_Tongue')
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mats['tongue'])
    ob = bpy.data.objects.new('Pip_Tongue', me)
    ob.location = Vector(D.MOUTH_C + np.array([0, 0.009, -0.012]))
    ob.rotation_euler = (math.radians(-18), 0, 0)
    return link(ob, coll)


# ------------------------------------------------------------------ scarf
SCARF_C = np.array([0.0, -0.038, 0.256])
SCARF_TILT = math.radians(14)        # front lower than back


def scarf_ring_point(theta, phi, rx=0.084, ry=0.076, rt=0.027):
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


_SZ = np.array([0, 0, -0.024])
SCARF_TAIL_A = np.array([[-0.052, -0.110, 0.262], [-0.090, -0.118, 0.236], [-0.126, -0.110, 0.204],
                         [-0.156, -0.094, 0.174], [-0.182, -0.076, 0.150]]) + _SZ
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
            (SCARF_TAIL_A, [0.050, 0.058, 0.064, 0.066, 0.062], (0.3, -1, 0.3), 0.5, 'tail_a'),
            (SCARF_TAIL_B, [0.046, 0.054, 0.060, 0.062, 0.058], (0.2, -0.5, 1), -0.6, 'tail_b')):
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
