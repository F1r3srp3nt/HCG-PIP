"""Pip v3 design - built fresh from reference/pip_v3_target.png (627 x 627 px).

Every part is placed in the *reference camera's* image space: (u, v) are pixel
coordinates in the target image and d is depth in metres (+ = away from the
camera). cam_pt() turns that into Blender world space. The camera is
orthographic, 10 degrees above level, so the render lines up with the artwork.
"""
import math
import numpy as np

IMG = 627
K = 0.0012                    # metres per reference pixel
ORTHO = IMG * K               # camera ortho_scale
ELEV = math.radians(10.0)
RIGHT = np.array([1.0, 0.0, 0.0])
FWD = np.array([0.0, math.cos(ELEV), -math.sin(ELEV)])     # camera view direction
UP = np.cross(RIGHT, FWD)                                   # camera up (screen up)
ORIGIN = np.array([0.0, 0.0, 0.40])                         # world point at image centre, d = 0


def cam_pt(u, v, d=0.0):
    u, v, d = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(d, float))
    return (ORIGIN + RIGHT * ((u - IMG / 2) * K)[..., None] + UP * ((IMG / 2 - v) * K)[..., None]
            + FWD * d[..., None])


def px(n):
    return n * K


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# Pip faces the viewer, turned toward screen-right; his head turns a touch more.
FACING = unit(-FWD + 0.42 * RIGHT)

# ------------------------------------------------------------------ landmarks (pixels)
EYES = {   # screen-left eye is Pip's right eye ('R'), screen-right is his left ('L')
    'R': dict(box=(274, 190, 384, 268), iris=(333, 237, 31, 31), pupil=(334, 236, 16, 17),
              hi=[(330, 221, 6.5), (347, 249, 3.0)], flick=-1),
    'L': dict(box=(426, 181, 492, 254), iris=(456, 221, 23, 29), pupil=(456, 219, 12, 16),
              hi=[(448, 207, 6.0), (463, 233, 2.6)], flick=+1),
}
NOSE = (428, 259)
MOUTH_BOX = (360, 262, 452, 330)

# ------------------------------------------------------------------ SDF helpers
def sd_ellipsoid(p, c, r):
    q = p - c
    k0 = np.linalg.norm(q / r, axis=-1)
    k1 = np.linalg.norm(q / (np.asarray(r) ** 2), axis=-1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


def sd_round_cone(p, a, b, ra, rb):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    l2 = ba @ ba
    rr = ra - rb
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    x = pa * l2 - y[..., None] * ba
    x2 = (x * x).sum(-1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - ra
    d2 = np.sqrt(x2 + y2) * il2 - ra
    d1 = np.sqrt(x2 + z2) * il2 - rb
    return np.where(np.sign(z) * a2 * z2 > k, d1, np.where(np.sign(y) * a2 * y2 < k, d2, d3))


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def ell_cam(c_uvd, r_px, depth_r):
    """Ellipsoid whose radii follow the camera axes (right, up, view)."""
    c = cam_pt(*c_uvd)
    R = np.stack([RIGHT, UP, FWD], 1)          # columns
    r = np.array([px(r_px[0]), px(r_px[1]), depth_r])

    def f(p):
        q = (p - c) @ R
        return sd_ellipsoid(q, np.zeros(3), r)
    return f


def tuft_cam(c_uvd, dir_uv, len_px, wid_px, depth_r):
    """Flattened, oriented ellipsoid for a swept lock of fur (long axis in the image plane)."""
    c = cam_pt(*c_uvd)
    du = unit(np.array(dir_uv, float))
    a = unit(RIGHT * du[0] - UP * du[1])
    b = unit(np.cross(FWD, a))
    R = np.stack([a, b, FWD], 1)
    r = np.array([px(len_px), px(wid_px), depth_r])
    return lambda p: sd_ellipsoid((p - c) @ R, np.zeros(3), r)


def cone_cam(a_uvd, b_uvd, ra_px, rb_px):
    a, b = cam_pt(*a_uvd), cam_pt(*b_uvd)
    return lambda p: sd_round_cone(p, a, b, px(ra_px), px(rb_px))


# ------------------------------------------------------------------ the body
# Primitive radii are the art's silhouette minus the fur length on top.
TAIL = [((262, 548, 0.10), 38), ((178, 570, 0.15), 70), ((110, 508, 0.17), 84), ((96, 414, 0.17), 80),
        ((120, 340, 0.15), 62), ((160, 296, 0.12), 34), ((188, 278, 0.10), 9)]


def primitives():
    P = []

    def add(name, bone, label, k, fn):
        P.append((name, bone, label, k, fn))

    # head
    add('cranium', 'head', 'head', 0.05, ell_cam((372, 198, 0.0), (116, 100), 0.125))
    add('crown', 'head', 'crown', 0.04, ell_cam((370, 142, 0.02), (80, 48), 0.090))
    add('jowl', 'head', 'head', 0.04, ell_cam((392, 262, -0.045), (92, 50), 0.090))
    add('muzzle', 'head', 'muzzle', 0.028, ell_cam((414, 272, -0.105), (38, 27), 0.050))
    add('snout', 'head', 'muzzle', 0.02, ell_cam((426, 258, -0.13), (21, 16), 0.032))
    add('cheek.R', 'head', 'cheek', 0.04, ell_cam((268, 266, -0.01), (40, 36), 0.070))
    add('cheek.L', 'head', 'cheek', 0.04, ell_cam((494, 248, 0.05), (26, 28), 0.060))
    # sculpted crest locks sweeping up and to screen-left
    for c, dr, ln, wd in (((352, 122, -0.06), (-0.55, -0.83), 50, 22), ((306, 142, -0.07), (-0.95, -0.30), 50, 20),
                          ((392, 116, -0.05), (-0.25, -0.97), 40, 19), ((270, 170, -0.06), (-0.99, 0.12), 36, 17),
                          ((426, 126, -0.03), (0.25, -0.97), 30, 15), ((334, 166, -0.10), (-0.8, 0.35), 34, 15),
                          ((372, 150, -0.10), (-0.6, -0.5), 34, 16)):
        add('lock', 'head', 'crown', 0.025, tuft_cam(c, dr, ln, wd, 0.028))
    # neck / torso / haunches
    add('neck', 'chest', 'chest', 0.05, ell_cam((382, 338, 0.06), (82, 40), 0.09))
    add('chest', 'chest', 'chest', 0.06, ell_cam((392, 420, 0.075), (112, 96), 0.11))
    add('belly', 'hips', 'body', 0.06, ell_cam((394, 510, 0.07), (125, 70), 0.12))
    add('haunch.R', 'thigh.R', 'leg', 0.05, ell_cam((300, 520, 0.05), (62, 62), 0.10))
    add('haunch.L', 'thigh.L', 'leg', 0.05, ell_cam((500, 508, 0.10), (52, 58), 0.09))
    # feet with toe lobes
    add('foot.R', 'foot.R', 'foot', 0.025, ell_cam((320, 575, -0.075), (46, 24), 0.065))
    add('foot.L', 'foot.L', 'foot', 0.025, ell_cam((494, 571, -0.035), (32, 22), 0.058))
    for k, (uu, dd) in enumerate(((290, -0.125), (318, -0.13), (346, -0.125))):
        add(f'toe{k}.R', 'foot.R', 'foot', 0.010, ell_cam((uu, 586, dd), (13, 15), 0.018))
    for k, (uu, dd) in enumerate(((476, -0.085), (498, -0.088), (518, -0.082))):
        add(f'toe{k}.L', 'foot.L', 'foot', 0.010, ell_cam((uu, 584, dd), (10, 14), 0.016))
    # arms: his right arm (screen-left) wraps to the die's left face,
    # his left arm (screen-right) climbs the die's right side to its top corner
    add('upper_arm.R', 'upper_arm.R', 'arm', 0.03, cone_cam((300, 372, 0.02), (300, 430, -0.05), 36, 33))
    add('forearm.R', 'forearm.R', 'arm', 0.02, cone_cam((300, 430, -0.05), (338, 420, -0.12), 33, 30))
    add('paw.R', 'paw.R', 'paw', 0.014, ell_cam((342, 420, -0.135), (36, 32), 0.036))
    add('upper_arm.L', 'upper_arm.L', 'arm', 0.03, cone_cam((496, 362, 0.07), (526, 440, 0.03), 31, 29))
    add('forearm.L', 'forearm.L', 'arm', 0.02, cone_cam((526, 440, 0.03), (515, 392, -0.04), 29, 27))
    add('paw.L', 'paw.L', 'paw', 0.014, ell_cam((515, 384, -0.05), (33, 30), 0.036))
    # tail
    for i in range(len(TAIL) - 1):
        (a, ra), (b, rb) = TAIL[i], TAIL[i + 1]
        add(f'tail.{i + 1:02d}', f'tail.{i + 1:02d}', 'tail', 0.04 if i == 0 else 0.025,
            cone_cam(a, b, ra, rb))
    return P


def body_sdf(p):
    prims = primitives()
    d = prims[0][4](p)
    for name, bone, label, k, fn in prims[1:]:
        d = smin(d, fn(p), k)
    return d


# ------------------------------------------------------------------ ears (cupped shells)
EARS = {   # base (u, v, d), tip (u, v, d), max half-width px, front tilt toward camera
    'R': dict(base=(254, 214, 0.03), tip=(196, 22, 0.05), hw=86, side=-1),
    'L': dict(base=(468, 172, 0.07), tip=(496, 18, 0.09), hw=70, side=+1),
}
EAR_CUP = 0.020


def ear_frame(e):
    B, T = cam_pt(*e['base']), cam_pt(*e['tip'])
    A = unit(T - B)
    N = unit(-FWD - A * (-FWD @ A) + 0.25 * RIGHT * e['side'])   # inner face toward camera
    N = unit(N - A * (N @ A))
    S = unit(np.cross(A, N))
    if (S @ RIGHT) * e['side'] < 0:
        S = -S
    return B, T, A, S, N, np.linalg.norm(T - B)


def ear_halfwidth(e, u):
    u = np.asarray(u, float)
    w = np.where(u < 0.36, 0.86 + 0.14 * np.clip(u / 0.36, 0, 1),
                 np.clip(1 - (np.clip(u - 0.36, 0, None) / 0.64) ** 1.9, 0, 1) ** 0.62)
    return px(e['hw']) * w + 0.003 * (1 - u)


def ear_point(e, u, X, Y):
    B, T, A, S, N, L = ear_frame(e)
    u = np.asarray(u)[..., None]
    X = np.asarray(X)[..., None]
    Y = np.asarray(Y)[..., None]
    hw = ear_halfwidth(e, u)
    th = 0.010 + 0.016 * (1 - X ** 2) * (1 - 0.6 * u)
    cup = EAR_CUP * (1 - X ** 2) * np.sin(np.pi * np.clip(u * 1.05, 0, 1)) ** 0.8
    return B + u * L * A + X * hw * S + (Y * th / 2 - cup) * N


def ear_mesh(e, nu=80, nv=56):
    verts, params = [], []
    us = np.linspace(-0.12, 1.0, nu)
    ring = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    B, T, A, S, N, L = ear_frame(e)
    for u in us[:-1]:
        X, Y = np.cos(ring), np.sin(ring)
        P = ear_point(e, np.full(nv, max(u, 0.0)), X, Y)
        if u < 0:
            P = P + u * L * A
        verts.append(P)
        params.append(np.stack([np.full(nv, u), X, Y], -1))
    verts = np.concatenate(verts + [ear_point(e, 1.0, 0.0, 0.0)[None]])
    params = np.concatenate(params + [np.array([[1.0, 0, 0]])])
    nr = nu - 1
    faces = []
    for i in range(nr - 1):
        for j in range(nv):
            a, b = i * nv + j, i * nv + (j + 1) % nv
            faces.append((a, b, b + nv, a + nv))
    t = len(verts) - 1
    for j in range(nv):
        faces.append(((nr - 1) * nv + j, (nr - 1) * nv + (j + 1) % nv, t))
    faces.append(tuple(range(nv - 1, -1, -1)))
    return verts, faces, params


# ------------------------------------------------------------------ die
DIE_EDGE = 0.166
DIE_C = (425, 432, -0.135)
# yaw about world Z, then pitch toward camera, then a small roll: the art shows the
# top face (1), a broad left face and a narrow right face, tipped toward his left paw
DIE_EULER_DEG = (24.0, 8.0, -38.0)

# ------------------------------------------------------------------ scarf (image-space paths)
SCARF_RING = dict(c=(384, 334, 0.065), rx=122, rz=0.098, tube=27, tilt_px=8)
SCARF_KNOT = (282, 352, -0.04)
SCARF_TAILS = [
    [(280, 356, -0.05), (250, 372, -0.045), (220, 388, -0.04), (192, 400, -0.035), (168, 404, -0.03)],
    [(284, 364, -0.05), (270, 396, -0.05), (252, 424, -0.05), (226, 446, -0.045), (198, 460, -0.04)],
]
SCARF_TAIL_W = [[38, 44, 48, 50, 46], [40, 48, 54, 58, 54]]
