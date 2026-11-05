"""Pip design data: proportions, landmarks and the signed-distance body.

Everything here is plain numpy so it can be tuned and re-run quickly.
Units are metres, Pip faces -Y, Z is up, +X is Pip's own left side.
"""
import numpy as np

# ---------------------------------------------------------------- landmarks
# The head was blocked out around HC0 and is re-scaled/re-seated with H() so
# the chibi proportions (big head, short plump body) are one knob to turn.
HC0 = np.array([0.0, -0.045, 0.400])
HC1 = np.array([0.0, -0.040, 0.368])
HS = 1.14


def H(p):
    return HC1 + (np.asarray(p, float) - HC0) * HS


EYE_R = 0.0385 * HS
EYE_C = {s: H([s * 0.054, -0.110, 0.398]) for s in (1, -1)}
EYE_GAZE = {s: (lambda v: v / np.linalg.norm(v))(np.array([s * 0.20, -1.0, 0.03])) for s in (1, -1)}
NOSE_C = H([0.0, -0.186, 0.378])
MOUTH_C = H([0.0, -0.176, 0.347])
MOUTH_R = np.array([0.031, 0.022, 0.0175]) * HS
MOUTH_TOP = H([0, 0, 0.356])[2]
DIE_EDGE = 0.098
DIE_C = np.array([0.0, -0.172, 0.168])

# tail spine (Pip's right side, curling up behind him)
TAIL_PTS = np.array([
    [-0.020, 0.095, 0.060],
    [-0.080, 0.150, 0.072],
    [-0.158, 0.160, 0.110],
    [-0.214, 0.138, 0.178],
    [-0.236, 0.100, 0.250],
    [-0.222, 0.058, 0.306],
    [-0.192, 0.030, 0.332],
])
TAIL_RAD = np.array([0.034, 0.056, 0.072, 0.074, 0.064, 0.046, 0.018])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ear_frame(side):
    """Ear base point, length axis A, across axis S, front normal N."""
    base = H([side * 0.070, -0.014, 0.455])
    A = _unit([side * 0.50, 0.16, 0.85])
    N = _unit([side * 0.30, -1.0, 0.10])
    N = _unit(N - A * (N @ A))
    S = np.cross(A, N) * side          # points away from the head centre-line
    if S[0] * side < 0:
        S = -S
    return base, A, S, N


EAR_LEN = 0.255
EAR_W = 0.086        # max half width
EAR_T = 0.016        # max thickness
EAR_CUP = 0.022      # how deep the inner ear is cupped


def ear_halfwidth(u):
    w = (1 - u) ** 0.85 * (0.70 + 1.5 * u)
    return EAR_W * w / 0.98


def ear_point(side, u, X, Y):
    """u in [0,1] base->tip, X in [-1,1] across, Y in [-1,1] back->front."""
    base, A, S, N = ear_frame(side)
    u = np.asarray(u)[..., None]
    X = np.asarray(X)[..., None]
    Y = np.asarray(Y)[..., None]
    hw = ear_halfwidth(u)
    th = 0.0085 + EAR_T * (1 - X ** 2) * (1 - 0.7 * u)
    cup = EAR_CUP * (1 - X ** 2) * np.sin(np.pi * np.clip(u * 1.08, 0, 1)) ** 0.8
    bend = 0.035 * u ** 2            # tip leans slightly back
    flare = 0.020 * u ** 2           # and slightly outward
    return (base + u * EAR_LEN * A + X * hw * S + (Y * th / 2 - cup - bend) * N
            + flare * S * side * 0 + flare * np.array([side, 0, 0]))


def ear_mesh(side, nu=72, nv=48):
    """Closed ear mesh built from a parametrised squashed sphere."""
    verts, params = [], []
    us = np.linspace(-0.08, 1.0, nu)
    ring = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    for u in us[:-1]:
        X = np.cos(ring)
        Y = np.sin(ring)
        P = ear_point(side, np.full(nv, max(u, 0.0)), X, Y)
        if u < 0:   # extend the base straight down into the head
            base, A, S, N = ear_frame(side)
            P = P + u * EAR_LEN * A
        verts.append(P)
        params.append(np.stack([np.full(nv, u), X, Y], -1))
    tip = ear_point(side, 1.0, 0.0, 0.0)
    verts = np.concatenate(verts + [tip[None]])
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
    faces.append(tuple(range(nv - 1, -1, -1)))      # base cap
    if side < 0:
        faces = [f[::-1] for f in faces]
    return verts, faces, params


# ------------------------------------------------------------- SDF helpers
def sd_ellipsoid(p, c, r):
    q = (p - c)
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


# limb joints (shared with the rig): shoulder, elbow, wrist, paw tip / hip, knee, ankle, toe
def arm_joints(s):
    return (np.array([s * 0.058, -0.055, 0.232]), np.array([s * 0.080, -0.088, 0.152]),
            np.array([s * 0.075, -0.150, 0.180]), np.array([s * 0.074, -0.194, 0.192]))


def leg_joints(s):
    return (np.array([s * 0.074, 0.020, 0.088]), np.array([s * 0.076, -0.090, 0.084]),
            np.array([s * 0.076, -0.076, 0.030]), np.array([s * 0.073, -0.176, 0.020]))


def primitives():
    """(name, bone, label, k, sdf-fn). Mirrored parts are expanded."""
    P = []

    def E(name, bone, label, k, c, r, head=False):
        if head:
            c, r = H(c), np.array(r) * HS
        P.append((name, bone, label, k, lambda p, c=np.array(c), r=np.array(r): sd_ellipsoid(p, c, r)))

    def C(name, bone, label, k, a, b, ra, rb):
        P.append((name, bone, label, k, lambda p, a=a, b=b, ra=ra, rb=rb: sd_round_cone(p, a, b, ra, rb)))

    E('pelvis', 'hips', 'body', 0.04, (0, 0.025, 0.088), (0.112, 0.102, 0.088))
    E('belly', 'spine', 'body', 0.045, (0, -0.018, 0.150), (0.093, 0.083, 0.088))
    E('chest', 'chest', 'chest', 0.035, (0, -0.045, 0.210), (0.077, 0.060, 0.058))
    E('neck', 'neck', 'neck', 0.035, (0, -0.035, 0.256), (0.064, 0.058, 0.045))
    E('head', 'head', 'head', 0.045, (0, -0.045, 0.400), (0.128, 0.112, 0.106), True)
    E('crown', 'head', 'crown', 0.035, (0, -0.010, 0.448), (0.092, 0.082, 0.058), True)
    E('muzzle', 'head', 'muzzle', 0.026, (0, -0.133, 0.366), (0.047, 0.050, 0.033), True)
    E('bridge', 'head', 'muzzle', 0.022, (0, -0.122, 0.392), (0.032, 0.050, 0.026), True)
    E('chin', 'jaw', 'chin', 0.020, (0, -0.126, 0.334), (0.042, 0.044, 0.022), True)
    for s, sx in ((1, 'L'), (-1, 'R')):
        E(f'cheek.{sx}', 'head', 'cheek', 0.030, (s * 0.094, -0.078, 0.358), (0.056, 0.050, 0.044), True)
        E(f'thigh.{sx}', f'thigh.{sx}', 'leg', 0.030, (s * 0.076, -0.035, 0.066), (0.057, 0.080, 0.062))
        E(f'foot.{sx}', f'foot.{sx}', 'foot', 0.020, (s * 0.073, -0.128, 0.024), (0.039, 0.060, 0.024))
        sh, el, wr, tip = arm_joints(s)
        C(f'upper_arm.{sx}', f'upper_arm.{sx}', 'arm', 0.025, sh, el, 0.031, 0.027)
        C(f'forearm.{sx}', f'forearm.{sx}', 'arm', 0.014, el, wr, 0.027, 0.025)
        E(f'paw.{sx}', f'paw.{sx}', 'paw', 0.012, wr + (tip - wr) * 0.45, (0.026, 0.031, 0.028))
    for i in range(len(TAIL_PTS) - 1):
        C(f'tail.{i + 1:02d}', f'tail.{i + 1:02d}', 'tail', 0.03 if i == 0 else 0.02,
          TAIL_PTS[i], TAIL_PTS[i + 1], TAIL_RAD[i], TAIL_RAD[i + 1])
    return P


def sd_mouth(p):
    """Open 'D' smile: rounded bottom, flat upper lip line."""
    x2 = (p[..., 0] - MOUTH_C[0]) ** 2
    q = p.copy()
    q[..., 2] = q[..., 2] - 7.0 * x2          # corners curl up into a smile
    return np.maximum(sd_ellipsoid(q, MOUTH_C, MOUTH_R), q[..., 2] - MOUTH_TOP)


def carves():
    out = []
    for s in (1, -1):
        out.append(('eye_socket', 0.010, lambda p, c=EYE_C[s]: np.linalg.norm(p - c, axis=-1) - EYE_R * 1.02))
    out.append(('mouth', 0.005, sd_mouth))
    return out


def body_sdf(p, return_parts=False):
    prims = primitives()
    parts = [fn(p) for (_, _, _, _, fn) in prims]
    d = parts[0]
    for (name, bone, label, k, fn), dp in zip(prims[1:], parts[1:]):
        d = smin(d, dp, k)
    for name, k, fn in carves():
        d = -smin(-d, fn(p), k)
    if return_parts:
        return d, prims, parts
    return d
