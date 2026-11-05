"""Pip design data: proportions, landmarks and the signed-distance body.

Everything here is plain numpy so it can be tuned and re-run quickly.
Units are metres, Pip faces -Y, Z is up, +X is Pip's own left side.
"""
import numpy as np

# ---------------------------------------------------------------- landmarks
# The head was blocked out around HC0 and is re-scaled/re-seated with H() so
# the chibi proportions (big head, short plump body) are one knob to turn.
HC0 = np.array([0.0, -0.045, 0.400])
HC1 = np.array([0.0, -0.040, 0.352])
HS = 1.22


def H(p):
    return HC1 + (np.asarray(p, float) - HC0) * HS


EYE_R = 0.0455 * HS
EYE_C = {s: H([s * 0.058, -0.097, 0.397]) for s in (1, -1)}
EYE_GAZE = {s: (lambda v: v / np.linalg.norm(v))(np.array([s * 0.20, -1.0, 0.03])) for s in (1, -1)}
NOSE_C = H([0.0, -0.180, 0.385])
# v2 mouth: a relaxed, closed slit (a thin gap so the jaw and lip corners can
# open it into a grin). MOUTH_C sits on the front of the muzzle.
MOUTH_C = H([0.0, -0.170, 0.354])
MOUTH_HALF_W = 0.033 * HS          # half width of the mouth line
MOUTH_GAP = 0.0030                 # half thickness of the slit
MOUTH_DEPTH = 0.024                # how far the slit cuts back into the muzzle
MOUTH_SMILE = 6.5                  # corners curve up: z += MOUTH_SMILE * x^2
DIE_EDGE = 0.160
DIE_C = np.array([0.0, -0.240, 0.165])

# tail spine (Pip's right side, curling up behind him)
TAIL_PTS = np.array([
    [-0.020, 0.095, 0.060],
    [-0.080, 0.150, 0.072],
    [-0.158, 0.160, 0.110],
    [-0.214, 0.138, 0.178],
    [-0.238, 0.100, 0.246],
    [-0.226, 0.064, 0.298],
    [-0.200, 0.040, 0.326],
])
TAIL_RAD = np.array([0.036, 0.060, 0.076, 0.078, 0.066, 0.044, 0.012])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ear_frame(side):
    """Ear base point, length axis A, across axis S, front normal N."""
    base = H([side * 0.070, -0.020, 0.446])
    A = _unit([side * 0.52, 0.12, 0.85])
    N = _unit([side * 0.10, -1.0, 0.04])
    N = _unit(N - A * (N @ A))
    S = np.cross(A, N) * side          # points away from the head centre-line
    if S[0] * side < 0:
        S = -S
    return base, A, S, N


EAR_LEN = 0.262
EAR_W = 0.094        # max half width
EAR_T = 0.016        # max thickness
EAR_CUP = 0.017      # how deep the inner ear is cupped


def ear_halfwidth(u):
    """v2: broad rounded leaf - stays wide to mid-length, then rounds to a soft point."""
    u = np.asarray(u, float)
    lo = 0.97 + 0.03 * np.clip(u / 0.36, 0, 1)
    hi = np.sqrt(np.clip(1 - (np.clip(u - 0.36, 0, None) / 0.64) ** 1.8, 0, 1)) ** 0.9
    return EAR_W * np.where(u < 0.36, lo, hi) + 0.004 * (1 - u)


def ear_point(side, u, X, Y):
    """u in [0,1] base->tip, X in [-1,1] across, Y in [-1,1] back->front."""
    base, A, S, N = ear_frame(side)
    u = np.asarray(u)[..., None]
    X = np.asarray(X)[..., None]
    Y = np.asarray(Y)[..., None]
    hw = ear_halfwidth(u)
    th = 0.0095 + EAR_T * (1 - X ** 2) * (1 - 0.6 * u)
    cup = EAR_CUP * (1 - X ** 2) * np.sin(np.pi * np.clip(u * 1.08, 0, 1)) ** 0.8
    bend = 0.022 * u ** 2            # tip leans slightly back
    flare = 0.012 * u ** 2           # and slightly outward
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
    return (np.array([s * 0.068, -0.060, 0.222]), np.array([s * 0.114, -0.114, 0.150]),
            np.array([s * 0.112, -0.206, 0.194]), np.array([s * 0.106, -0.252, 0.212]))


def leg_joints(s):
    return (np.array([s * 0.074, 0.020, 0.088]), np.array([s * 0.076, -0.090, 0.084]),
            np.array([s * 0.076, -0.076, 0.030]), np.array([s * 0.073, -0.182, 0.020]))


def primitives():
    """(name, bone, label, k, sdf-fn). Mirrored parts are expanded."""
    P = []

    def E(name, bone, label, k, c, r, head=False):
        if head:
            c, r = H(c), np.array(r) * HS
        P.append((name, bone, label, k, lambda p, c=np.array(c), r=np.array(r): sd_ellipsoid(p, c, r)))

    def C(name, bone, label, k, a, b, ra, rb):
        P.append((name, bone, label, k, lambda p, a=a, b=b, ra=ra, rb=rb: sd_round_cone(p, a, b, ra, rb)))

    E('pelvis', 'hips', 'body', 0.04, (0, 0.025, 0.088), (0.118, 0.106, 0.090))
    E('belly', 'spine', 'body', 0.045, (0, -0.016, 0.150), (0.098, 0.086, 0.090))
    E('chest', 'chest', 'chest', 0.035, (0, -0.045, 0.200), (0.079, 0.062, 0.058))
    E('neck', 'neck', 'neck', 0.035, (0, -0.035, 0.240), (0.066, 0.060, 0.045))
    E('head', 'head', 'head', 0.045, (0, -0.045, 0.402), (0.122, 0.112, 0.114), True)
    E('crown', 'head', 'crown', 0.035, (0, -0.020, 0.454), (0.086, 0.078, 0.066), True)
    E('muzzle', 'head', 'muzzle', 0.026, (0, -0.132, 0.368), (0.046, 0.048, 0.032), True)
    E('bridge', 'head', 'muzzle', 0.022, (0, -0.122, 0.394), (0.030, 0.050, 0.024), True)
    E('chin', 'jaw', 'chin', 0.020, (0, -0.124, 0.340), (0.038, 0.040, 0.020), True)
    for s, sx in ((1, 'L'), (-1, 'R')):
        E(f'cheek.{sx}', 'head', 'cheek', 0.030, (s * 0.096, -0.074, 0.356), (0.056, 0.050, 0.044), True)
        E(f'thigh.{sx}', f'thigh.{sx}', 'leg', 0.030, (s * 0.076, -0.035, 0.066), (0.057, 0.080, 0.062))
        E(f'foot.{sx}', f'foot.{sx}', 'foot', 0.020, (s * 0.074, -0.130, 0.026), (0.044, 0.064, 0.026))
        for k, dx in enumerate((-0.022, 0.0, 0.022)):      # three soft toes
            E(f'toe{k}.{sx}', f'foot.{sx}', 'foot', 0.008, (s * 0.074 + dx, -0.182, 0.018),
              (0.0135, 0.016, 0.0165))
        sh, el, wr, tip = arm_joints(s)
        C(f'upper_arm.{sx}', f'upper_arm.{sx}', 'arm', 0.025, sh, el, 0.036, 0.031)
        C(f'forearm.{sx}', f'forearm.{sx}', 'arm', 0.014, el, wr, 0.031, 0.028)
        E(f'paw.{sx}', f'paw.{sx}', 'paw', 0.012, wr + (tip - wr) * 0.45, (0.028, 0.036, 0.033))
    # sculpted hair locks: the swept crest between the ears and spiky cheek tufts
    for k, (b, t, rb) in enumerate((((0.000, -0.060, 0.505), (-0.050, -0.098, 0.548), 0.022),
                                    ((0.030, -0.030, 0.512), (-0.006, -0.070, 0.566), 0.020),
                                    ((-0.030, -0.040, 0.508), (-0.080, -0.072, 0.530), 0.018),
                                    ((0.045, -0.075, 0.490), (0.030, -0.120, 0.512), 0.015))):
        C(f'lock{k}', 'head', 'crown', 0.018, H(b), H(t), rb * HS, 0.004)
    for s, sx in ((1, 'L'), (-1, 'R')):
        for k, (b, t, rb) in enumerate((((0.120, -0.075, 0.372), (0.172, -0.070, 0.352), 0.020),
                                        ((0.118, -0.065, 0.340), (0.160, -0.055, 0.306), 0.018))):
            C(f'ctuft{k}.{sx}', 'head', 'cheek', 0.016, H((s * b[0], b[1], b[2])), H((s * t[0], t[1], t[2])),
              rb * HS, 0.004)
    for i in range(len(TAIL_PTS) - 1):
        C(f'tail.{i + 1:02d}', f'tail.{i + 1:02d}', 'tail', 0.03 if i == 0 else 0.02,
          TAIL_PTS[i], TAIL_PTS[i + 1], TAIL_RAD[i], TAIL_RAD[i + 1])
    return P


def mouth_z(x):
    return MOUTH_C[2] + MOUTH_SMILE * (x - MOUTH_C[0]) ** 2


MOUTH_OPEN_H = 0.021              # how far the grin opens below the smile line


def mouth_open_bottom(x):
    k = np.clip(1 - (np.asarray(x) / MOUTH_HALF_W) ** 2, 0, 1)
    return mouth_z(x) - MOUTH_OPEN_H * k ** 0.6


def mouth_region(p, pad=0.0):
    """Area of the muzzle the open grin can cover (no fur grows here)."""
    x, z = p[..., 0], p[..., 2]
    return ((np.abs(x) < MOUTH_HALF_W + pad) & (z < mouth_z(x) + pad)
            & (z > mouth_open_bottom(x) - pad) & (p[..., 1] < MOUTH_C[1] + 0.022))


def carves():
    out = []
    for s in (1, -1):
        out.append(('eye_socket', 0.008, lambda p, c=EYE_C[s]: np.linalg.norm(p - c, axis=-1) - EYE_R * 1.10))
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
