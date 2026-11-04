"""Pip's fur colour map and grooming fields (all evaluated on the rest pose)."""
import numpy as np
import pip_design as D


def hexlin(h):
    h = h.lstrip('#')
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


WHITE = hexlin('#F4F2EF')
BLUE_TINT = hexlin('#BCCBEF')
BLUE_MARK = hexlin('#8DA5E2')
EAR_INNER = hexlin('#6884CF')
EAR_INNER_LIGHT = hexlin('#8EA5E2')
CYAN = hexlin('#5AD8EA')
CYAN_DEEP = hexlin('#2BB3D8')
MOUTH = hexlin('#5B1822')
PAD = hexlin('#B6AFC9')


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def tangent(v, n):
    return unit(v - n * (v * n).sum(-1, keepdims=True))


def classify(an, n_body):
    """Label per vertex from the dominant SDF primitive / ear shell."""
    dist = an['dist']
    names = an['names']
    labels = np.array([p[2] for p in an['prims']])
    lab = labels[np.argmin(dist, axis=1)].astype(object)
    nv = dist.shape[0]
    is_ear = np.zeros(nv, bool)
    is_ear[n_body:] = True
    return lab, is_ear


def fields(co, nrm, an, lab, is_ear, ear_side, ear_par, rng):
    """Returns colour, fur length, lift, comb direction, clump, density."""
    N = len(co)
    col = np.tile(WHITE, (N, 1))
    length = np.full(N, 0.022)
    lift = np.full(N, 0.45)
    clump = np.full(N, 0.6)
    dens = np.ones(N)
    up = np.array([0, 0, 1.0])
    back = np.array([0, 1.0, 0])
    comb = tangent(np.tile(-up + 0.4 * back, (N, 1)), nrm)

    def L(*names):
        m = np.zeros(N, bool)
        for nm in names:
            m |= (lab == nm)
        return m & ~is_ear

    # ---- body blue wash on the back / outsides
    back_f = sstep(0.0, 0.9, nrm @ unit(np.array([0, 1.0, 0.35])))
    m = L('body', 'leg', 'foot', 'arm', 'neck')
    col[m] = col[m] + (BLUE_TINT - WHITE) * (0.55 * back_f[m])[:, None]
    side_f = sstep(0.3, 0.95, np.abs(nrm[:, 0])) * (lab == 'leg')
    col = col + (BLUE_TINT - WHITE) * (0.35 * side_f)[:, None]

    # ---- head
    hm = L('head', 'crown')
    comb[hm] = tangent(np.tile(back + 0.25 * up, (hm.sum(), 1)), nrm[hm])
    length[hm] = 0.019
    head_back = sstep(-0.2, 0.8, nrm @ unit(np.array([0, 1.0, 0.25])))
    col[hm] = col[hm] + (BLUE_TINT - WHITE) * (0.45 * head_back[hm])[:, None]
    # crown crest: long, tufty, swept up and back
    zc = D.H([0, 0, 0.455])[2]
    crest = hm & (co[:, 2] > zc)
    w = sstep(zc, zc + 0.04, co[:, 2])
    length[crest] = 0.019 + 0.024 * w[crest]
    lift[crest] = 0.55
    clump[crest] = 0.8
    comb[crest] = tangent(up + 0.7 * back + 0.5 * np.c_[np.sign(co[crest, 0]), np.zeros((crest.sum(), 2))]
                          + rng.normal(0, 0.35, (crest.sum(), 3)), nrm[crest])
    # forehead fur sweeps up into the crest
    fh = hm & (nrm[:, 1] < -0.35) & (co[:, 2] > D.H([0, 0, 0.425])[2])
    comb[fh] = tangent(np.tile(up + 0.2 * back, (fh.sum(), 1)), nrm[fh])
    length[fh] = 0.016
    # blue brow arcs + feathered streaks above each eye
    for s in (1, -1):
        q = co - D.EYE_C[s]
        qx = s * q[:, 0]
        rho = np.hypot(qx, q[:, 2])
        phi = np.degrees(np.arctan2(q[:, 2], qx))
        arc = (sstep(D.EYE_R + 0.006, D.EYE_R + 0.012, rho) * (1 - sstep(D.EYE_R + 0.018, D.EYE_R + 0.026, rho))
               * sstep(35, 55, phi) * (1 - sstep(125, 150, phi)))
        th = np.degrees(np.arctan2(q[:, 2] + 0.035, qx + 0.004))
        streak = np.zeros(len(co))
        for a0, w0 in ((62, 3.5), (80, 4.5), (98, 3.5)):
            streak = np.maximum(streak, 1 - sstep(w0 * 0.5, w0, np.abs(th - a0)))
        streak *= sstep(D.EYE_R + 0.012, D.EYE_R + 0.02, rho) * (1 - sstep(D.EYE_R + 0.04, D.EYE_R + 0.055, rho))
        mark = np.maximum(arc, 0.9 * streak) * (nrm[:, 1] < 0.15) * hm
        col = col + (BLUE_MARK - col) * (0.9 * mark)[:, None]
    # small blue flame in the middle of the forehead
    q = co - D.H([0, -0.115, 0.455])
    fl = (1 - sstep(0.008, 0.016, np.abs(q[:, 0]) + 0.25 * np.abs(q[:, 2]))) * hm * (nrm[:, 1] < 0)
    col = col + (BLUE_MARK - col) * (0.7 * fl)[:, None]

    # ---- face: short fur, cleared around eyes / nose / mouth
    face = hm & (nrm[:, 1] < -0.55) & (co[:, 2] < D.H([0, 0, 0.44])[2])
    length[face] = 0.011
    lift[face] = 0.25
    m = L('muzzle')
    length[m] = 0.006
    lift[m] = 0.2
    clump[m] = 0.2
    comb[m] = tangent(np.tile(back + 0.3 * up, (m.sum(), 1)), nrm[m])
    m = L('chin')
    length[m] = 0.010
    comb[m] = tangent(np.tile(-up + 0.5 * back, (m.sum(), 1)), nrm[m])
    for s in (1, -1):
        de = np.linalg.norm(co - D.EYE_C[s], axis=1) - D.EYE_R
        dens[de < 0.006] = 0
        near = de < 0.022
        length[near] = np.minimum(length[near], 0.004 + 0.5 * np.maximum(de[near], 0))
        lift[near] = 0.15
        # sweep away from the eye
        away = tangent(co[near] - D.EYE_C[s], nrm[near])
        comb[near] = unit(comb[near] * 0.3 + away)
    dn = np.linalg.norm(co - D.NOSE_C, axis=1)
    dens[dn < 0.024] = 0
    dm = D.sd_mouth(co)
    mouth = dm < 0.0035
    dens[dm < 0.008] = 0
    col[mouth & ~is_ear] = MOUTH

    # ---- cheeks: big outward tufts
    m = L('cheek')
    cheek_out = np.c_[np.sign(co[m, 0]), np.zeros(m.sum()), np.zeros(m.sum())]
    comb[m] = tangent(cheek_out * 1.0 - up * 0.55 + back * 0.35, nrm[m])
    length[m] = 0.048
    lift[m] = 0.35
    clump[m] = 0.85
    # cheek fur that faces forward stays shorter so the face reads
    fwd = m & (nrm[:, 1] < -0.6)
    length[fwd] = 0.02

    # ---- chest bib & neck
    m = L('chest', 'neck')
    comb[m] = tangent(np.tile(-up - 0.25 * back, (m.sum(), 1)), nrm[m])
    length[m] = 0.032
    clump[m] = 0.6
    m = L('body')
    length[m] = 0.029
    # ---- legs / feet / arms / paws
    m = L('leg')
    comb[m] = tangent(np.tile(back * 0.8 - up, (m.sum(), 1)), nrm[m])
    length[m] = 0.027
    m = L('foot')
    comb[m] = tangent(np.tile(-back, (m.sum(), 1)), nrm[m])
    length[m] = 0.017
    lift[m] = 0.35
    m = L('arm')
    comb[m] = tangent(np.tile(-up - 0.6 * back, (m.sum(), 1)), nrm[m])
    length[m] = 0.023
    m = L('paw')
    comb[m] = tangent(np.tile(-back - 0.3 * up, (m.sum(), 1)), nrm[m])
    length[m] = 0.012
    lift[m] = 0.2
    clump[m] = 0.3

    # ---- tail: long, fluffy, cyan tip with a zig-zag edge
    m = L('tail')
    t = an['tail_t']
    tt = an['tail_tan']
    comb[m] = tangent(tt[m] * 1.0 + nrm[m] * 0.15, nrm[m])
    length[m] = 0.036 + 0.036 * np.sin(np.pi * np.clip(t[m] * 1.1, 0, 1)) ** 0.7
    lift[m] = 0.45
    clump[m] = 0.65
    # angle around the tail axis for the zig-zag
    seg_c = co - D.TAIL_PTS[3]
    ang = np.arctan2(seg_c[:, 2], seg_c[:, 0] + seg_c[:, 1] * 0.5)
    edge = 0.60 + 0.045 * np.sign(np.sin(ang * 4.0)) * np.abs(np.sin(ang * 4.0)) ** 0.5
    tip = sstep(edge - 0.015, edge + 0.015, t) * m
    tipcol = CYAN_DEEP + (CYAN - CYAN_DEEP) * sstep(0.62, 0.95, t)[:, None]
    col = col + (tipcol - col) * tip[:, None]
    base_blue = (1 - tip) * m * 0.25 * sstep(0.2, 0.55, t)
    col = col + (BLUE_TINT - col) * base_blue[:, None]

    # ---- ears
    for s in (1, -1):
        em = is_ear & (ear_side == s)
        if not em.any():
            continue
        u, X, Y = ear_par[em, 0], ear_par[em, 1], ear_par[em, 2]
        base, A, S, Nf = D.ear_frame(s)
        rim = np.abs(X) > 0.8
        inner = (Y > 0.15) & ~rim
        outer = ~inner & ~rim
        c = np.tile(WHITE, (em.sum(), 1))
        c[outer] = WHITE + (BLUE_TINT - WHITE) * (0.35 + 0.35 * u[outer])[:, None]
        cen = sstep(0.95, 0.2, np.abs(X)) * sstep(0.05, 0.3, u) * (1 - sstep(0.75, 0.95, u))
        ic = EAR_INNER + (EAR_INNER_LIGHT - EAR_INNER) * (0.6 * cen)[:, None]
        c[inner] = ic[inner]
        # cyan tuft spot low in the inner ear
        spot = np.exp(-(((u - 0.2) / 0.07) ** 2 + ((X + s * 0.0 - 0.15 * s) / 0.25) ** 2))
        c[inner] = c[inner] + (CYAN - c[inner]) * (0.85 * spot[inner])[:, None]
        col[em] = c
        lng = np.full(em.sum(), 0.008)
        lft = np.full(em.sum(), 0.15)
        cmb = tangent(np.tile(A, (em.sum(), 1)), nrm[em])
        dn_ = np.full(em.sum(), 0.9)
        lng[rim] = 0.013
        lft[rim] = 0.35
        cmb[rim] = tangent(A * 0.6 + S * np.sign(X[rim])[:, None] * 0.6, nrm[em][rim])
        # inner ear: soft light tufts near the base, fine fuzz elsewhere
        tuft = inner & (u < 0.42)
        lng[inner] = 0.004
        dn_[inner] = 0.35
        lng[tuft] = 0.018 * (1 - u[tuft] / 0.42) + 0.005
        dn_[tuft] = 0.45
        lft[tuft] = 0.5
        cmb[tuft] = tangent(np.tile(A + Nf * 0.6, (tuft.sum(), 1)), nrm[em][tuft])
        # roots buried inside the head are dropped
        buried = D.body_sdf(co[em]) < -0.0015
        dn_[buried] = 0
        length[em] = lng
        lift[em] = lft
        comb[em] = cmb
        dens[em] = dn_
        clump[em] = 0.35
    # body roots that sit inside an ear shell
    dens[(an['ear_d'].min(1) < 0.0) & ~is_ear] = 0
    return dict(color=col, length=length, lift=lift, comb=comb, clump=clump, density=dens)
