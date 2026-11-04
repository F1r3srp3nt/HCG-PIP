"""Build Pip the arctic fox from scratch and save blender/pip.blend.

Run with Blender's Python (bpy 5.2+), plus numpy / scipy / scikit-image:
    python build_pip.py [--strands 180000] [--out pip.blend]
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import bpy
from mathutils import Vector, Matrix

import pip_design as D
import pip_mesh as PM
import pip_colors as PC
import pip_parts as PP
import pip_materials as PMAT
import pip_rig as PR
import pip_fur as PF


def log(*a):
    print('[pip]', *a, flush=True)


def coll(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


def smart_uv(ob):
    bpy.context.view_layer.objects.active = ob
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.003)
    bpy.ops.object.mode_set(mode='OBJECT')


def build_character(n_strands):
    scene = bpy.context.scene
    C_pip = coll('Pip')
    C_geo = coll('Pip_Geometry', C_pip)
    C_rig = coll('Pip_Rig', C_pip)
    C_wgt = coll('Widgets', C_pip)
    C_wgt.hide_render = True
    C_wgt.hide_viewport = True
    C_prop = coll('Props')
    mats = PMAT.make_materials()

    t = time.time()
    me = PM.build_body_mesh()
    body = bpy.data.objects.new('Pip_Body', me)
    C_geo.objects.link(body)
    me.materials.append(mats['skin'])
    n_all = len(me.vertices)
    n_ear = sum(len(D.ear_mesh(s, nu=96, nv=64)[0]) for s in (1, -1))
    n_body = n_all - n_ear
    log('body mesh', n_all, 'verts', f'{time.time() - t:.1f}s')

    co = np.empty(n_all * 3)
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    nrm = np.empty(n_all * 3)
    me.vertices.foreach_get('normal', nrm)
    nrm = nrm.reshape(-1, 3)
    edges = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get('vertices', edges)
    edges = edges.reshape(-1, 2)

    an = PM.vertex_analysis(co)
    lab, is_ear = PC.classify(an, n_body)
    # ear params straight from the parametric shells
    ear_side = np.zeros(n_all, int)
    ear_par = np.zeros((n_all, 3))
    off = n_body
    for s in (1, -1):
        ev, ef, par = D.ear_mesh(s, nu=96, nv=64)
        ear_side[off:off + len(ev)] = s
        ear_par[off:off + len(ev)] = par
        off += len(ev)
    ear_par[:, 0] = np.clip(ear_par[:, 0], 0, 1)
    rng = np.random.default_rng(3)
    F = PC.fields(co, nrm, an, lab, is_ear, ear_side, ear_par, rng)

    # scarf first: fur under it is removed
    scarf, scarf_parts = PP.make_scarf(mats, C_geo)
    sv = np.empty(len(scarf.data.vertices) * 3)
    scarf.data.vertices.foreach_get('co', sv)
    from scipy.spatial import cKDTree
    dsc, _ = cKDTree(sv.reshape(-1, 3)).query(co)
    F['density'][dsc < 0.010] = 0
    F['length'] = np.where(dsc < 0.03, np.minimum(F['length'], 0.012 + 0.6 * np.maximum(dsc - 0.01, 0)),
                           F['length'])

    ca = me.color_attributes.new('pip_color', 'FLOAT_COLOR', 'POINT')
    ca.data.foreach_set('color', np.c_[F['color'], np.ones(n_all)].reshape(-1).astype(np.float32))
    me.color_attributes.active_color = ca
    smart_uv(body)

    # ---- face parts, die
    face = {}
    for s in (1, -1):
        sx = 'L' if s > 0 else 'R'
        face[f'eye.{sx}'] = PP.make_eye(s, mats, C_geo)
        lids = PP.make_lids(s, mats, C_geo)
        face[f'lid_up.{sx}'] = lids['up']
        face[f'lid_low.{sx}'] = lids['low']
        face[f'lash.{sx}'] = lids['lash']
    face['nose'] = PP.make_nose(mats, C_geo)
    face['tongue'] = PP.make_tongue(mats, C_geo)
    die = PP.make_die(mats, C_prop)

    # ---- rig
    rig = PR.build_rig(C_rig, C_wgt)
    bones, Wt = PR.compute_weights(co, an, is_ear, ear_side, ear_par, edges, n_body)
    PR.assign_weights(body, bones, Wt)
    body.parent = rig
    body.add_rest_position_attribute = True   # needed by 'Deform Curves on Surface'
    am = body.modifiers.new('Armature', 'ARMATURE')
    am.object = rig
    log('weights done')
    # scarf weights
    sco = sv.reshape(-1, 3)
    names = list(scarf['part_names'])
    groups = {}

    def g(name):
        if name not in groups:
            groups[name] = scarf.vertex_groups.new(name=name)
        return groups[name]
    for i, (p, pid) in enumerate(zip(sco, scarf_parts)):
        part = names[pid]
        if part in ('ring', 'knot'):
            wn = np.clip((p[2] - 0.235) / 0.05, 0, 1)
            g('neck').add([i], float(wn), 'REPLACE')
            g('chest').add([i], float(1 - wn), 'REPLACE')
        else:
            path = PP.SCARF_TAIL_A if part == 'tail_a' else PP.SCARF_TAIL_B
            tag = part[-1]
            d0 = np.linalg.norm(p - path[0])
            dl = np.linalg.norm(path[-1] - path[0])
            k = np.clip(d0 / dl, 0, 1)
            w2 = np.clip((k - 0.45) / 0.3, 0, 1)
            w1 = (1 - w2) * np.clip(k / 0.15, 0, 1)
            g(f'scarf_{tag}.02').add([i], float(w2), 'REPLACE')
            g(f'scarf_{tag}.01').add([i], float(w1), 'REPLACE')
            g('chest').add([i], float(1 - w1 - w2), 'REPLACE')
    scarf.parent = rig
    sm = scarf.modifiers.new('Armature', 'ARMATURE')
    sm.object = rig
    bpy.context.view_layer.objects.active = scarf
    bpy.ops.object.modifier_move_to_index(modifier='Armature', index=0)

    for s, sx in ((1, 'L'), (-1, 'R')):
        PR.bone_parent(face[f'eye.{sx}'], rig, f'eye.{sx}')
        PR.bone_parent(face[f'lid_up.{sx}'], rig, f'lid_up.{sx}')
        PR.bone_parent(face[f'lid_low.{sx}'], rig, f'lid_low.{sx}')
        PR.bone_parent(face[f'lash.{sx}'], rig, f'lid_up.{sx}')
    PR.bone_parent(face['nose'], rig, 'head')
    PR.bone_parent(face['tongue'], rig, 'jaw')

    # catch-light card: only visible in reflections (eyes / nose)
    bpy.ops.mesh.primitive_circle_add(vertices=24, radius=0.045, fill_type='NGON',
                                      location=(-0.10, -0.42, 0.56))
    cl = bpy.context.active_object
    cl.name = 'Eye_Catchlight'
    for c in cl.users_collection:
        c.objects.unlink(cl)
    C_geo.objects.link(cl)
    cl.rotation_euler = (math.radians(70), 0, math.radians(-12))
    cl.data.materials.append(mats['catch'])
    cl.visible_camera = False
    cl.visible_diffuse = False
    cl.visible_shadow = False
    cl.visible_transmission = False
    cl.visible_volume_scatter = False
    PR.bone_parent(cl, rig, 'head')

    # die is held through a constraint so it can be released and caught
    ct = die.constraints.new('COPY_TRANSFORMS')
    ct.name = 'Held by Pip'
    ct.target = rig
    ct.subtarget = 'die_socket'
    die['note'] = "Animate 'Held by Pip' influence: 1 = in Pip's paws, 0 = free (uses the die's own keys)."

    # ---- fur
    t = time.time()
    vattr = dict(co=co, nrm=nrm, color=F['color'], length=F['length'], lift=F['lift'],
                 comb=F['comb'], clump=F['clump'], density=F['density'])
    fur = PF.build_fur(body, C_geo, mats['fur'], vattr, n_strands)
    log('fur', n_strands, f'{time.time() - t:.1f}s')
    return dict(rig=rig, body=body, fur=fur, die=die, scarf=scarf, face=face, mats=mats)


def build_stage(mats):
    scene = bpy.context.scene
    C = coll('Stage')
    # cyclorama backdrop facing the hero camera
    verts, faces = [], []
    prof = []
    for y in np.linspace(-3.0, 0.9, 12):
        prof.append((y, 0.0))
    for a in np.linspace(0, math.pi / 2, 14)[1:]:
        prof.append((0.9 + math.sin(a) * 1.2, 1.2 - math.cos(a) * 1.2))
    for z in np.linspace(1.4, 4.0, 5):
        prof.append((2.1, z))
    xs = np.linspace(-4, 4, 3)
    for (y, z) in prof:
        for x in xs:
            verts.append((x, y, z))
    nx = len(xs)
    for i in range(len(prof) - 1):
        for j in range(nx - 1):
            a = i * nx + j
            faces.append((a, a + 1, a + nx + 1, a + nx))
    me = bpy.data.meshes.new('Backdrop')
    me.from_pydata(verts, [], faces)
    for p in me.polygons:
        p.use_smooth = True
    me.materials.append(mats['backdrop'])
    bd = bpy.data.objects.new('Backdrop', me)
    C.objects.link(bd)
    bd.rotation_euler = (0, 0, math.radians(-30))
    bd.location = (0.3, 0.4, 0)

    def area(name, loc, target, size, power, color):
        L = bpy.data.lights.new(name, 'AREA')
        L.shape = 'DISK'
        L.size = size
        L.energy = power
        L.color = color
        o = bpy.data.objects.new(name, L)
        C.objects.link(o)
        o.location = loc
        d = Vector(target) - Vector(loc)
        o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return o
    area('Key', (-0.55, -1.45, 1.35), (0, -0.05, 0.32), 1.1, 95, (1.0, 0.96, 0.90))
    area('Fill', (-1.4, -0.9, 0.55), (0, -0.05, 0.3), 1.4, 28, (0.80, 0.88, 1.0))
    area('Rim_L', (1.05, 1.10, 1.05), (0, 0, 0.35), 0.7, 85, (0.85, 0.95, 1.0))
    area('Rim_R', (-1.0, 0.9, 0.9), (0, 0, 0.35), 0.7, 70, (0.65, 0.92, 1.0))
    area('Top', (0.0, 0.2, 1.8), (0, 0, 0.3), 0.8, 30, (1.0, 1.0, 1.0))

    w = bpy.data.worlds.new('Pip_World')
    scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.010, 0.020, 0.035, 1)
    bg.inputs['Strength'].default_value = 1.0

    # hero three-quarter camera (fixed) and a front turntable camera
    def cam(name, loc, target, lens):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.sensor_width = 36
        o = bpy.data.objects.new(name, cd)
        C.objects.link(o)
        o.location = loc
        d = Vector(target) - Vector(loc)
        o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return o
    cam3q = cam('Cam_ThreeQuarter', (-1.05, -1.52, 0.56), (-0.07, -0.02, 0.345), 58)
    camtt = cam('Cam_Turntable', (0.0, -1.85, 0.40), (-0.02, 0.0, 0.31), 68)
    scene.camera = cam3q
    return dict(cam3q=cam3q, camtt=camtt)


def render_settings():
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    s.cycles.device = 'CPU'
    s.cycles.samples = 64
    s.cycles.use_denoising = True
    s.cycles.max_bounces = 6
    s.cycles.diffuse_bounces = 2
    s.cycles.glossy_bounces = 2
    s.cycles.transmission_bounces = 4
    s.cycles.transparent_max_bounces = 8
    s.cycles.use_adaptive_sampling = True
    s.cycles.adaptive_threshold = 0.03
    s.render.resolution_x = 960
    s.render.resolution_y = 960
    s.render.fps = 24
    s.frame_start = 1
    s.frame_end = 120
    s.render.film_transparent = False
    s.view_settings.view_transform = 'AgX'
    s.view_settings.look = 'AgX - Punchy'
    s.render.hair_type = 'STRAND' if hasattr(s.render, 'hair_type') else None
    s.cycles_curves.shape = 'THICK' if hasattr(s, 'cycles_curves') else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strands', type=int, default=180000)
    ap.add_argument('--out', default=os.path.join(HERE, 'pip.blend'))
    ap.add_argument('--no-anim', action='store_true')
    args = ap.parse_args([a for a in sys.argv[1:] if a.startswith('--') or not a.endswith('.py')])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    T0 = time.time()
    ch = build_character(args.strands)
    st = build_stage(ch['mats'])
    render_settings()
    if not args.no_anim:
        import pip_anim
        pip_anim.animate(ch, st)
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=args.out, compress=True)
    log('saved', args.out, f'{time.time() - T0:.1f}s total')


if __name__ == '__main__':
    main()
