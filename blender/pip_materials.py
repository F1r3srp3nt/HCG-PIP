"""Cycles materials for Pip."""
import bpy
from pip_colors import hexlin


def _new(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    out.location = (600, 0)
    return m, nt, out


def rgba(h):
    c = hexlin(h)
    return (c[0], c[1], c[2], 1.0)


def principled(nt, out, color=None, rough=0.5, **kw):
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    b.location = (250, 0)
    if color is not None:
        b.inputs['Base Color'].default_value = rgba(color) if isinstance(color, str) else color
    b.inputs['Roughness'].default_value = rough
    for k, v in kw.items():
        b.inputs[k].default_value = v
    nt.links.new(b.outputs[0], out.inputs['Surface'])
    return b


def make_materials():
    M = {}
    # --- skin under the fur, coloured by the same map as the fur
    m, nt, out = _new('Pip_Skin')
    b = principled(nt, out, rough=0.9, **{'Sheen Weight': 0.25, 'Sheen Roughness': 0.6,
                                         'Specular IOR Level': 0.15})
    a = nt.nodes.new('ShaderNodeAttribute')
    a.attribute_name = 'pip_color'
    a.location = (-200, 0)
    hsv = nt.nodes.new('ShaderNodeHueSaturation')
    hsv.inputs['Value'].default_value = 0.92
    nt.links.new(a.outputs['Color'], hsv.inputs['Color'])
    nt.links.new(hsv.outputs['Color'], b.inputs['Base Color'])
    M['skin'] = m

    # --- fur strands
    m, nt, out = _new('Pip_Fur')
    h = nt.nodes.new('ShaderNodeBsdfHairPrincipled')
    h.model = 'CHIANG'
    h.parametrization = 'COLOR'
    h.inputs['Roughness'].default_value = 0.42
    h.inputs['Radial Roughness'].default_value = 0.75
    h.inputs['Coat'].default_value = 0.25
    a = nt.nodes.new('ShaderNodeAttribute')
    a.attribute_name = 'fur_color'
    a.location = (-500, 0)
    info = nt.nodes.new('ShaderNodeHairInfo')
    info.location = (-500, -250)
    # roots a touch deeper, tips lighter for a soft plush gradient
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.location = (-300, -250)
    ramp.color_ramp.elements[0].color = (0.72, 0.74, 0.80, 1)
    ramp.color_ramp.elements[1].position = 0.55
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    nt.links.new(info.outputs['Intercept'], ramp.inputs['Fac'])
    mul = nt.nodes.new('ShaderNodeMix')
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.inputs['Factor'].default_value = 1.0
    mul.location = (-100, 0)
    nt.links.new(a.outputs['Color'], mul.inputs[6])
    nt.links.new(ramp.outputs['Color'], mul.inputs[7])
    # hair colour parametrisation darkens; lift it back towards the map
    gam = nt.nodes.new('ShaderNodeGamma')
    gam.inputs['Gamma'].default_value = 0.85
    nt.links.new(mul.outputs[2], gam.inputs['Color'])
    nt.links.new(gam.outputs['Color'], h.inputs['Color'])
    # mix in a little diffuse for a softer, plush read
    dif = nt.nodes.new('ShaderNodeBsdfDiffuse')
    nt.links.new(gam.outputs['Color'], dif.inputs['Color'])
    mix = nt.nodes.new('ShaderNodeMixShader')
    mix.inputs['Fac'].default_value = 0.35
    nt.links.new(h.outputs[0], mix.inputs[1])
    nt.links.new(dif.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
    M['fur'] = m

    # --- eyes: iris painted from the eye's local +Z (gaze) axis
    m, nt, out = _new('Pip_Eye')
    b = principled(nt, out, rough=0.3, **{'Coat Weight': 1.0, 'Coat Roughness': 0.03,
                                         'Coat IOR': 1.45, 'Specular IOR Level': 0.6})
    tc = nt.nodes.new('ShaderNodeTexCoord')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    norm = nt.nodes.new('ShaderNodeVectorMath')
    norm.operation = 'NORMALIZE'
    nt.links.new(tc.outputs['Object'], norm.inputs[0])
    nt.links.new(norm.outputs['Vector'], sep.inputs[0])
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    cr = ramp.color_ramp
    cr.interpolation = 'EASE'
    stops = [(0.0, '#EEF0F5'), (0.42, '#EEF0F5'), (0.45, '#1E0D04'), (0.53, '#6A2C08'),
             (0.64, '#C0600E'), (0.74, '#E8921F'), (0.82, '#F6B843'), (0.84, '#3B1A06'),
             (0.855, '#050304'), (1.0, '#050304')]
    cr.elements[0].position, cr.elements[0].color = stops[0][0], rgba(stops[0][1])
    cr.elements[1].position, cr.elements[1].color = stops[-1][0], rgba(stops[-1][1])
    for pos, col in stops[1:-1]:
        e = cr.elements.new(pos)
        e.color = rgba(col)
    nt.links.new(sep.outputs['Z'], ramp.inputs['Fac'])
    # lighter amber toward the bottom of the iris, darker under the lid
    grad = nt.nodes.new('ShaderNodeMapRange')
    grad.inputs['From Min'].default_value = 0.35
    grad.inputs['From Max'].default_value = -0.45
    grad.inputs['To Min'].default_value = 0.55
    grad.inputs['To Max'].default_value = 1.35
    nt.links.new(sep.outputs['Y'], grad.inputs['Value'])
    mul = nt.nodes.new('ShaderNodeMix')
    mul.data_type = 'RGBA'
    mul.blend_type = 'MULTIPLY'
    mul.inputs['Factor'].default_value = 1.0
    nt.links.new(ramp.outputs['Color'], mul.inputs[6])
    comb = nt.nodes.new('ShaderNodeCombineColor')
    nt.links.new(grad.outputs['Result'], comb.inputs[0])
    nt.links.new(grad.outputs['Result'], comb.inputs[1])
    nt.links.new(grad.outputs['Result'], comb.inputs[2])
    nt.links.new(comb.outputs[0], mul.inputs[7])
    nt.links.new(mul.outputs[2], b.inputs['Base Color'])
    M['eye'] = m

    # --- eyelids (fur coloured, dark lash line along the cut edge)
    m, nt, out = _new('Pip_Lid')
    b = principled(nt, out, rough=0.65, **{'Sheen Weight': 0.6})
    a = nt.nodes.new('ShaderNodeAttribute')
    a.attribute_name = 'lash'
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = rgba('#E9ECF6')
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = rgba('#140F1C')
    nt.links.new(a.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    M['lid'] = m

    m, nt, out = _new('Pip_Lash')
    principled(nt, out, '#120D18', 0.4)
    M['lash'] = m
    m, nt, out = _new('Pip_Nose')
    principled(nt, out, '#16141E', 0.25, **{'Coat Weight': 0.7, 'Coat Roughness': 0.08})
    M['nose'] = m
    m, nt, out = _new('Pip_Tongue')
    principled(nt, out, '#E27587', 0.32, **{'Subsurface Weight': 0.2, 'Coat Weight': 0.3})
    M['tongue'] = m

    # --- scarf: soft navy fleece with gentle colour variation
    m, nt, out = _new('Pip_Scarf')
    b = principled(nt, out, rough=0.85, **{'Sheen Weight': 0.5, 'Sheen Roughness': 0.35,
                                          'Sheen Tint': (0.35, 0.42, 0.9, 1)})
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 60
    noise.inputs['Detail'].default_value = 6
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = rgba('#131B58')
    ramp.color_ramp.elements[1].color = rgba('#26389A')
    nt.links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    bump = nt.nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.25
    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    M['scarf'] = m

    # --- gold die
    m, nt, out = _new('Gold_Die')
    b = principled(nt, out, '#F2B23A', 0.3, **{'Metallic': 0.55, 'Coat Weight': 0.45,
                                              'Coat Roughness': 0.12})
    noise = nt.nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = 140
    mr = nt.nodes.new('ShaderNodeMapRange')
    mr.inputs['To Min'].default_value = 0.24
    mr.inputs['To Max'].default_value = 0.38
    nt.links.new(noise.outputs['Fac'], mr.inputs['Value'])
    nt.links.new(mr.outputs['Result'], b.inputs['Roughness'])
    M['gold'] = m
    m, nt, out = _new('Die_Pips')
    principled(nt, out, '#2E1A0B', 0.55)
    M['pip'] = m

    # --- eye catch-light card (only seen in reflections)
    m, nt, out = _new('Catchlight')
    e = nt.nodes.new('ShaderNodeEmission')
    e.inputs['Strength'].default_value = 25.0
    nt.links.new(e.outputs[0], out.inputs['Surface'])
    M['catch'] = m

    # --- studio backdrop
    m, nt, out = _new('Backdrop')
    b = principled(nt, out, rough=0.95)
    tc = nt.nodes.new('ShaderNodeTexCoord')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(tc.outputs['Generated'], sep.inputs[0])
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = rgba('#213C55')
    ramp.color_ramp.elements[1].color = rgba('#0E1A28')
    nt.links.new(sep.outputs['Y'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    M['backdrop'] = m
    return M
