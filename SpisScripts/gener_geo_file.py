import math


def write_header(mesh_scale: float = 1):

    cube_text = f'''

scale = {mesh_scale};

resAN = scale;
resSC = scale*4;
resBC = scale*20;


Point(0) = {{0, 0, 0, resSC}};'''

    return cube_text


def write_cube_surface(size: tuple = (5, 5, 5), corner: tuple = (5, 5, 5), prefix: int = 1,
                       name="spacecraft", var_name="SL_SC_ANT", start_point_id: int = 30):

    num_prefix = str(prefix)
    dx, dy, dz = size
    cx, cy, cz = corner

    # Helper to format (possibly negative) IDs with prefix:  -44 -> "-144", 45 -> "145"
    def id_str(n: int) -> str:
        sign = "-" if n < 0 else ""
        return f"{sign}{num_prefix}{abs(n)}"

    # cube_points = []
    # cube_points.append((corner[0] + dx, corner[1] + dy, corner[2] + dz))
    # cube_points.append((corner[0], corner[1] + dy, corner[2] + dz))
    # cube_points.append((corner[0], corner[1] , corner[2] + dz))
    # cube_points.append((corner[0] + dx, corner[1], corner[2] + dz))
    # cube_points.append((corner[0] + dx, corner[1] + dy, corner[2]))
    # cube_points.append((corner[0], corner[1] + dy, corner[2]))
    # cube_points.append((corner[0], corner[1], corner[2]))
    # cube_points.append((corner[0] + dx, corner[1], corner[2]))

    cube_points = [
        (cx + dx, cy + dy, cz + dz),  # 30
        (cx,      cy + dy, cz + dz),  # 31
        (cx,      cy,      cz + dz),  # 32
        (cx + dx, cy,      cz + dz),  # 33
        (cx + dx, cy + dy, cz),       # 34
        (cx,      cy + dy, cz),       # 35
        (cx,      cy,      cz),       # 36
        (cx + dx, cy,      cz),       # 37
    ]

    # Generate Point() definitions
    point_lines = []
    for idx, (x, y, z) in enumerate(cube_points, start=start_point_id):
        point_lines.append(
            f"Point({num_prefix}{idx}) = {{{x},{y},{z} , resSC}};"
        )
    points_str = "\n".join(point_lines)

    # Line definitions: (line_id, from_point, to_point) using local IDs 30–37
    lines = [
        (43, 30, 31),
        (44, 31, 32),
        (45, 32, 33),
        (46, 33, 30),
        (47, 34, 35),
        (48, 35, 36),
        (49, 36, 37),
        (50, 37, 34),
        (51, 34, 30),
        (52, 35, 31),
        (53, 36, 32),
        (54, 37, 33),
    ]
    line_lines = [
        f"Line({num_prefix}{lid}) = {{{num_prefix}{p1}, {num_prefix}{p2}}};"
        for (lid, p1, p2) in lines
    ]
    lines_str = "\n".join(line_lines)

    # Curve loops: surface_id -> list of line IDs (with sign for orientation)
    curve_loops = {
        33: [48, 53, -44, -52],
        34: [52, -43, -51, 47],
        35: [51, -46, -54, 50],
        36: [49, 54, -45, -53],
        37: [44, 45, 46, 43],
        38: [49, 50, 47, 48],
    }

    curve_loop_lines = []
    surface_lines = []

    # For surfaces: all positive except 38, which uses reversed sign inside
    surface_map = {
        33: 33,
        34: 34,
        35: 35,
        36: 36,
        37: 37,
        38: -38,  # same as original: Surface(138) = {-138}
    }

    for sid, loop_lines in curve_loops.items():
        loop_str = ", ".join(id_str(l) for l in loop_lines)
        curve_loop_lines.append(
            f"Curve Loop({num_prefix}{sid}) = {{{loop_str}}};"
        )
        surface_lines.append(
            f"Surface({num_prefix}{sid}) = {{{id_str(surface_map[sid])}}};"
        )

    curve_loops_str = "\n".join(curve_loop_lines)
    surfaces_str = "\n".join(surface_lines)

    surf_ids = ",".join(f"{num_prefix}{sid}" for sid in range(33, 39))

    cube_text = f'''
    
    
/* cube {name} points */
{points_str}

/* SC lines*/
{lines_str}


/* SC surface */
{curve_loops_str}
{surfaces_str}

{var_name} = {num_prefix};
Surface Loop({var_name}) = {{{surf_ids}}};
Physical Surface("{name}", {var_name}) = {{{surf_ids}}};
'''

# Curve Loop({num_prefix}33) = {{{num_prefix}48, {num_prefix}53, -{num_prefix}44, -{num_prefix}52}};
# Surface({num_prefix}33) = {{{num_prefix}33}};
# Curve Loop({num_prefix}34) = {{{num_prefix}52, -{num_prefix}43, -{num_prefix}51, {num_prefix}47}};
# Surface({num_prefix}34) = {{{num_prefix}34}};
# Curve Loop({num_prefix}35) = {{{num_prefix}51, -{num_prefix}46, -{num_prefix}54, {num_prefix}50}};
# Surface({num_prefix}35) = {{{num_prefix}35}};
# Curve Loop({num_prefix}36) = {{{num_prefix}49, {num_prefix}54, -{num_prefix}45, -{num_prefix}53}};
# Surface({num_prefix}36) = {{{num_prefix}36}};
# Curve Loop({num_prefix}37) = {{{num_prefix}44, {num_prefix}45, {num_prefix}46, {num_prefix}43}};
# Surface({num_prefix}37) = {{{num_prefix}37}};
# Curve Loop({num_prefix}38) = {{{num_prefix}49, {num_prefix}50, {num_prefix}47, {num_prefix}48}};
# Surface({num_prefix}38) = {{-{num_prefix}38}};
#
# {var_name} = {num_prefix};
# Surface Loop({var_name}) = {{{num_prefix}33,{num_prefix}34,{num_prefix}35,{num_prefix}36,{num_prefix}37,{num_prefix}38}};
# Physical Surface("{name}", {var_name}) = {{{num_prefix}33,{num_prefix}34,{num_prefix}35,{num_prefix}36,{num_prefix}37,{num_prefix}38}};

    return cube_text


def write_sphere_surface(r: float, middle: tuple = (5, 5, 5), prefix: int = 1, name="spacecraft", var_name="SL_SC_ANT"):
    num_prefix = str(prefix)
    radius = r
    cx, cy, cz = middle

    # --- helpers ---------------------------------------------------------
    # Point IDs are in a 100-block: prefix + 2-digit local id (e.g. 1 + 00 -> 100)
    def point_id(local_id: int) -> str:
        return f"{num_prefix}{local_id:02d}"

    # For curve/loop/surface IDs (no 100-block), but with sign.
    def id_str(n: int) -> str:
        sign = "-" if n < 0 else ""
        return f"{sign}{num_prefix}{abs(n)}"

    # --- points ----------------------------------------------------------
    # Local IDs: 90 = center, 00/10/11/12/13/14 = axes points
    point_defs = [
        (90, (cx, cy, cz)),  # center
        (0, (cx + radius, cy, cz)),  # +X
        (10, (cx - radius, cy, cz)),  # -X
        (11, (cx, cy + radius, cz)),  # +Y
        (12, (cx, cy - radius, cz)),  # -Y
        (13, (cx, cy, cz + radius)),  # +Z
        (14, (cx, cy, cz - radius)),  # -Z
    ]

    point_lines = [
        f"Point({point_id(pid)}) = {{{x}, {y}, {z}, resBC}};"
        for pid, (x, y, z) in point_defs
    ]
    points_str = "\n".join(point_lines)

    # --- ellipses --------------------------------------------------------
    # Ellipse(id) = {start, center, third, end}  -- all are point IDs (local)
    ellipse_point_map = {
        27: (10, 90, 0, 13),
        28: (0, 90, 10, 13),
        29: (11, 90, 12, 13),
        30: (12, 90, 11, 13),
        31: (14, 90, 13, 0),
        32: (14, 90, 13, 10),
        33: (10, 90, 0, 12),
        34: (12, 90, 11, 0),
        35: (0, 90, 10, 11),
        36: (10, 90, 0, 11),
        37: (14, 90, 13, 11),
        38: (12, 90, 14, 14),
    }

    ellipse_lines = []
    for eid, (p1, p2, p3, p4) in ellipse_point_map.items():
        ellipse_lines.append(
            "Ellipse(" + id_str(eid).lstrip("-") + ") = "
                                                   "{"
                                                   f"{point_id(p1)}, {point_id(p2)}, {point_id(p3)}, {point_id(p4)}"
                                                   "};"
        )
    ellipses_str = "\n".join(ellipse_lines)

    # --- line loops & surfaces -------------------------------------------
    # Loops are built from ellipse IDs (with orientation via sign)
    loop_map = {
        40: [30, -28, -34],
        41: [28, -29, -35],
        43: [29, -27, 36],
        45: [33, 30, -27],
        47: [38, 31, -34],
        49: [31, 35, -37],
        51: [37, -36, -32],
        53: [32, 33, 38],
    }

    # Surface(loop) relationships and orientation sign (as in original code)
    surface_map = {
        40: (40, +1),
        41: (42, +1),
        43: (44, +1),
        45: (46, -1),
        47: (48, -1),
        49: (50, +1),
        51: (52, +1),
        53: (54, +1),
    }

    loop_lines = []
    surface_lines = []

    for loop_id, ellipse_ids in loop_map.items():
        loop_content = ", ".join(id_str(eid) for eid in ellipse_ids)
        loop_lines.append(
            f"Line Loop({id_str(loop_id).lstrip('-')}) = {{{loop_content}}};"
        )

        surf_id, sign = surface_map[loop_id]
        loop_ref = id_str(loop_id) if sign > 0 else id_str(-loop_id)
        surface_lines.append(
            f"Surface({id_str(surf_id).lstrip('-')}) = {{{loop_ref}}};"
        )

    loops_str = "\n".join(loop_lines)
    surfaces_str = "\n".join(surface_lines)

    # Surfaces included in the final Surface Loop / Physical Surface
    surf_ids = [40, 42, 44, 46, 48, 50, 52, 54]
    surf_ids_str = ", ".join(id_str(sid) for sid in surf_ids)

    cube_text = f'''
    
/* sphere {name} points */
{points_str}

/* SPHERE curves */
{ellipses_str}

/* SPHERE surface */
{loops_str}
{surfaces_str}

{var_name} = {num_prefix};
Surface Loop({var_name}) = {{{surf_ids_str}}};
Physical Surface("{name}", {var_name}) = {{{surf_ids_str}}};'''


# /* sphere {name} points */
# Point(390) = {{0, 0, 0, resBC}};
#
# Point(300) = {{{radius}, 0, 0, resBC}};
# Point(310) = {{-{radius}, 0, 0, resBC}};
# Point(311) = {{0, {radius}, 0, resBC}};
# Point(312) = {{0, -{radius}, 0, resBC}};
# Point(313) = {{0, 0, {radius}, resBC}};
# Point(314) = {{0, 0, -{radius}, resBC}};
#
# /* SPHERE curves */
# Ellipse({num_prefix}27) = {{{num_prefix}10, {num_prefix}90, {num_prefix}00, {num_prefix}13}};
# Ellipse({num_prefix}28) = {{{num_prefix}00, {num_prefix}90, {num_prefix}10, {num_prefix}13}};
# Ellipse({num_prefix}29) = {{{num_prefix}11, {num_prefix}90, {num_prefix}12, {num_prefix}13}};
# Ellipse({num_prefix}30) = {{{num_prefix}12, {num_prefix}90, {num_prefix}11, {num_prefix}13}};
# Ellipse({num_prefix}31) = {{{num_prefix}14, {num_prefix}90, {num_prefix}13, {num_prefix}00}};
# Ellipse({num_prefix}32) = {{{num_prefix}14, {num_prefix}90, {num_prefix}13, {num_prefix}10}};
# Ellipse({num_prefix}33) = {{{num_prefix}10, {num_prefix}90, {num_prefix}00, {num_prefix}12}};
# Ellipse({num_prefix}34) = {{{num_prefix}12, {num_prefix}90, {num_prefix}11, {num_prefix}00}};
# Ellipse({num_prefix}35) = {{{num_prefix}00, {num_prefix}90, {num_prefix}10, {num_prefix}11}};
# Ellipse({num_prefix}36) = {{{num_prefix}10, {num_prefix}90, {num_prefix}00, {num_prefix}11}};
# Ellipse({num_prefix}37) = {{{num_prefix}14, {num_prefix}90, {num_prefix}13, {num_prefix}11}};
# Ellipse({num_prefix}38) = {{{num_prefix}12, {num_prefix}90, {num_prefix}14, {num_prefix}14}};
#
# /* SPHERE surface */
# Line Loop({num_prefix}40) = {{{num_prefix}30, -{num_prefix}28, -{num_prefix}34}};
# Surface({num_prefix}40) = {{{num_prefix}40}};
# Line Loop({num_prefix}41) = {{{num_prefix}28, -{num_prefix}29, -{num_prefix}35}};
# Surface({num_prefix}42) = {{{num_prefix}41}};
# Line Loop({num_prefix}43) = {{{num_prefix}29, -{num_prefix}27, {num_prefix}36}};
# Surface({num_prefix}44) = {{{num_prefix}43}};
# Line Loop({num_prefix}45) = {{{num_prefix}33, {num_prefix}30, -{num_prefix}27}};
# Surface({num_prefix}46) = {{-{num_prefix}45}};
# Line Loop({num_prefix}47) = {{{num_prefix}38, {num_prefix}31, -{num_prefix}34}};
# Surface({num_prefix}48) = {{-{num_prefix}47}};
# Line Loop({num_prefix}49) = {{{num_prefix}31, {num_prefix}35, -{num_prefix}37}};
# Surface({num_prefix}50) = {{{num_prefix}49}};
# Line Loop({num_prefix}51) = {{{num_prefix}37, -{num_prefix}36, -{num_prefix}32}};
# Surface({num_prefix}52) = {{{num_prefix}51}};
# Line Loop({num_prefix}53) = {{{num_prefix}32, {num_prefix}33, {num_prefix}38}};
# Surface({num_prefix}54) = {{{num_prefix}53}};
#
# {var_name} = {num_prefix};
# Surface Loop({var_name}) = {{{num_prefix}40, {num_prefix}42, {num_prefix}44, {num_prefix}46, {num_prefix}48, {num_prefix}50, {num_prefix}52, {num_prefix}54}};
# Physical Surface("{name}", {var_name}) = {{{num_prefix}40, {num_prefix}42, {num_prefix}44, {num_prefix}46, {num_prefix}48, {num_prefix}50, {num_prefix}52, {num_prefix}54}};

    return cube_text


def write_cylinder_surface(radius: float, height: float, position: tuple = (0.0, 0.0, 0.0),
                           direction: tuple = (0.0, 0.0, 1.0), prefix: int = 1,
                           name="cylinder", var_name="SL_CYL"):
    """
    Generate a Gmsh .geo snippet for a cylindrical surface, using the same
    'data-table' style as write_sphere_surface.

    - radius: cylinder radius
    - height: cylinder height
    - position: bottom center of the cylinder axis (x, y, z)
    - direction: axis direction vector (dx, dy, dz) – will be normalized
    - prefix: numeric prefix for all entity IDs
    - name: name of the Physical Surface
    - var_name: Gmsh variable used as Surface Loop ID
    """

    num_prefix = str(prefix)
    cx, cy, cz = position
    ax, ay, az = direction

    # ---------- helpers ---------------------------------------------------
    def id_str(n: int) -> str:
        """Return prefixed ID with sign, e.g. -40 -> '-140' for prefix=1."""
        sign = "-" if n < 0 else ""
        return f"{sign}{num_prefix}{abs(n)}"

    def point_id(local: int) -> str:
        """2-digit local ID: 0 -> '100', 90 -> '190' if prefix=1."""
        return f"{num_prefix}{local:02d}"

    def normalize(v):
        x, y, z = v
        norm = math.sqrt(x * x + y * y + z * z)
        if norm == 0.0:
            return (0.0, 0.0, 1.0)
        return (x / norm, y / norm, z / norm)

    def cross(a, b):
        ax, ay, az = a
        bx, by, bz = b
        return (
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx,
        )

    def add(a, b, scale=1.0):
        ax, ay, az = a
        bx, by, bz = b
        return (ax + scale * bx, ay + scale * by, az + scale * bz)

    # ---------- axis + local basis ----------------------------------------
    axis = normalize((ax, ay, az))

    # pick some vector not parallel to axis to build orthonormal basis
    if abs(axis[0]) < 0.9:
        w = (1.0, 0.0, 0.0)
    else:
        w = (0.0, 1.0, 0.0)

    u = normalize(cross(axis, w))  # first perpendicular
    v = normalize(cross(axis, u))  # second perpendicular

    # half_h = height / 2.0
    # center = (cx, cy, cz)
    # c_bot = add(center, axis, scale=-half_h)
    # c_top = add(center, axis, scale=+half_h)

    # position is *bottom* center now
    c_bot = (cx, cy, cz)
    c_top = add(c_bot, axis, scale=height)

    # ---------- points ----------------------------------------------------
    # Use local IDs:
    # 90 = bottom center, 91 = top center
    #  0,10,20,30 = bottom ring (+u,+v,-u,-v)
    #  1,11,21,31 = top ring    (+u,+v,-u,-v)

    def ring_points(center_point):
        return [
            add(center_point, u, scale=+radius),  # +u
            add(center_point, v, scale=+radius),  # +v
            add(center_point, u, scale=-radius),  # -u
            add(center_point, v, scale=-radius),  # -v
        ]

    rb0, rb1, rb2, rb3 = ring_points(c_bot)
    rt0, rt1, rt2, rt3 = ring_points(c_top)

    point_defs = [
        (90, c_bot),
        (91, c_top),
        (0,  rb0),
        (10, rb1),
        (20, rb2),
        (30, rb3),
        (1,  rt0),
        (11, rt1),
        (21, rt2),
        (31, rt3),
    ]

    point_lines = [
        f"Point({point_id(pid)}) = {{{x}, {y}, {z}, resAN}};"
        for pid, (x, y, z) in point_defs
    ]
    points_str = "\n".join(point_lines)

    # ---------- circles (bottom + top) ------------------------------------
    # Circles use: Circle(id) = {start, center, end}
    circle_map = {
        60: (0, 90, 10),  # bottom +u -> +v
        61: (10, 90, 20),
        62: (20, 90, 30),
        63: (30, 90, 0),

        64: (1, 91, 11),  # top +u -> +v
        65: (11, 91, 21),
        66: (21, 91, 31),
        67: (31, 91, 1),
    }

    circle_lines = []
    for cid, (p1, pc, p2) in circle_map.items():
        circle_lines.append(
            f"Circle({id_str(cid).lstrip('-')}) = "
            f"{{{point_id(p1)}, {point_id(pc)}, {point_id(p2)}}};"
        )
    circles_str = "\n".join(circle_lines)

    # ---------- vertical lines --------------------------------------------
    line_map = {
        70: (0, 1),
        71: (10, 11),
        72: (20, 21),
        73: (30, 31),
    }

    line_lines = []
    for lid, (pa, pb) in line_map.items():
        line_lines.append(
            f"Line({id_str(lid).lstrip('-')}) = "
            f"{{{point_id(pa)}, {point_id(pb)}}};"
        )
    lines_str = "\n".join(line_lines)

    # ---------- line loops & surfaces -------------------------------------
    # side patches (4), bottom disk, top disk
    loop_map = {
        40: [60, 71, -64, -70],  # side 1
        41: [61, 72, -65, -71],  # side 2
        42: [62, 73, -66, -72],  # side 3
        43: [63, 70, -67, -73],  # side 4
        44: [60, 61, 62, 63],  # bottom
        45: [64, 65, 66, 67],  # top
    }

    # Surface(local_surf_id) = { ± loop_id }
    # (top disk uses negative loop to flip normal)
    surface_map = {
        40: (50, +1),
        41: (51, +1),
        42: (52, +1),
        43: (53, +1),
        44: (54, +1),
        45: (55, -1),
    }

    loop_lines = []
    surface_lines = []

    for loop_id, edges in loop_map.items():
        loop_content = ", ".join(id_str(e) for e in edges)
        loop_lines.append(
            f"Line Loop({id_str(loop_id).lstrip('-')}) = {{{loop_content}}};"
        )

        surf_id, sign = surface_map[loop_id]
        loop_ref = id_str(loop_id if sign > 0 else -loop_id)
        surface_lines.append(
            f"Surface({id_str(surf_id).lstrip('-')}) = {{{loop_ref}}};"
        )

    loops_str = "\n".join(loop_lines)
    surfaces_str = "\n".join(surface_lines)

    # surfaces in the final Surface Loop / Physical Surface
    surface_ids = [50, 51, 52, 53, 54, 55]
    surface_ids_str = ", ".join(id_str(sid) for sid in surface_ids)

    cylinder_text = f"""
    
    
/* cylinder {name} points */
{points_str}

/* CYLINDER circles */
{circles_str}

/* CYLINDER vertical lines */
{lines_str}

/* CYLINDER surfaces */
{loops_str}
{surfaces_str}

{var_name} = {num_prefix};
Surface Loop({var_name}) = {{{surface_ids_str}}};
Physical Surface("{name}", {var_name}) = {{{surface_ids_str}}};
"""
    return cylinder_text


def write_volume_def(volume_names: tuple, value: int):
    volumes = ""
    for v in volume_names:
        volumes += str(v) + ", "

    volumes = volumes[:-2]
    cube_text = f'''

//=========================
//Volume
//=========================
V_COMP = {value};
Volume(V_COMP) = {{{volumes}}};

Physical Volume("Computational volume", {value}) = {{{value}}};	//Computational volume'''

    return cube_text

if __name__ == "__main__":
    # write_sphere_with_cube_geo()
    geo_text = ""

    geo_text += write_header(0.25)
    geo_text += write_cube_surface((2.5, 3.1, 2.7), (-1.25, -1.55, -1.35), 1, "spacecraft", "SL_SC")
    geo_text += write_cube_surface((3, 3.6, 0.4), (-1.5, -1.8, 1.45), 2, "shield", "SH_SC")
    geo_text += write_cube_surface((6.3, 1.2, 0.1), (-7.95, -0.6, -0.05), 3, "spacecraft_panelL", "SL_SC_PAN_L")
    geo_text += write_cube_surface((6.3, 1.2, 0.1), (1.65, -0.6, -0.05), 4, "spacecraft_panelR", "SL_SC_PAN_R")
    geo_text += write_sphere_surface(25.0, (0, 0, 0), 5, "EXT BOUNDARY", "SL_boundary")
    geo_text += write_cylinder_surface(0.1, 6.5, (0, 1.75, 0), (0, 1, 0), 6, "antenna1", "ANT1")
    geo_text += write_cylinder_surface(0.1, 6.5, (1.4, -1.7, 0), (1, -1/2.0, 0), 7, "antenna2", "ANT2")
    geo_text += write_cylinder_surface(0.1, 6.5, (-1.4, -1.7, 0), (-1, -1 / 2.0, 0), 8, "antenna3", "ANT3")
    # geo_text += write_cylinder_surface(0.1, 0.1, (1.5, 1.5, 2.1), (0, 0, 1), 6, "thrust", "THRST")

    geo_text += write_volume_def(("SL_SC", "SH_SC", "SL_SC_PAN_L", "SL_SC_PAN_R", "ANT1", "ANT2", "ANT3", "SL_boundary"), 1000)

    geo_text += '''

//======================================
// END
//======================================'''

    name = "sat_orbiter.geo"
    with open(name, "w", encoding="utf-8") as f:
        f.write(geo_text)

    print(f"Wrote {name}")
