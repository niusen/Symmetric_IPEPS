"""Spatial sqrt(3) x sqrt(3) kagome PESS, with virtual spaces from TOML.

Edit SETTINGS below and run this file in an editor. No command-line options.
This is a plotting script; no evolution or measurement code is modified.
"""
from pathlib import Path
from fractions import Fraction
import math
import re
import tomllib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.font_manager import FontProperties

# -------------------- Editable settings --------------------
HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "d4_virtual_spaces.toml"
STATE_KEY = None  # Use "D4" with the full selected_su2_bond_ranks.toml file.
OUTPUT_DIR = HERE / "figures"
OUTPUT_NAME = "kagome_d4_sqrt3_virtual_spaces"
DPI = 200
FULL_SPACE_LABELS = True
SAVE_PNG = True
SAVE_SVG = True
# ---------------- End of editable settings -----------------


def representation(value):
    terms = re.findall(r"(\d+(?:/\d+)?)\s*=>\s*(\d+)", value)
    if not terms:
        raise ValueError(f"Unsupported SU2 space: {value}")
    return tuple((Fraction(j), int(n)) for j, n in terms if int(n))


def space_math(space):
    parts = []
    for j, n in space:
        text = str(j.numerator) if j.denominator == 1 else rf"\frac{{{j.numerator}}}{{{j.denominator}}}"
        parts.extend([text] * n if n <= 3 else [rf"({text})^{{\oplus {n}}}"])
    return r"\oplus".join(parts)


def space_dimension(space):
    return sum(int(2*j + 1)*n for j, n in space)


def load_records(path, state_key=None):
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if state_key:
        data = data[state_key]
    meta = data.get("metadata", {})
    bonds = {(int(row["cell"]), int(row["bond"])): row for row in data["bonds"]}
    if set(bonds) != {(r, b) for r in (1, 2, 3) for b in (1, 2, 3)}:
        raise ValueError("Expected nine B tensors for a sqrt3 cell")
    spaces = {}
    for key, row in bonds.items():
        for side in ("up", "down"):
            space = representation(row[f"{side}_space"])
            if space_dimension(space) != int(row[f"{side}_dimension"]):
                raise ValueError(f"Dimension mismatch at {key}, {side}")
            spaces[key, side] = space
    return spaces, meta


def cell_geometry():
    """Rigid spatial embedding of the ORIGINAL Julia connectivity.

    U(R) connects B1(R),B2(R),B3(R).
    D(R) connects B1(R+ex),B2(R),B3(R+ey).
    B is the midpoint of its adjacent simplex centres.
    The drawn hexagon is a Wigner-Seitz fundamental domain of translations
    (1,1),(-1,2). It owns 9 B tensors and shows 3 periodic images.
    """
    ax, ay = np.array([1., 0.]), np.array([.5, -math.sqrt(3)/2])
    delta = (ax + ay)/3
    centre = np.array([.5, math.sqrt(3)/6])
    position = lambda xy: xy[0]*ax + xy[1]*ay
    rep = lambda xy: 1 + (xy[0] - xy[1]) % 3
    # These are periodic copies of the stored representatives (0,0),(1,0),(2,0).
    ups = {1: (0, 0), 2: (1, 0), 3: (1, -1)}
    downs = {1: (0, 0), 2: (0, -1), 3: (1, -1)}
    offsets = {1: (delta-ax)/2, 2: delta/2, 3: (delta-ay)/2}
    nodes, cores, edges = {}, {}, []
    for r, xy in ups.items():
        cores["U", r] = position(xy)-centre
        for b in (1, 2, 3):
            key = (xy, b)
            nodes[key] = dict(pos=position(xy)+offsets[b]-centre, rep=r, b=b, owned=True)
            edges.append((('U', r), key, "up"))
    for r, xy in downs.items():
        cores["D", r] = position(xy)+delta-centre
        for b, shift in ((1, (1, 0)), (2, (0, 0)), (3, (0, 1))):
            owner = (xy[0]+shift[0], xy[1]+shift[1])
            key = (owner, b)
            if key not in nodes:
                nodes[key] = dict(pos=position(owner)+offsets[b]-centre,
                                  rep=rep(owner), b=b, owned=False)
            edges.append((('D', r), key, "down"))
    assert len(cores) == 6 and len(edges) == 18
    assert sum(n['owned'] for n in nodes.values()) == 9 and len(nodes) == 12
    for core, node, side in edges:
        assert abs(np.linalg.norm(cores[core]-nodes[node]['pos'])-1/(2*math.sqrt(3))) < 1e-12
    # The three image sites must differ from their owned counterpart by
    # integer combinations of the exact sqrt3 translation vectors.
    translations = np.array([[1, -1], [1, 2]], dtype=float)
    for (xy, b), node in nodes.items():
        if not node['owned']:
            owned_xy = ups[node['rep']]
            coefficients = np.linalg.solve(translations, np.array(xy)-owned_xy)
            assert np.allclose(coefficients, np.round(coefficients))
    return nodes, cores, edges


def draw():
    spaces, metadata = load_records(INPUT_FILE, STATE_KEY)
    nodes, cores, edges = cell_geometry()
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    chinese = font_path.is_file()
    font = FontProperties(fname=str(font_path)) if chinese else FontProperties(family="DejaVu Sans")
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "kagome-virtual-cell",
                         "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(12, 12), facecolor="white")
    unique = sorted(set(spaces.values()), key=lambda x: (any(j.denominator == 2 for j, n in x), str(x)))
    extra_legend = .035*((len(unique)+1)//2-1)
    ax = fig.add_axes([.04, .205+extra_legend, .92, .715-extra_legend])
    ax.set(aspect="equal", xlim=(-1.14, 1.14), ylim=(-1.04, 1.08))
    ax.axis("off")
    vertices = np.array([(math.cos(t), math.sin(t)) for t in np.arange(6)*math.pi/3])
    ax.add_patch(Polygon(vertices, closed=True, facecolor="#fafbfc", edgecolor="#83909b",
                         linewidth=1.5, linestyle=(0, (5, 4)), zorder=0))
    def color(space):
        parity = {int(2*j) % 2 for j, n in space}
        return "#2166ac" if parity == {1} else "#d95f02" if parity == {0} else "#238b45"
    short = {space: rf"V_{{{i+1}}}" for i, space in enumerate(unique)}
    labels = []
    for core, key, side in edges:
        node = nodes[key]
        start, end = cores[core], node['pos']
        space = spaces[(node['rep'], node['b']), side]
        ax.plot(*np.array([start, end]).T, color=color(space), lw=3, zorder=2)
        midpoint = (start+end)/2
        label = space_math(space) if FULL_SPACE_LABELS else short[space]
        labels.append(ax.text(*midpoint, f"${label}$", fontsize=15, color="#20272c",
                              ha="center", va="center", zorder=5,
                              bbox=dict(facecolor="#fafbfc", edgecolor="none", pad=.08)))
    # Boundary B tensors have a matching periodic image on the opposite edge.
    for key, node in nodes.items():
        p = node['pos']; radial = p/np.linalg.norm(p)
        boundary = np.linalg.norm(p) > .8
        if boundary:
            outside_side = "down" if node['owned'] else "up"
            outside_space = spaces[(node['rep'], node['b']), outside_side]
            ax.plot(*np.array([p, p+.09*radial]).T, lw=2, color=color(outside_space),
                    linestyle=(0, (2, 2)), zorder=2)
        if node['owned']:
            direction = np.array([-radial[1], radial[0]]) if boundary else radial
            endpoint = p+.125*direction
            ax.plot(*np.array([p, endpoint]).T, lw=1.6, color="#303840", zorder=3)
            ax.plot(*endpoint, marker="o", markersize=2.6, color="#303840", zorder=3)
        ax.add_patch(Circle(p, .054, facecolor="white" if not node['owned'] else "#edf0f3",
                            edgecolor="#89939b" if not node['owned'] else "#303840",
                            lw=1.5, linestyle=(0, (2, 2)) if not node['owned'] else "solid", zorder=7))
        ax.text(*p, rf"$B_{{{node['rep']},{node['b']}}}$", ha="center", va="center",
                fontsize=13.5, color="#77818b" if not node['owned'] else "#20272c", zorder=8)
    anchor = int(metadata.get("auxiliary_anchor", metadata.get("anchor", 1)))
    aux_spin = metadata.get("auxiliary_spin", "1/2")
    if aux_spin:
        start = cores['U', anchor]
        endpoint = start*.24
        ax.plot(*np.array([start, endpoint]).T, color="#8246af", lw=2,
                linestyle=(0, (4, 3)), zorder=3)
        ax.plot(*endpoint, marker="s", markersize=7, markerfacecolor="white",
                markeredgecolor="#8246af", zorder=4)
        direction = endpoint-start
        perpendicular = np.array([-direction[1], direction[0]])/np.linalg.norm(direction)
        midpoint = (start+endpoint)/2 + .075*perpendicular
        aux_label = space_math(((Fraction(str(aux_spin)), 1),))
        ax.text(*midpoint, rf"$V_q={aux_label}$", color="#8246af", fontsize=16, ha="center", zorder=5,
                bbox=dict(facecolor="#fafbfc", edgecolor="none", pad=.1))
    for (kind, r), p in cores.items():
        angle = math.pi/2 if kind == 'U' else -math.pi/2
        polygon = p + .067*np.array([(math.cos(angle+k*2*math.pi/3), math.sin(angle+k*2*math.pi/3)) for k in range(3)])
        ax.add_patch(Polygon(polygon, closed=True, facecolor="#39434d", edgecolor="#39434d", zorder=9))
        ax.text(*(p+np.array([0, -.012 if kind == 'U' else .012])), rf"${kind}_{r}$",
                ha="center", va="center", color="white", fontsize=13, zorder=10)
    dmax = int(metadata.get("Dmax", max(map(space_dimension, spaces.values()))))
    title = rf"$D={dmax}$ kagome iPESS  /  $\sqrt{{3}}\times\sqrt{{3}}$"
    fig.text(.5, .967, title, ha="center", fontsize=22)
    subtitle = "18 条虚腿 · 9 个独立 B 张量 · 3 个上三角 + 3 个下三角" if chinese else "18 virtual legs / 9 independent B tensors / 3 up + 3 down simplices"
    fig.text(.5, .93, subtitle, ha="center", fontsize=13, fontproperties=font, color="#47515b")
    for i, space in enumerate(unique):
        x = .13 + (i%2)*.44
        y = .17 + extra_legend - (i//2)*.035
        fig.lines.append(plt.Line2D([x, x+.045], [y, y], transform=fig.transFigure,
                                    color=color(space), lw=3))
        formula = space_math(space) if FULL_SPACE_LABELS else short[space]+"="+space_math(space)
        fig.text(x+.06, y, f"${formula}$   (dim {space_dimension(space)})",
                 fontsize=17, va="center")
    legend = r"$U_r=T_u^{(r)},\quad D_r=T_d^{(r)},\quad B_{r,b}=B_b^{(r)},\quad p=\frac{1}{2}$"
    fig.text(.5, .125, legend, ha="center", fontsize=15)
    note = "空心同名 B 为周期像；细黑短腿为物理指标；紫色 q 为辅助指标" if chinese else "Hollow B: periodic image / black stub: physical leg / purple q: auxiliary leg"
    fig.text(.5, .085, note, ha="center", fontsize=13, fontproperties=font, color="#47515b")
    arrows = "空间标注省略对偶：B 端为 V*，相邻 T 端为 V。虚线六边形为元胞边界。" if chinese else "Dual notation suppressed: V* at B, V at T. Dashed hexagon: cell boundary."
    fig.text(.5, .05, arrows, ha="center", fontsize=12, fontproperties=font, color="#47515b")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    if SAVE_PNG:
        path = OUTPUT_DIR/(OUTPUT_NAME+".png"); fig.savefig(path, dpi=DPI, facecolor="white"); paths.append(path)
    if SAVE_SVG:
        path = OUTPUT_DIR/(OUTPUT_NAME+".svg")
        fig.savefig(path, facecolor="white", metadata={"Date": None})
        # Canonical whitespace for generated SVG paths and clean Git diffs.
        path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n",
                        encoding="utf-8")
        paths.append(path)
    plt.close(fig)
    print("Validated: 6 simplices, 9 owned B + 3 periodic images, 18 virtual legs.")
    for path in paths:
        print(path.resolve())
    return paths


if __name__ == "__main__":
    draw()
