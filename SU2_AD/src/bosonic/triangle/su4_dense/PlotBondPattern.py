"""Plot every nearest-neighbor bond, including unit-cell translations.

Usage: python PlotBondPattern.py /path/to/observables_bonds.csv
Requires numpy and matplotlib. No Julia evolution or direction averaging.
"""
from pathlib import Path
import argparse
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon


def load_bonds(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    nx = max(int(r["source_x"]) for r in rows)
    ny = max(int(r["source_y"]) for r in rows)
    assert len(rows) == 3 * nx * ny
    assert len({(r["source_x"], r["source_y"], r["direction"]) for r in rows}) == len(rows)
    for r in rows:
        for k in ("source_x", "source_y", "target_x", "target_y", "target_cell_x", "target_cell_y",
                  "displacement_x", "displacement_y", "bond_id"):
            r[k] = int(r[k])
        r["coupling"] = float(r["coupling"])
        dx, dy = r["displacement_x"], r["displacement_y"]
        assert r["target_x"] + nx*r["target_cell_x"] - r["source_x"] == dx
        assert r["target_y"] + ny*r["target_cell_y"] - r["source_y"] == dy
        assert math.isclose(np.linalg.norm(position(dx, dy)), 1.0)
    return rows, nx, ny


def position(u, v):
    # Primitive vectors at 120 degrees. x+y is the third nearest-neighbor vector.
    return np.array([u - v/2, np.sqrt(3)*v/2])


def site_name(u, v, nx, ny, cell_label=False):
    x, y = u % nx, v % ny
    name = "ABCD"[x + nx*y] if (nx, ny) == (2, 2) else f"{x+1},{y+1}"
    cx, cy = u // nx, v // ny
    return f"{name}({cx:+d},{cy:+d})" if cell_label and (cx or cy) else name


def outline(ax, u, v, nx, ny):
    corners = [(u-.5, v-.5), (u+nx-.5, v-.5), (u+nx-.5, v+ny-.5), (u-.5, v+ny-.5)]
    ax.add_patch(Polygon([position(*p) for p in corners], fill=False, edgecolor="#555555",
                         lw=1.2, ls=(0, (4, 3)), zorder=1))


def plot(path):
    rows, nx, ny = load_bonds(path)
    values = np.array([r["coupling"] for r in rows])
    lo, hi = values.min(), values.max()
    norm = Normalize(lo, hi if hi > lo else lo+1e-8)
    cmap = plt.get_cmap("coolwarm")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (left, right) = plt.subplots(1, 2, figsize=(18, 7), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.3, 1]})

    def edge(ax, start, end, value):
        a, b = position(*start), position(*end)
        strength = (hi-value)/(hi-lo) if hi > lo else .5
        ax.plot([a[0], b[0]], [a[1], b[1]], color=cmap(norm(value)), lw=2+5*strength,
                solid_capstyle="round", zorder=2)
        return a, b

    vertices = set()
    for r in rows:
        s = (r["source_x"]-1, r["source_y"]-1)
        t = (s[0]+r["displacement_x"], s[1]+r["displacement_y"])
        vertices.update((s, t))
        a, b = edge(left, s, t, r["coupling"])
        mid = (a+b)/2
        angle = math.degrees(math.atan2(*(b-a)[::-1]))
        if angle > 90:
            angle -= 180
        left.text(*mid, f'{r["bond_id"]}: {r["coupling"]:.6f}', rotation=angle,
                  rotation_mode="anchor", ha="center", va="center", fontsize=9,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93, "pad": 1.7}, zorder=4)
    outline(left, 0, 0, nx, ny)
    for u, v in sorted(vertices):
        p = position(u, v)
        home = 0 <= u < nx and 0 <= v < ny
        left.scatter(*p, s=210 if home else 135, facecolor="#263544" if home else "white",
                     edgecolor="#263544", lw=1.2, zorder=5)
        left.text(*p, site_name(u, v, nx, ny), ha="center", va="center",
                  color="white" if home else "#263544", fontsize=10, fontweight="bold", zorder=6)
        if not home:
            left.annotate(site_name(u, v, nx, ny, True), p, xytext=(0, 17), textcoords="offset points",
                          ha="center", fontsize=9, color="#444444", zorder=6)
    left.set_title("All independent bonds, with boundary connections\nLabels: bond ID and coupling; dashed outline = one unit cell", pad=20)
    left.margins(.22)

    lookup = {(r["source_x"]-1, r["source_y"]-1, r["direction"]): r for r in rows}
    width, height = 2*nx, 2*ny
    for u in range(width+1):
        for v in range(height+1):
            for direction in ("x", "y", "diagonal"):
                r = lookup[(u % nx, v % ny, direction)]
                target = (u+r["displacement_x"], v+r["displacement_y"])
                if target[0] <= width and target[1] <= height:
                    edge(right, (u, v), target, r["coupling"])
            p = position(u, v)
            right.scatter(*p, s=110, color="#263544", zorder=5)
            right.text(*p, site_name(u, v, nx, ny), ha="center", va="center", color="white", fontsize=8, zorder=6)
    for u in (0, nx):
        for v in (0, ny):
            outline(right, u, v, nx, ny)
    right.set_title("Periodic repetition of the same measured bonds\nStronger antiferromagnetic coupling = thicker line", pad=20)
    right.margins(.08)
    for ax in (left, right):
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=[left, right], shrink=.65, pad=.015,
                 label=r"Coupling $\langle P_{ij}\rangle-1/4$")
    environment=rows[0].get("environment","not specified")
    caption={"ctmrg":"CTMRG environment","simple_lambda":"Simple-lambda environment"}.get(environment,environment)
    fig.suptitle(f"{nx}x{ny} triangular cell: all {len(rows)} inequivalent nearest-neighbor bonds\n"
                 f"{caption}; cell translations retained explicitly", fontsize=15)
    stem=path.stem[:-6] if path.stem.endswith("_bonds") else path.stem
    output = path.with_name(stem+"_lattice.png")
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(output.resolve())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bonds_csv", type=Path)
    plot(parser.parse_args().bonds_csv)
