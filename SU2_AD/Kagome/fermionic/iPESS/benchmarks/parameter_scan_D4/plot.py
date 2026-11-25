"""Plot the completed numerical comparison; run summarize.jl first."""
from pathlib import Path
import tomllib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2] / "results" / "parameter_scan_D4"
with (root / "comparison.toml").open("rb") as f:
    data = tomllib.load(f)
points = data["points"]
colors = {"su": "#4c78a8", "fu": "#f58518", "ad": "#54a24b"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
for method in ("su", "fu", "ad"):
    # The isolated-triangle control can have roundoff-level error: show it in
    # the data table, rather than placing it on this logarithmic scale.
    pp = [p for p in points if p["ratio"] > 0 and p[method]["absolute_error"] > 0]
    axes[0].plot([p["ratio"] for p in pp], [p[method]["absolute_error"] for p in pp],
                 "o-", label=method.upper(), color=colors[method], alpha=0.85)
for method in ("fu", "ad"):
    pp = [p for p in points if p["ratio"] > 0]
    axes[1].plot([p["ratio"] for p in pp], [p[method]["energy_gain_from_su"] for p in pp],
                 "o-", label=method.upper(), color=colors[method])
extended = data.get("controls", {}).get("extended_fu_ratio_0p4")
if extended:
    baseline = next(p["su"]["energy"] for p in points if p["ratio"] == 0.4)
    axes[1].scatter([0.4], [baseline - extended["energy"]], marker="D", s=55,
                    facecolors="none", edgecolors=colors["fu"], label="FU, longer run at r=0.4")
axes[0].set_yscale("log")
axes[0].set_ylabel(r"Energy error per site, $E-E_{exact}$")
axes[0].set_title("Error against free-fermion exact energy")
axes[1].set_yscale("symlog", linthresh=1e-8)
axes[1].axhline(0, color="0.5", lw=0.7)
axes[1].set_ylabel(r"Energy gain per site, $E_{SU}-E_{method}$")
axes[1].set_title("Improvement from the same SU state")
for ax in axes:
    ax.set_xlabel(r"Breathing ratio $r=t_{down}/t_{up}$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
axes[1].legend(loc="lower right", frameon=True, framealpha=0.95)
fig.suptitle(r"Spinful kagome, $U=0$, $D_{max}=4$, $\mu=-(1+r)/2$" + "\nFinite optimization runs; not global optima", fontsize=12)
fig.savefig(root / "comparison.png", dpi=180)
fig.savefig(root / "comparison.pdf")
print(root / "comparison.png")
