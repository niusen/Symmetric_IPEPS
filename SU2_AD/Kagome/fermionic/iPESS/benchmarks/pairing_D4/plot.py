from pathlib import Path
import tomllib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]/"results"/"pairing_D4"
with (root/"comparison.toml").open("rb") as f:
    data=tomllib.load(f)
points=data["points"]
x=[p["delta"] for p in points]
colors={"su":"#4c78a8","fu":"#f58518","ad":"#54a24b"}
fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
for method in ("su","fu","ad"):
    axes[0].plot(x,[p[method]["energy_error"] for p in points],"o-",color=colors[method],label=method.upper())
    axes[2].plot(x,[p[method]["pair_real_error"] for p in points],"o-",color=colors[method],label=method.upper())
for method in ("fu","ad"):
    axes[1].plot(x,[p[method]["energy_gain_from_su"] for p in points],"o-",color=colors[method],label=method.upper())
axes[0].set_yscale("log")
axes[0].set_ylabel(r"$E-E_{exact}$ per site")
axes[1].set_yscale("symlog",linthresh=1e-8)
axes[1].set_ylabel(r"$E_{SU}-E_{method}$ per site")
axes[2].set_ylabel(r"$Re\langle c_\downarrow c_\uparrow\rangle-F_{exact}$")
for ax,title in zip(axes,("Energy error","Energy gain from SU","Pair amplitude error")):
    ax.set_title(title)
    ax.set_xlabel(r"Pairing source $\Delta$")
    ax.set_xticks(x)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
fig.suptitle(r"Uniform spinful kagome: $D_{max}=4$, $t=1$, $\mu=-1$, $U=0$"+"\nFinite optimization runs; explicit pairing source",fontsize=12)
fig.savefig(root/"comparison.png",dpi=180)
print(root/"comparison.png")
