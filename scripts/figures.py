"""Figures for the README (figures/*.png). Run frontier.py first."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mrsddp import SDDP, MeanCVaR
from mrsddp.problems import MultiRegionAllocation

out = Path("figures"); out.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. convergence of the lower bound on the allocation problem
model = MultiRegionAllocation(T=6, regions=3, n_scen=5)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for lam, style in ((0.0, "-"), (0.5, "--"), (1.0, ":")):
    sddp = SDDP(model, MeanCVaR(lam, 0.9)).train(300, patience=25)
    ax.plot(np.arange(1, len(sddp.log.lower_bound) + 1), sddp.log.lower_bound, style, color="black",
            label=f"lam = {lam}")
ax.set_xlabel("iteration")
ax.set_ylabel("lower bound")
ax.set_title("SDDP lower bound, allocation problem (T = 6, 3 regions)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "convergence.png", dpi=150)

# 2. mean-risk frontiers from results/frontier_*.csv
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, name, xl, yl, digits in ((axes[0], "allocation", "mean total cost", "CVaR 90% of total cost", 0),
                                 (axes[1], "portfolio", "mean terminal wealth", "mean of worst 10% wealth", 3)):
    d = np.genfromtxt(f"results/frontier_{name}.csv", delimiter=",", names=True)
    ax.plot(d["mean"], d["cvar90"], "o-", color="black", ms=4)
    # policies that (almost) coincide share one label
    groups = {}
    for lam, x, y in zip(d["lam"], d["mean"], d["cvar90"]):
        groups.setdefault((round(x, digits), round(y, digits)), []).append((lam, x, y))
    for members in groups.values():
        lams = [m[0] for m in members]
        text = f"{lams[0]:g}" if len(lams) == 1 else f"{lams[0]:g} to {lams[-1]:g}"
        ax.annotate(text, members[0][1:], textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(name)
fig.suptitle("Out-of-sample mean-risk trade-off of nested mean-CVaR policies (labels: lam)", fontsize=10)
fig.tight_layout()
fig.savefig(out / "frontier.png", dpi=150)
print("figures written")
