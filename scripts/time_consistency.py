"""Re-planning the static mean-CVaR problem at stage 2 (results/time_consistency.md)."""
from pathlib import Path

from mrsddp.consistency import replanning_study
from mrsddp.problems import MultiRegionAllocation

ALPHA = 0.8
model = MultiRegionAllocation(T=4, regions=2, n_scen=3)
out = Path("results"); out.mkdir(exist_ok=True)
lines = ["| lam | root value | naive re-planning | loss | with (eta, past cost) as state |",
         "|---|---|---|---|---|"]
for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
    r = replanning_study(model, lam, ALPHA)
    lines.append(f"| {lam} | {r['root_value']:.3f} | {r['naive_value']:.3f} | "
                 f"{r['naive_value'] - r['root_value']:.3f} | {r['augmented_value']:.3f} |")
    print(lines[-1])
(out / "time_consistency.md").write_text("\n".join(lines) + "\n")
