"""SDDP lower bound vs. the extensive form on small trees (results/validation.md)."""
import time
from pathlib import Path

from mrsddp import SDDP, MeanCVaR
from mrsddp.extensive import solve_extensive
from mrsddp.problems import MultiRegionAllocation, Portfolio

ALPHA = 0.8
out = Path("results"); out.mkdir(exist_ok=True)
lines = ["| problem | lam | extensive form | SDDP bound | rel. gap | iterations | SDDP time (s) |",
         "|---|---|---|---|---|---|---|"]
for name, model in [("allocation T=4 R=2 S=3", MultiRegionAllocation(T=4, regions=2, n_scen=3)),
                    ("allocation T=5 R=2 S=3", MultiRegionAllocation(T=5, regions=2, n_scen=3)),
                    ("portfolio T=4 n=2 S=3", Portfolio(T=4, assets=2, n_scen=3)),
                    ("portfolio T=5 n=3 S=3", Portfolio(T=5, assets=3, n_scen=3))]:
    for lam in (0.0, 0.5, 1.0):
        exact = solve_extensive(model, "nested" if lam else "neutral", lam=lam, alpha=ALPHA).value
        sddp = SDDP(model, MeanCVaR(lam, ALPHA))
        t0 = time.perf_counter()
        sddp.train(400, patience=30)
        lb = sddp.log.lower_bound[-1]
        gap = abs(lb - exact) / max(1.0, abs(exact))
        lines.append(f"| {name} | {lam} | {exact:.6f} | {lb:.6f} | {gap:.1e} | {len(sddp.log.lower_bound)} "
                     f"| {time.perf_counter() - t0:.1f} |")
        print(lines[-1])
(out / "validation.md").write_text("\n".join(lines) + "\n")
