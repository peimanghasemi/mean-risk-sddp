"""Mean-risk frontier of nested mean-CVaR policies (results/frontier_*.md).

For every lam a policy is trained with SDDP and evaluated out of sample on the
same test paths. Reported: mean and CVaR of the total cost (for the portfolio:
of terminal wealth, so larger is better).
"""
import time
from pathlib import Path

import numpy as np

from mrsddp import SDDP, MeanCVaR, cvar
from mrsddp.problems import MultiRegionAllocation, Portfolio

ALPHA = 0.9
out = Path("results"); out.mkdir(exist_ok=True)

# the portfolio moves to cash quickly, so it gets a finer grid of small lam
for name, model, sign, lams in [
        ("allocation", MultiRegionAllocation(T=6, regions=3, n_scen=5), 1, (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)),
        ("portfolio", Portfolio(T=5, assets=4, n_scen=8), -1, (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3))]:
    test = model.sample_indices(2000, seed=2024)
    p = np.full(len(test), 1 / len(test))
    head = ("| lam | mean cost | CVaR90 cost | bound | iterations | time (s) |" if sign > 0 else
            "| lam | mean wealth | mean of worst 10% wealth | bound | iterations | time (s) |")
    lines = [head, "|---|---|---|---|---|---|"]
    csv = ["lam,mean,cvar90"]
    for lam in lams:
        sddp = SDDP(model, MeanCVaR(lam, ALPHA))
        t0 = time.perf_counter()
        sddp.train(300, patience=25)
        cost = sddp.simulate(paths=test)
        if sign > 0:
            row = f"| {lam} | {cost.mean():.2f} | {cvar(cost, p, ALPHA):.2f} | {sddp.log.lower_bound[-1]:.2f}"
        else:
            row = f"| {lam} | {-cost.mean():.4f} | {-cvar(cost, p, ALPHA):.4f} | {-sddp.log.lower_bound[-1]:.4f}"
        lines.append(row + f" | {len(sddp.log.lower_bound)} | {time.perf_counter() - t0:.0f} |")
        csv.append(f"{lam},{sign * cost.mean()},{sign * cvar(cost, p, ALPHA)}")
        print(name, lines[-1])
    (out / f"frontier_{name}.md").write_text("\n".join(lines) + "\n")
    (out / f"frontier_{name}.csv").write_text("\n".join(csv) + "\n")
