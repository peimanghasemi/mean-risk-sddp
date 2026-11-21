"""Simulation-tuned target policies vs. SDDP on the allocation problem (results/policy_search.md)."""
import time
from pathlib import Path

import numpy as np

from mrsddp import SDDP, MeanCVaR, cvar
from mrsddp.policies import simulate_target_policy, spsa_tune
from mrsddp.problems import MultiRegionAllocation

ALPHA = 0.9
model = MultiRegionAllocation(T=6, regions=3, n_scen=5)
train = model.sample_indices(500, seed=7)
test = model.sample_indices(2000, seed=2024)
p = np.full(len(test), 1 / len(test))
out = Path("results"); out.mkdir(exist_ok=True)

lines = ["| lam | policy | mean cost | CVaR90 cost | mean-CVaR of total cost | gap to SDDP bound | time (s) |",
         "|---|---|---|---|---|---|---|"]
for lam in (0.0, 0.5):
    risk = MeanCVaR(lam, ALPHA)
    sddp = SDDP(model, risk)
    t0 = time.perf_counter()
    sddp.train(300, patience=25)
    t_sddp = time.perf_counter() - t0
    bound = sddp.log.lower_bound[-1]

    t0 = time.perf_counter()
    tuned = spsa_tune(model, risk, train, iterations=400)
    t_spsa = time.perf_counter() - t0

    rows = [("myopic (s = 1)", simulate_target_policy(model, np.ones(model.R), test), 0.0),
            (f"SPSA-tuned (s = {np.round(tuned.s, 2).tolist()})", simulate_target_policy(model, tuned.s, test), t_spsa),
            ("SDDP", sddp.simulate(paths=test), t_sddp)]
    for name, cost, secs in rows:
        obj = risk(cost, p)
        # the SDDP bound is for the nested objective; it is only comparable to the
        # simulated objective in the risk-neutral case
        gap = f"{(obj - bound) / bound:.1%}" if lam == 0 else "-"
        lines.append(f"| {lam} | {name} | {cost.mean():.2f} | {cvar(cost, p, ALPHA):.2f} | {obj:.2f} | "
                     f"{gap} | {secs:.1f} |")
        print(lines[-1])
(out / "policy_search.md").write_text("\n".join(lines) + "\n")
