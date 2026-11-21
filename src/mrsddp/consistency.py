"""Time consistency of the static mean-CVaR problem.

The static problem  min (1-lam) E[Z] + lam CVaR_alpha(Z),  Z = total cost of a path,
is solved once at the root. When the decision maker arrives at a node m and
re-optimizes the same criterion for the remaining horizon, the plan can change:
the problem is not time consistent.

Writing CVaR in Rockafellar-Uryasev form shows how to fix this. With the root
threshold eta* and the cost already paid on the way to m as extra state
variables, the conditional problem at m keeps the original plan. In the
language of Kovacova and Rudloff (2021) the weight of the conditional problem
moves along the tree instead of staying fixed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extensive import path_cost_distribution, solve_extensive
from .model import MultiStageModel
from .risk import MeanCVaR


@dataclass
class ReplanResult:
    node: int
    probability: float
    change_naive: float        # relative change of the node decision, naive re-planning
    change_augmented: float    # same with (eta*, past cost) carried as state
                               # (can be nonzero when the LP has several optimal plans)


def _leaves_under(nodes, m):
    out = []
    for n in nodes:
        if n.children:
            continue
        nid = n.id
        while nid is not None and nid != m:
            nid = nodes[nid].parent
        if nid == m:
            out.append(n.id)
    return out


def replanning_study(model: MultiStageModel, lam: float, alpha: float, stage: int = 2):
    """Solve the static problem at the root, re-plan at every node of `stage`
    and measure how much the plan and the root objective change."""
    rho = MeanCVaR(lam, alpha)
    root = solve_extensive(model, "static", lam=lam, alpha=alpha)
    nodes = root.nodes
    root_costs, root_probs = path_cost_distribution(model, root)
    leaf_pos = {n.id: i for i, n in enumerate([n for n in nodes if not n.children])}

    def stage_cost(nid):
        n = nodes[nid]
        return model.stage_lp(n.t, model.support(n.t)[0][n.k]).c @ root.y[nid]

    results = []
    naive_costs = root_costs.copy()
    aug_costs = root_costs.copy()
    for m in (n for n in nodes if n.t == stage):
        parent = nodes[m.parent]
        x_in = root.y[parent.id][model.stage_lp(parent.t, model.support(parent.t)[0][parent.k]).state]
        past, nid = 0.0, parent.id
        while nid is not None:
            past += stage_cost(nid)
            nid = nodes[nid].parent

        naive = solve_extensive(model, "static", lam=lam, alpha=alpha, t0=m.t, k0=m.k, x_in=x_in)
        aug = solve_extensive(model, "static", lam=lam, alpha=alpha, t0=m.t, k0=m.k, x_in=x_in,
                              past_cost=past, eta_fixed=root.eta)

        ref = root.y[m.id]
        scale = max(1e-9, np.abs(ref).sum())
        results.append(ReplanResult(m.id, m.prob,
                                    float(np.abs(naive.y[0] - ref).sum() / scale),
                                    float(np.abs(aug.y[0] - ref).sum() / scale)))

        # cost of the subtree if the naive re-plan is followed from m on
        under = _leaves_under(nodes, m.id)
        for target, sol in ((naive_costs, naive), (aug_costs, aug)):
            sub_costs, _ = path_cost_distribution(model, sol)
            for leaf, cost in zip(under, sub_costs):
                target[leaf_pos[leaf]] = past + cost

    return dict(
        root_value=root.value,
        root_value_check=rho(root_costs, root_probs),
        naive_value=rho(naive_costs, root_probs),
        augmented_value=rho(aug_costs, root_probs),
        replans=results,
    )
