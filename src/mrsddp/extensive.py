"""Extensive (deterministic equivalent) formulations on the full scenario tree.

Only for small trees. Used to check SDDP and to study time consistency.

objective="neutral"  expected total cost
objective="nested"   nested mean-CVaR, V_n = c'y_n + rho(V_children), written with
                     the Rockafellar-Uryasev representation at every node
objective="static"   mean-CVaR of the total cost of a path, evaluated once at the root
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .model import MultiStageModel


@dataclass
class Node:
    id: int
    t: int
    k: int
    parent: int | None
    prob: float          # probability relative to the root of the (sub)tree
    cprob: float         # probability given the parent
    children: list


def build_tree(model: MultiStageModel, t0: int = 1, k0: int | None = None) -> list[Node]:
    """Tree that starts at stage t0. If k0 is given the root is that realization,
    otherwise the root stage must have a single realization."""
    xis, p = model.support(t0)
    if k0 is None:
        if len(xis) != 1:
            raise ValueError("first stage must be deterministic, pass k0")
        k0 = 0
    nodes = [Node(0, t0, k0, None, 1.0, 1.0, [])]
    frontier = [0]
    for t in range(t0 + 1, model.T + 1):
        _, p = model.support(t)
        nxt = []
        for nid in frontier:
            for k, pk in enumerate(p):
                child = Node(len(nodes), t, k, nid, nodes[nid].prob * pk, pk, [])
                nodes.append(child)
                nodes[nid].children.append(child.id)
                nxt.append(child.id)
        frontier = nxt
    return nodes


@dataclass
class ExtensiveSolution:
    value: float
    y: dict              # node id -> stage decisions
    eta: float | None
    nodes: list


def solve_extensive(model: MultiStageModel, objective: str = "neutral", lam: float = 0.0,
                    alpha: float = 0.9, t0: int = 1, k0: int | None = None,
                    x_in: np.ndarray | None = None, past_cost: float = 0.0,
                    eta_fixed: float | None = None) -> ExtensiveSolution:
    nodes = build_tree(model, t0, k0)
    x_in = model.x0 if x_in is None else x_in
    lps = {n.id: model.stage_lp(n.t, model.support(n.t)[0][n.k]) for n in nodes}

    # column layout: y of every node, then objective-specific variables
    offset, col = {}, 0
    for n in nodes:
        offset[n.id] = col
        col += len(lps[n.id].c)
    ny_total = col

    def ycols(nid):
        return np.arange(offset[nid], offset[nid] + len(lps[nid].c))

    leaves = [n for n in nodes if not n.children]
    extra = {}
    if objective == "nested":
        for name, items in (("V", nodes), ("eta", [n for n in nodes if n.children]), ("u", nodes[1:])):
            extra[name] = {n.id: col + i for i, n in enumerate(items)}
            col += len(items)
    elif objective == "static":
        if eta_fixed is None:
            extra["eta"] = col
            col += 1
        extra["u"] = {n.id: col + i for i, n in enumerate(leaves)}
        col += len(leaves)
    elif objective != "neutral":
        raise ValueError(objective)
    N = col

    eq_rows, eq_rhs, ub_rows, ub_rhs = [], [], [], []

    def add(rows, rhs, entries, b):
        r = sparse.lil_matrix((1, N))
        for j, v in entries:
            r[0, j] += v
        rows.append(r.tocsr())
        rhs.append(b)

    # stage constraints; the parent state is y_parent[state]
    for n in nodes:
        lp = lps[n.id]
        yc = ycols(n.id)
        if n.parent is None:
            pc, x_const = None, x_in
        else:
            pc, x_const = ycols(n.parent)[lps[n.parent].state], None
        for A, E, b, rows, rhs in ((lp.A_eq, lp.E_eq, lp.b_eq, eq_rows, eq_rhs),
                                   (lp.A_ub, lp.E_ub, lp.b_ub, ub_rows, ub_rhs)):
            for i in range(len(b)):
                ent = [(yc[j], A[i, j]) for j in np.flatnonzero(A[i])]
                bi = b[i]
                if pc is None:
                    bi -= E[i] @ x_const
                else:
                    ent += [(pc[j], E[i, j]) for j in np.flatnonzero(E[i])]
                add(rows, rhs, ent, bi)

    c = np.zeros(N)
    bounds = [None] * N
    for n in nodes:
        for j, bd in zip(ycols(n.id), lps[n.id].bounds):
            bounds[j] = bd

    if objective == "neutral":
        for n in nodes:
            c[ycols(n.id)] += n.prob * lps[n.id].c
    elif objective == "nested":
        V, ETA, U = extra["V"], extra["eta"], extra["u"]
        for n in nodes:
            bounds[V[n.id]] = (None, None)
            ent = [(j, v) for j, v in zip(ycols(n.id), lps[n.id].c) if v] + [(V[n.id], -1.0)]
            if n.children:
                bounds[ETA[n.id]] = (None, None)
                ent.append((ETA[n.id], lam))
                for ch in n.children:
                    cp = nodes[ch].cprob
                    ent += [(V[ch], (1 - lam) * cp), (U[ch], lam * cp / (1 - alpha))]
                    bounds[U[ch]] = (0, None)
                    add(ub_rows, ub_rhs, [(V[ch], 1.0), (ETA[n.id], -1.0), (U[ch], -1.0)], 0.0)
            add(ub_rows, ub_rhs, ent, 0.0)
        c[V[0]] = 1.0
    else:  # static mean-CVaR of the path cost (plus a constant past cost)
        for n in nodes:
            c[ycols(n.id)] += (1 - lam) * n.prob * lps[n.id].c
        eta_col = extra.get("eta")
        if eta_col is not None:
            bounds[eta_col] = (None, None)
            c[eta_col] = lam
        for leaf in leaves:
            uc = extra["u"][leaf.id]
            bounds[uc] = (0, None)
            c[uc] = lam * leaf.prob / (1 - alpha)
            ent, nid = [(uc, -1.0)], leaf.id
            while nid is not None:
                ent += [(j, v) for j, v in zip(ycols(nid), lps[nid].c) if v]
                nid = nodes[nid].parent
            rhs = -past_cost
            if eta_col is not None:
                ent.append((eta_col, -1.0))
            else:
                rhs += eta_fixed
            add(ub_rows, ub_rhs, ent, rhs)

    res = linprog(c, A_ub=sparse.vstack(ub_rows) if ub_rows else None, b_ub=ub_rhs or None,
                  A_eq=sparse.vstack(eq_rows) if eq_rows else None, b_eq=eq_rhs or None,
                  bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"extensive form failed: {res.message}")

    value = res.fun
    eta = None
    if objective == "static":
        value += (1 - lam) * past_cost
        if extra.get("eta") is not None:
            eta = float(res.x[extra["eta"]])
        else:
            eta = eta_fixed
            value += lam * eta_fixed
    y = {n.id: res.x[ycols(n.id)] for n in nodes}
    return ExtensiveSolution(float(value), y, eta, nodes)


def path_cost_distribution(model: MultiStageModel, sol: ExtensiveSolution) -> tuple[np.ndarray, np.ndarray]:
    """Total cost and probability of every leaf path of a solved tree."""
    nodes = sol.nodes
    lp_c = {n.id: model.stage_lp(n.t, model.support(n.t)[0][n.k]).c for n in nodes}
    costs, probs = [], []
    for n in nodes:
        if n.children:
            continue
        total, nid = 0.0, n.id
        while nid is not None:
            total += lp_c[nid] @ sol.y[nid]
            nid = nodes[nid].parent
        costs.append(total)
        probs.append(n.prob)
    return np.array(costs), np.array(probs)
