"""Multi-stage stochastic linear programs with stagewise independent, finitely supported noise.

At stage t = 1..T the realization xi_t is observed and the stage problem

    min  c^T y + theta
    s.t. A_eq y + E_eq x = b_eq
         A_ub y + E_ub x <= b_ub
         bounds on y

is solved, where x is the state coming from stage t-1 and y[state] is the state
passed on. theta approximates the (risk-adjusted) cost of stages t+1..T.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class StageLP:
    c: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    E_eq: np.ndarray
    A_ub: np.ndarray
    b_ub: np.ndarray
    E_ub: np.ndarray
    bounds: list
    state: np.ndarray          # indices of y that form the outgoing state


class MultiStageModel:
    """Subclasses set T, n_state, x0, theta_lb and implement support() and stage_lp()."""

    T: int
    n_state: int
    x0: np.ndarray
    theta_lb: float = 0.0

    def support(self, t: int) -> tuple[list, np.ndarray]:
        """Realizations and probabilities of the noise at stage t (1-based)."""
        raise NotImplementedError

    def stage_lp(self, t: int, xi) -> StageLP:
        raise NotImplementedError

    def sample_indices(self, n: int, seed: int = 0) -> np.ndarray:
        """n sample paths, given as the index of the realization at every stage."""
        rng = np.random.default_rng(seed)
        cols = []
        for t in range(1, self.T + 1):
            _, p = self.support(t)
            cols.append(rng.choice(len(p), size=n, p=p))
        return np.column_stack(cols)


@dataclass
class StageSolution:
    value: float           # c^T y + theta
    stage_cost: float      # c^T y
    y: np.ndarray
    state: np.ndarray
    grad: np.ndarray       # subgradient of value w.r.t. the incoming state


def solve_stage(lp: StageLP, x: np.ndarray, cuts: list | None, theta_lb: float) -> StageSolution:
    """Solve a stage LP for incoming state x. cuts=None means no cost-to-go (last stage).

    A cut (a, g) reads theta >= a + g^T y[state].
    """
    ny = len(lp.c)
    nv = ny + (cuts is not None)

    c = np.zeros(nv); c[:ny] = lp.c
    A_eq = np.zeros((len(lp.b_eq), nv)); A_eq[:, :ny] = lp.A_eq
    b_eq = lp.b_eq - lp.E_eq @ x

    rows, rhs = [], []
    if len(lp.b_ub):
        A = np.zeros((len(lp.b_ub), nv)); A[:, :ny] = lp.A_ub
        rows.append(A); rhs.append(lp.b_ub - lp.E_ub @ x)
    bounds = list(lp.bounds)
    if cuts is not None:
        c[ny] = 1.0
        bounds.append((theta_lb, None))
        if cuts:
            A = np.zeros((len(cuts), nv))
            A[:, ny] = -1.0
            A[:, lp.state] = np.array([g for _, g in cuts])
            rows.append(A); rhs.append(-np.array([a for a, _ in cuts]))

    res = linprog(c, A_ub=np.vstack(rows) if rows else None, b_ub=np.concatenate(rhs) if rhs else None,
                  A_eq=A_eq if len(b_eq) else None, b_eq=b_eq if len(b_eq) else None,
                  bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"stage LP failed: {res.message}")

    # b depends on x through -E x, so d value / d x = -E^T (duals)
    grad = np.zeros(len(x))
    if len(b_eq):
        grad -= lp.E_eq.T @ res.eqlin.marginals
    if len(lp.b_ub):
        grad -= lp.E_ub.T @ res.ineqlin.marginals[:len(lp.b_ub)]
    y = res.x[:ny]
    return StageSolution(float(res.fun), float(lp.c @ y), y, y[lp.state].copy(), grad)
