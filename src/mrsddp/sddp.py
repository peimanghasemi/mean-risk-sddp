"""SDDP for nested mean-CVaR objectives.

Backward pass at stage t and trial state x_hat (from stage t-1):

    theta_{t-1} >= sum_j w_j (Q_j + g_j^T (x - x_hat)),

with Q_j, g_j the value and subgradient of stage t for realization j and w the
mean-CVaR weights of (Q_j). lam = 0 gives risk-neutral SDDP (Pereira and Pinto
1991); lam > 0 the risk-averse version of Shapiro (2011) and Philpott and de
Matos (2012). The first-stage value is a lower bound on the nested objective.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .model import MultiStageModel, solve_stage
from .risk import MeanCVaR


@dataclass
class Log:
    lower_bound: list = field(default_factory=list)
    seconds: float = 0.0


class SDDP:
    def __init__(self, model: MultiStageModel, risk: MeanCVaR = MeanCVaR()):
        self.model, self.risk = model, risk
        self.cuts = {t: [] for t in range(1, model.T)}     # cuts[t]: cost of stages t+1..T
        self.log = Log()

    def _solve(self, t, xi, x):
        m = self.model
        return solve_stage(m.stage_lp(t, xi), x, self.cuts.get(t), m.theta_lb)

    def _forward(self, path):
        x, states = self.model.x0, []
        for t in range(1, self.model.T + 1):
            xis, _ = self.model.support(t)
            x = self._solve(t, xis[path[t - 1]], x).state
            states.append(x)
        return states

    def _backward(self, states):
        for t in range(self.model.T, 1, -1):
            x_hat = states[t - 2]
            xis, p = self.model.support(t)
            sols = [self._solve(t, xi, x_hat) for xi in xis]
            Q = np.array([s.value for s in sols])
            G = np.array([s.grad for s in sols])
            w = self.risk.weights(Q, p)
            g = w @ G
            self.cuts[t - 1].append((float(w @ Q - g @ x_hat), g))

    def first_stage_value(self) -> float:
        xis, p = self.model.support(1)
        vals = np.array([self._solve(1, xi, self.model.x0).value for xi in xis])
        return self.risk(vals, p)

    def train(self, iterations: int = 200, seed: int = 0, patience: int = 20, tol: float = 1e-9,
              verbose: bool = False) -> "SDDP":
        """Stop after `iterations` or when the bound has not moved for `patience` iterations."""
        start = time.perf_counter()
        paths = self.model.sample_indices(iterations, seed)
        still = 0
        for it, path in enumerate(paths):
            self._backward(self._forward(path))
            lb = self.first_stage_value()
            prev = self.log.lower_bound[-1] if self.log.lower_bound else None
            self.log.lower_bound.append(lb)
            if verbose and it % 10 == 0:
                print(f"{it:4d}  {lb:.6f}")
            if prev is not None and abs(lb - prev) <= tol * max(1.0, abs(lb)):
                still += 1
                if still >= patience:
                    break
            else:
                still = 0
        self.log.seconds += time.perf_counter() - start
        return self

    def simulate(self, n: int | None = None, seed: int = 1, paths: np.ndarray | None = None) -> np.ndarray:
        """Total cost of the policy on sampled (or given) paths."""
        if paths is None:
            paths = self.model.sample_indices(n, seed)
        out = np.empty(len(paths))
        for k, path in enumerate(paths):
            x, total = self.model.x0, 0.0
            for t in range(1, self.model.T + 1):
                xis, _ = self.model.support(t)
                sol = self._solve(t, xis[path[t - 1]], x)
                total += sol.stage_cost
                x = sol.state
            out[k] = total
        return out
