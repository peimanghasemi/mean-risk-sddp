"""Simple parametric policies for the allocation problem, tuned by simulation.

The policy ships up to a target level s_r * E[demand_{t+1,r}] in every region
and scales shipments down proportionally when the capacity is exceeded.
The multipliers s are tuned with SPSA (Spall 1992) on a fixed set of training
paths, projected onto a box after every step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problems.inventory import MultiRegionAllocation
from .risk import MeanCVaR


def simulate_target_policy(model: MultiRegionAllocation, s: np.ndarray, paths: np.ndarray) -> np.ndarray:
    """Total cost of the target policy on every path (vectorized over paths)."""
    R, T = model.R, model.T
    n = len(paths)
    stock = np.zeros((n, R))
    transit = np.zeros((n, R))
    total = np.zeros(n)
    for t in range(1, T + 1):
        demand = np.stack(model.support(t)[0])[paths[:, t - 1]]
        available = stock + transit
        served = np.minimum(available, demand)
        stock = np.minimum(available - served, model.storage)
        lost = demand - served
        if t < T:
            target = s * model.mean[t]                 # mean demand of the next period
            ship = np.maximum(0.0, target - stock)
            over = ship.sum(axis=1, keepdims=True) / model.capacity
            ship = ship / np.maximum(1.0, over)
        else:
            ship = np.zeros_like(stock)
        transit = ship
        total += (model.costs["lost"] * lost.sum(1) + model.costs["hold"] * stock.sum(1)
                  + model.costs["ship"] * ship.sum(1))
    return total


@dataclass
class SPSAResult:
    s: np.ndarray
    history: list = field(default_factory=list)


def spsa_tune(model: MultiRegionAllocation, risk: MeanCVaR, train_paths: np.ndarray,
              iterations: int = 300, s0: float = 1.0, box: tuple = (0.0, 4.0),
              a: float = 0.05, c: float = 0.1, seed: int = 0) -> SPSAResult:
    rng = np.random.default_rng(seed)
    probs = np.full(len(train_paths), 1.0 / len(train_paths))

    def f(s):
        return risk(simulate_target_policy(model, s, train_paths), probs)

    s = np.full(model.R, s0)
    scale = max(1.0, abs(f(s)))
    res = SPSAResult(s.copy())
    for k in range(iterations):
        ak = a / (k + 1 + 0.1 * iterations) ** 0.602
        ck = c / (k + 1) ** 0.101
        delta = rng.choice([-1.0, 1.0], size=model.R)
        g = (f(np.clip(s + ck * delta, *box)) - f(np.clip(s - ck * delta, *box))) / (2 * ck * delta)
        s = np.clip(s - ak * g / scale, *box)
        res.history.append(f(s))
    res.s = s
    return res
