"""Multi-region allocation of a scarce resource with a one-period lead time.

Every period a central supplier with capacity C ships to R regions. Shipments
decided in period t arrive in t+1. Demand is random, unmet demand is lost and
penalized, leftover stock is held at a cost.

State x_t = (stock_1..R, in_transit_1..R).
Stage variables y = (served, lost, stock, ship), each of length R.
"""
from __future__ import annotations

import numpy as np

from ..model import MultiStageModel, StageLP


class MultiRegionAllocation(MultiStageModel):
    def __init__(self, T: int = 6, regions: int = 3, n_scen: int = 5, capacity: float | None = None,
                 mean_demand: float = 10.0, cv: float = 0.5, season: float = 0.3,
                 ship_cost: float = 1.0, hold_cost: float = 0.5, lost_cost: float = 8.0,
                 storage: float = 60.0, seed: int = 0):
        self.T, self.R = T, regions
        self.n_state = 2 * regions
        self.x0 = np.zeros(self.n_state)
        self.theta_lb = 0.0
        rng = np.random.default_rng(seed)
        # regions differ in size; demand follows a seasonal pattern over the horizon
        scale = mean_demand * rng.uniform(0.6, 1.4, regions)
        self.mean = np.array([scale * (1 + season * np.sin(2 * np.pi * (t - 1) / T)) for t in range(1, T + 1)])
        self.capacity = capacity if capacity is not None else 1.05 * self.mean.sum(axis=1).mean()
        self.costs = dict(ship=ship_cost, hold=hold_cost, lost=lost_cost)
        self.storage = storage

        k = 1.0 / cv ** 2       # gamma shape for the given coefficient of variation
        self._support = {1: ([np.zeros(regions)], np.ones(1))}
        for t in range(2, T + 1):
            d = rng.gamma(k, self.mean[t - 1] / k, size=(n_scen, regions)).round(1)
            self._support[t] = (list(d), np.full(n_scen, 1.0 / n_scen))

    def support(self, t):
        return self._support[t]

    def stage_lp(self, t, demand):
        R = self.R
        served, lost, stock, ship = (np.arange(R) + i * R for i in range(4))
        ny = 4 * R
        c = np.zeros(ny)
        c[lost] = self.costs["lost"]
        c[stock] = self.costs["hold"]
        c[ship] = self.costs["ship"]

        I = np.eye(R)
        A_eq = np.zeros((2 * R, ny)); E_eq = np.zeros((2 * R, self.n_state)); b_eq = np.zeros(2 * R)
        # stock = old stock + arrivals - served
        A_eq[:R, stock] = I; A_eq[:R, served] = I
        E_eq[:R, :R] = -I; E_eq[:R, R:] = -I
        # served + lost = demand
        A_eq[R:, served] = I; A_eq[R:, lost] = I
        b_eq[R:] = demand

        A_ub = np.zeros((1, ny)); A_ub[0, ship] = 1.0
        b_ub = np.array([self.capacity if t < self.T else 0.0])
        E_ub = np.zeros((1, self.n_state))

        bounds = [(0, None)] * ny
        for j in stock:
            bounds[j] = (0, self.storage)
        return StageLP(c, A_eq, b_eq, E_eq, A_ub, b_ub, E_ub, bounds, np.r_[stock, ship])

    # closed-form simulation used by the simple policies
    def step(self, t, x, demand, ship):
        R = self.R
        available = x[:R] + x[R:]
        served = np.minimum(available, demand)
        stock = np.minimum(available - served, self.storage)
        lost = demand - served
        cost = (self.costs["lost"] * lost.sum() + self.costs["hold"] * stock.sum()
                + self.costs["ship"] * ship.sum())
        return np.r_[stock, ship], cost
