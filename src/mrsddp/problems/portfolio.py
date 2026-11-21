"""Multi-period portfolio selection with proportional transaction costs.

The investor starts with cash, can rebalance at every stage, pays a fee kappa on
every trade and is evaluated on terminal wealth. Costs are negative wealth, so
minimizing a mean-CVaR of costs is the usual mean-risk trade-off on wealth.

State x_t = holdings after trading (cash, asset_1..n).
Stage variables y = (holdings, buy, sell).
"""
from __future__ import annotations

import numpy as np

from ..model import MultiStageModel, StageLP


class Portfolio(MultiStageModel):
    def __init__(self, T: int = 5, assets: int = 3, n_scen: int = 6, fee: float = 0.005,
                 rate: float = 0.002, wealth: float = 1.0, seed: int = 0):
        self.T, self.n = T, assets
        self.n_state = assets + 1
        self.x0 = np.r_[wealth, np.zeros(assets)]
        self.fee, self.rate = fee, rate
        # very loose bound on the cost-to-go (terminal wealth is bounded on a finite tree)
        self.theta_lb = -wealth * 10.0

        rng = np.random.default_rng(seed)
        mu = np.linspace(0.004, 0.02, assets)             # riskier assets earn more on average
        vol = np.linspace(0.015, 0.07, assets)
        corr = 0.3 + 0.7 * np.eye(assets)
        cov = np.outer(vol, vol) * corr
        self._support = {1: ([np.zeros(assets)], np.ones(1))}
        for t in range(2, T + 1):
            r = rng.multivariate_normal(mu, cov, size=n_scen)
            r -= r.mean(axis=0) - mu                      # match the mean exactly
            self._support[t] = (list(r), np.full(n_scen, 1.0 / n_scen))

    def support(self, t):
        return self._support[t]

    def stage_lp(self, t, returns):
        n = self.n
        h = np.arange(n + 1)                 # holdings: cash, assets
        buy = n + 1 + np.arange(n)
        sell = 2 * n + 1 + np.arange(n)
        ny = 3 * n + 1
        growth = np.r_[1 + self.rate, 1 + returns]

        c = np.zeros(ny)
        if t == self.T:
            c[h] = -1.0                      # terminal wealth

        A_eq = np.zeros((n + 1, ny)); E_eq = -np.diag(growth); b_eq = np.zeros(n + 1)
        # cash: h_0 = growth_0 x_0 - (1 + fee) sum buy + (1 - fee) sum sell
        A_eq[0, h[0]] = 1.0
        A_eq[0, buy] = 1 + self.fee
        A_eq[0, sell] = -(1 - self.fee)
        # assets: h_i = growth_i x_i + buy_i - sell_i
        for i in range(n):
            A_eq[1 + i, h[1 + i]] = 1.0
            A_eq[1 + i, buy[i]] = -1.0
            A_eq[1 + i, sell[i]] = 1.0

        A_ub = np.zeros((0, ny)); b_ub = np.zeros(0); E_ub = np.zeros((0, self.n_state))
        bounds = [(0, None)] * ny
        if t == self.T:                      # no trading at the end, wealth is marked to market
            for j in np.r_[buy, sell]:
                bounds[j] = (0, 0)
        return StageLP(c, A_eq, b_eq, E_eq, A_ub, b_ub, E_ub, bounds, h)
