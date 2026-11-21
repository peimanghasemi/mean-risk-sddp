"""Mean-CVaR on discrete distributions of costs.

rho(Z) = (1 - lam) E[Z] + lam CVaR_alpha(Z), with CVaR_alpha the mean of the worst
(1 - alpha) share of the costs. rho is coherent, so rho(Z) = sum_j w_j z_j for the
worst-case weights w of its dual set; these weights are what SDDP needs for cuts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cvar_weights(v: np.ndarray, p: np.ndarray, alpha: float) -> np.ndarray:
    """Dual weights of CVaR_alpha: mass p_j / (1 - alpha) on the largest values."""
    p = np.asarray(p, float)
    if alpha <= 0:
        return p.copy()
    w = np.zeros_like(p)
    left = 1.0
    for j in np.argsort(-np.asarray(v, float), kind="stable"):
        w[j] = min(p[j] / (1 - alpha), left)
        left -= w[j]
        if left <= 1e-15:
            break
    return w


def cvar(v, p=None, alpha: float = 0.9) -> float:
    v = np.asarray(v, float)
    p = np.full(len(v), 1 / len(v)) if p is None else np.asarray(p, float)
    return float(cvar_weights(v, p, alpha) @ v)


@dataclass(frozen=True)
class MeanCVaR:
    lam: float = 0.0
    alpha: float = 0.9

    def weights(self, v, p) -> np.ndarray:
        p = np.asarray(p, float)
        if self.lam == 0:
            return p.copy()
        return (1 - self.lam) * p + self.lam * cvar_weights(v, p, self.alpha)

    def __call__(self, v, p=None) -> float:
        v = np.asarray(v, float)
        p = np.full(len(v), 1 / len(v)) if p is None else p
        return float(self.weights(v, p) @ v)
