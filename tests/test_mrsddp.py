import numpy as np
import pytest

from mrsddp.consistency import replanning_study
from mrsddp.extensive import solve_extensive
from mrsddp.policies import simulate_target_policy, spsa_tune
from mrsddp.problems import MultiRegionAllocation, Portfolio
from mrsddp.risk import MeanCVaR, cvar, cvar_weights
from mrsddp.sddp import SDDP


def small_models():
    return [MultiRegionAllocation(T=4, regions=2, n_scen=3, seed=1), Portfolio(T=4, assets=2, n_scen=3, seed=1)]


def test_cvar_simple_cases():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.full(4, 0.25)
    assert cvar(v, p, 0.75) == pytest.approx(4.0)
    assert cvar(v, p, 0.5) == pytest.approx(3.5)
    assert cvar(v, p, 0.0) == pytest.approx(2.5)
    w = cvar_weights(v, p, 0.6)
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()


def test_mean_cvar_is_between_mean_and_cvar():
    rng = np.random.default_rng(0)
    v, p = rng.normal(size=50), np.full(50, 0.02)
    lo, hi = v @ p, cvar(v, p, 0.9)
    for lam in (0.0, 0.3, 1.0):
        assert lo - 1e-12 <= MeanCVaR(lam, 0.9)(v, p) <= hi + 1e-12


@pytest.mark.parametrize("model", small_models(), ids=["allocation", "portfolio"])
@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_sddp_matches_extensive_form(model, lam):
    exact = solve_extensive(model, "nested" if lam else "neutral", lam=lam, alpha=0.8).value
    sddp = SDDP(model, MeanCVaR(lam, 0.8))
    sddp.train(300, patience=30)
    lb = sddp.log.lower_bound
    assert lb[-1] == pytest.approx(exact, rel=1e-6, abs=1e-8)
    assert all(b >= a - 1e-8 for a, b in zip(lb, lb[1:]))   # bound never decreases


def test_simulated_cost_is_consistent_with_bound():
    model = MultiRegionAllocation(T=4, regions=2, n_scen=3, seed=2)
    sddp = SDDP(model)
    sddp.train(300, patience=30)
    costs = sddp.simulate(4000, seed=3)
    half_width = 4 * costs.std() / np.sqrt(len(costs))
    assert abs(costs.mean() - sddp.log.lower_bound[-1]) <= half_width


def test_nested_equals_neutral_when_lam_is_zero():
    model = MultiRegionAllocation(T=3, regions=2, n_scen=3)
    a = solve_extensive(model, "neutral").value
    b = solve_extensive(model, "nested", lam=0.0).value
    c = solve_extensive(model, "static", lam=0.0).value
    assert a == pytest.approx(b) and a == pytest.approx(c)


@pytest.mark.parametrize("lam", [0.3, 0.7])
def test_augmented_replanning_keeps_the_root_value(lam):
    model = MultiRegionAllocation(T=4, regions=2, n_scen=3)
    out = replanning_study(model, lam, 0.8)
    assert out["root_value"] == pytest.approx(out["root_value_check"], rel=1e-7)
    assert out["augmented_value"] == pytest.approx(out["root_value"], rel=1e-7)
    assert out["naive_value"] >= out["root_value"] - 1e-7


def test_target_policy_without_shipping_loses_all_demand():
    model = MultiRegionAllocation(T=4, regions=2, n_scen=3)
    paths = model.sample_indices(50, 0)
    cost = simulate_target_policy(model, np.zeros(2), paths)
    demand = sum(np.stack(model.support(t)[0])[paths[:, t - 1]].sum(1) for t in range(1, 5))
    assert np.allclose(cost, model.costs["lost"] * demand)


def test_spsa_improves_the_training_objective():
    model = MultiRegionAllocation(T=5, regions=3, n_scen=4)
    paths = model.sample_indices(200, 0)
    res = spsa_tune(model, MeanCVaR(0.0), paths, iterations=100, s0=0.5)
    start = simulate_target_policy(model, np.full(3, 0.5), paths).mean()
    assert res.history[-1] < start
