"""Structural verification, cone presence, edge cases, duality properties."""
import cvxpy as cp
import numpy as np
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.second_order import SOC

from cvxdual import dualize, solve_dual


def _is_solved(status: str) -> bool:
    return status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


# -- Cone presence in dual --

def test_soc_constraint_present():
    x = cp.Variable(3)
    t = cp.Variable()
    dual = dualize(cp.Problem(cp.Minimize(t), [SOC(t, x), cp.sum(x) == 1])).dual_problem
    assert any(isinstance(c, SOC) for c in dual.constraints)

def test_exp_constraint_present():
    x, y, z = cp.Variable(), cp.Variable(), cp.Variable()
    dual = dualize(cp.Problem(cp.Minimize(z), [ExpCone(x, y, z), y == 1, x >= 0])).dual_problem
    assert any(isinstance(c, ExpCone) for c in dual.constraints)

def test_pow_constraint_present():
    x = cp.Variable(nonneg=True)
    y = cp.Variable(nonneg=True)
    z = cp.Variable()
    dual = dualize(cp.Problem(cp.Minimize(x + y), [PowCone3D(x, y, z, 0.4), z == 1.0])).dual_problem
    assert any(isinstance(c, PowCone3D) for c in dual.constraints)

def test_psd_constraint_present():
    X = cp.Variable((2, 2), symmetric=True)
    dual = dualize(cp.Problem(cp.Minimize(cp.trace(X)), [X >> np.eye(2)])).dual_problem
    assert any(isinstance(c, PSD) for c in dual.constraints)


# -- Structural properties --

def test_maximize_to_minimize_conversion():
    x = cp.Variable(2)
    dual = dualize(cp.Problem(cp.Maximize(x[0] + x[1]), [x <= 1])).dual_problem
    assert isinstance(dual.objective, cp.Maximize)

def test_multiple_soc_cones():
    x1, t1 = cp.Variable(2), cp.Variable()
    x2, t2 = cp.Variable(3), cp.Variable()
    dual = dualize(cp.Problem(
        cp.Minimize(t1 + t2),
        [cp.norm(x1, 2) <= t1, cp.norm(x2, 2) <= t2, t1 >= 0, t2 >= 0],
    )).dual_problem
    assert sum(1 for c in dual.constraints if isinstance(c, SOC)) >= 2

def test_mixed_soc_plus_exp():
    x = cp.Variable(2)
    y, z = cp.Variable(), cp.Variable()
    dual = dualize(cp.Problem(
        cp.Minimize(cp.sum(x) + z), [cp.norm(x, 2) <= 1, cp.exp(y) <= z, y >= 0],
    )).dual_problem
    assert any(isinstance(c, SOC) for c in dual.constraints)
    assert any(isinstance(c, ExpCone) for c in dual.constraints)

def test_all_cone_types():
    x_lp = cp.Variable(2, nonneg=True)
    x_soc, t_soc = cp.Variable(2), cp.Variable()
    x_exp, z_exp = cp.Variable(), cp.Variable()
    x_pow, y_pow = cp.Variable(pos=True), cp.Variable(pos=True)
    dual = dualize(cp.Problem(
        cp.Minimize(cp.sum(x_lp) + t_soc + z_exp + x_pow + y_pow),
        [x_lp[0] + x_lp[1] >= 1, cp.norm(x_soc, 2) <= t_soc,
         cp.exp(x_exp) <= z_exp, cp.geo_mean(cp.hstack([x_pow, y_pow])) >= 1],
    )).dual_problem
    assert any(isinstance(c, SOC) for c in dual.constraints)
    assert any(isinstance(c, ExpCone) for c in dual.constraints)

def test_constant_term_preserved():
    x = cp.Variable(2)
    primal = cp.Problem(cp.Minimize(cp.sum(x) + 5.0), [x >= 0, cp.sum(x) >= 1])
    primal.solve(solver=cp.SCS, eps=1e-8)
    result = dualize(primal)
    result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
    assert abs(primal.value - result.dual_problem.value) <= 1e-4


# -- Duality properties --

def test_weak_duality():
    x = cp.Variable(2)
    primal = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
    primal.solve(solver=cp.SCS, eps=1e-7)
    result = dualize(primal)
    result.dual_problem.solve(solver=cp.SCS, eps=1e-7)
    assert result.dual_problem.value <= primal.value + 1e-4


# -- Edge cases --

def test_unbounded_primal():
    x = cp.Variable()
    result = dualize(cp.Problem(cp.Minimize(x), [x <= 1]))
    result.dual_problem.solve(solver=cp.SCS, eps=1e-7)
    assert result.dual_problem.status in [cp.INFEASIBLE, cp.UNBOUNDED, cp.INFEASIBLE_INACCURATE]

def test_infeasible_primal():
    x = cp.Variable()
    result = dualize(cp.Problem(cp.Minimize(x), [x >= 1, x <= 0]))
    result.dual_problem.solve(solver=cp.SCS, eps=1e-7)
    assert result.dual_problem.status in [
        cp.INFEASIBLE, cp.UNBOUNDED, cp.INFEASIBLE_INACCURATE, cp.UNBOUNDED_INACCURATE,
    ]

def test_large_dimension():
    np.random.seed(42)
    n = 50
    A = np.random.randn(10, n)
    b = np.random.randn(10)
    c = np.random.randn(n)
    x = cp.Variable(n)
    result = dualize(cp.Problem(cp.Minimize(c @ x), [A @ x >= b, x >= 0]))
    assert result.dual_problem.is_dcp()
