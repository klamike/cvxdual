"""Core functional tests: QP, multi-cone, domains, API."""
import cvxpy as cp
import numpy as np
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D

from cvxdual import DualizationResult, dualize, solve_dual


class TestQuadraticObjective:
    def test_quad_obj_dual_is_dcp(self):
        x = cp.Variable(2)
        result = dualize(cp.Problem(cp.Minimize(cp.quad_form(x, np.eye(2)) + cp.sum(x)), [x >= -1]))
        assert result.dual_problem.is_dcp()

    def test_quad_obj_strong_duality(self):
        x = cp.Variable(2)
        P = 2.0 * np.eye(2)
        c = np.array([1.0, -1.0])
        primal = cp.Problem(cp.Minimize(cp.quad_form(x, P) + c @ x), [x >= 0, cp.sum(x) <= 5])
        primal.solve(solver=cp.SCS, eps=1e-8)
        pval = primal.value
        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        assert abs(pval - result.dual_problem.value) < 1e-3

    def test_quad_obj_solution_recovery(self):
        x = cp.Variable(2)
        primal = cp.Problem(
            cp.Minimize(cp.quad_form(x, np.eye(2)) + np.array([1.0, 2.0]) @ x), [x >= 0],
        )
        primal.solve(solver=cp.SCS, eps=1e-8)
        x_ref = x.value.copy()
        x.value = None
        solve_dual(primal, solver=cp.SCS, eps=1e-8)
        assert np.allclose(x.value, x_ref, atol=1e-3)

    def test_quad_obj_with_equality_constraints(self):
        x = cp.Variable(3)
        primal = cp.Problem(cp.Minimize(cp.quad_form(x, np.eye(3))), [cp.sum(x) == 1, x >= 0])
        primal.solve(solver=cp.SCS, eps=1e-8)
        pval = primal.value
        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        assert abs(pval - result.dual_problem.value) < 1e-3


class TestDualizationResultAPI:
    def test_objective_offset_preserved(self):
        x = cp.Variable(2)
        primal = cp.Problem(cp.Minimize(cp.sum(x) + 42.0), [x >= 0, cp.sum(x) >= 1])
        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        primal.solve(solver=cp.SCS, eps=1e-8)
        assert abs(primal.value - result.dual_problem.value) < 1e-4

    def test_constraint_duals_populated_after_unpack(self):
        x = cp.Variable(2)
        primal = cp.Problem(cp.Minimize(x[0] + 2 * x[1]), [x >= 0, x[0] + x[1] >= 1])
        solve_dual(primal, solver=cp.SCS, eps=1e-8)
        assert primal.constraints[0].dual_value is not None
        assert primal.constraints[1].dual_value is not None


class TestMultiConeCorrectness:
    def test_exp_strong_duality(self):
        x, y, z = cp.Variable(2), cp.Variable(2), cp.Variable(2)
        primal = cp.Problem(cp.Minimize(cp.sum(z)), [ExpCone(x, y, z), y == 1, x >= 0])
        primal.solve(solver=cp.SCS, eps=1e-8)
        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        assert abs(primal.value - result.dual_problem.value) < 1e-3

    def test_exp_solution_recovery(self):
        x, y, z = cp.Variable(3), cp.Variable(3), cp.Variable(3)
        primal = cp.Problem(cp.Minimize(cp.sum(z)), [ExpCone(x, y, z), y == 1, x >= 0])
        primal.solve(solver=cp.SCS, eps=1e-8)
        z_ref = z.value.copy()
        z.value = None
        solve_dual(primal, solver=cp.SCS, eps=1e-8)
        assert np.allclose(z.value, z_ref, atol=1e-3)

    def test_pow3d_strong_duality(self):
        a = cp.Variable(2, nonneg=True)
        b = cp.Variable(2, nonneg=True)
        c = cp.Variable(2)
        alpha = np.array([0.3, 0.7])
        primal = cp.Problem(cp.Minimize(cp.sum(a) + cp.sum(b)), [PowCone3D(a, b, c, alpha), c == 1])
        primal.solve(solver=cp.SCS, eps=1e-8)
        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        assert abs(primal.value - result.dual_problem.value) < 1e-3

    def test_entropy_strong_duality(self):
        x = cp.Variable(3, pos=True)
        primal = cp.Problem(cp.Maximize(cp.sum(cp.entr(x))), [cp.sum(x) == 1])
        primal.solve(solver=cp.SCS, eps=1e-8)
        pval = primal.value
        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        result.unpack_primal()
        assert abs(primal.value - pval) < 1e-3

    def test_geo_mean_strong_duality(self):
        x = cp.Variable(3, pos=True)
        primal = cp.Problem(cp.Minimize(cp.sum(x)), [cp.geo_mean(x) >= 1])
        primal.solve(solver=cp.SCS, eps=1e-8)
        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        assert abs(primal.value - result.dual_problem.value) < 1e-3


class TestVariableDomains:
    def test_nonneg_binding(self):
        x = cp.Variable(2, nonneg=True)
        primal = cp.Problem(cp.Minimize(x[0] - 2 * x[1]), [x[0] + x[1] <= 3])
        primal.solve(solver=cp.SCS, eps=1e-8)
        pval, x_ref = primal.value, x.value.copy()
        x.value = None
        result = solve_dual(primal, solver=cp.SCS, eps=1e-8)
        assert abs(pval - result.dual_problem.value) < 1e-3
        assert np.allclose(x.value, x_ref, atol=1e-3)

    def test_nonpos_binding(self):
        x = cp.Variable(2, nonpos=True)
        primal = cp.Problem(cp.Minimize(-x[0] + 2 * x[1]), [x[0] + x[1] >= -3])
        primal.solve(solver=cp.SCS, eps=1e-8)
        pval, x_ref = primal.value, x.value.copy()
        x.value = None
        result = solve_dual(primal, solver=cp.SCS, eps=1e-8)
        assert abs(pval - result.dual_problem.value) < 1e-3
        assert np.allclose(x.value, x_ref, atol=1e-3)

    def test_pos_with_log(self):
        x = cp.Variable(2, pos=True)
        primal = cp.Problem(cp.Minimize(-cp.sum(cp.log(x))), [cp.sum(x) <= 2])
        primal.solve(solver=cp.SCS, eps=1e-8)
        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        assert abs(primal.value - result.dual_problem.value) < 1e-3

    def test_bounds_attribute(self):
        x = cp.Variable(2, bounds=[0, 5])
        primal = cp.Problem(cp.Minimize(x[0] - 2 * x[1]), [x[0] + x[1] <= 8])
        primal.solve(solver=cp.SCS, eps=1e-8)
        pval, x_ref = primal.value, x.value.copy()
        x.value = None
        solve_dual(primal, solver=cp.SCS, eps=1e-8)
        assert abs(pval - primal.value) < 1e-3
        assert np.allclose(x.value, x_ref, atol=1e-3)

    def test_nonneg_matches_explicit(self):
        c = np.array([1.0, -2.0])
        x1 = cp.Variable(2, nonneg=True)
        p1 = cp.Problem(cp.Minimize(c @ x1), [cp.sum(x1) <= 3])
        solve_dual(p1, solver=cp.SCS, eps=1e-8)
        x2 = cp.Variable(2)
        p2 = cp.Problem(cp.Minimize(c @ x2), [x2 >= 0, cp.sum(x2) <= 3])
        solve_dual(p2, solver=cp.SCS, eps=1e-8)
        assert abs(p1.value - p2.value) < 1e-4
        assert np.allclose(x1.value, x2.value, atol=1e-3)
