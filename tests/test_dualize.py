"""Core functional tests: variable inspection, QP, multi-cone, domains, API."""
import cvxpy as cp
import numpy as np
import pytest
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D

from cvxdual import DualVariable, DualizationResult, dualize, solve_dual


def _is_solved(status: str) -> bool:
    return status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


class TestDualVariableInspection:
    def test_names_follow_convention(self):
        x = cp.Variable(2)
        result = dualize(cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, cp.sum(x) == 1]))
        for dv in result.dual_variables:
            assert dv.name.startswith("dual_")

    def test_cone_types(self):
        x = cp.Variable(2)
        result = dualize(cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, cp.sum(x) == 1]))
        cone_types = {dv.cone_type for dv in result.dual_variables}
        assert "zero" in cone_types or "nonneg" in cone_types

    def test_variable_shape_matches_constraint(self):
        x = cp.Variable(3)
        result = dualize(cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, cp.sum(x) >= 1]))
        for dv in result.dual_variables:
            assert dv.variable.size == dv.constraint.size

    def test_A_and_b_block_shapes(self):
        x = cp.Variable(3)
        result = dualize(cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, cp.sum(x) >= 1]))
        for dv in result.dual_variables:
            if isinstance(dv.A_block, np.ndarray):
                assert dv.A_block.shape[0] == dv.variable.size
            if isinstance(dv.b_block, np.ndarray):
                assert dv.b_block.shape[0] == dv.variable.size

    def test_terms_are_expressions(self):
        x = cp.Variable(2)
        result = dualize(cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0]))
        for dv in result.dual_variables:
            assert isinstance(dv.objective_term, cp.Expression)
            assert isinstance(dv.constraint_term, cp.Expression)

class TestQuadraticObjective:
    def test_quad_obj_creates_slack(self):
        x = cp.Variable(2)
        result = dualize(cp.Problem(
            cp.Minimize(cp.quad_form(x, np.eye(2)) + cp.sum(x)), [x >= -1, cp.sum(x) >= 0],
        ))
        assert result.quad_slack is not None
        assert result.quad_slack.name() == "dual_quad_slack"

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

    def test_quad_obj_no_slack_for_linear(self):
        x = cp.Variable(2)
        assert dualize(cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])).quad_slack is None

    def test_quad_obj_with_equality_constraints(self):
        x = cp.Variable(3)
        primal = cp.Problem(cp.Minimize(cp.quad_form(x, np.eye(3))), [cp.sum(x) == 1, x >= 0])
        primal.solve(solver=cp.SCS, eps=1e-8)
        pval = primal.value
        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        assert abs(pval - result.dual_problem.value) < 1e-3


class TestDualizationResultAPI:
    def test_dual_for_constraint_returns_values(self):
        x = cp.Variable(2)
        c1, c2 = x >= 0, cp.sum(x) >= 1
        primal = cp.Problem(cp.Minimize(cp.sum(x)), [c1, c2])
        result = solve_dual(primal, solver=cp.SCS, eps=1e-8)
        assert result.dual_for_constraint(c1) is not None
        assert result.dual_for_constraint(c2) is not None

    def test_dual_for_nonexistent_constraint_returns_none(self):
        x = cp.Variable(2)
        result = dualize(cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0]))
        assert result.dual_for_constraint(x[0] >= 100) is None

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


class TestDualFeasibilityFromPrimal:
    """Solve the primal, dualize, verify dual solution satisfies all dual constraints."""

    def test_lp_primal_duals_feasible(self):
        x = cp.Variable(2)
        primal = cp.Problem(cp.Minimize(2 * x[0] + x[1]), [x >= 0, x[0] + x[1] >= 1])

        primal.solve(solver=cp.SCS, eps=1e-8)
        assert _is_solved(primal.status)
        pval = primal.value

        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        assert _is_solved(result.dual_problem.status)
        assert abs(pval - result.dual_problem.value) < 1e-4

        for dv in result.dual_variables:
            assert dv.variable.value is not None
            for con in dv.cone_constraints:
                assert np.max(con.violation()) < 1e-4, f"{dv.name}: cone violation {con.violation()}"
        assert np.max(result.dual_constraints.violation()) < 1e-4

        result.unpack_primal()
        for con in primal.constraints:
            assert con.dual_value is not None

    def test_socp_primal_duals_feasible(self):
        x = cp.Variable(3)
        t = cp.Variable()
        primal = cp.Problem(cp.Minimize(t), [cp.norm(x, 2) <= t, cp.sum(x) == 1])

        primal.solve(solver=cp.SCS, eps=1e-8)
        assert _is_solved(primal.status)
        pval = primal.value

        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-8)
        assert _is_solved(result.dual_problem.status)
        assert abs(pval - result.dual_problem.value) < 1e-4

        for dv in result.dual_variables:
            assert dv.variable.value is not None
            for con in dv.cone_constraints:
                assert np.max(con.violation()) < 1e-4, f"{dv.name}: cone violation {con.violation()}"
        assert np.max(result.dual_constraints.violation()) < 1e-4

    def test_psd_primal_duals_feasible(self):
        X = cp.Variable((2, 2), symmetric=True)
        primal = cp.Problem(cp.Minimize(cp.trace(X)), [X >> np.eye(2)])

        primal.solve(solver=cp.SCS, eps=1e-6)
        assert _is_solved(primal.status)
        pval = primal.value

        result = dualize(primal)
        result.dual_problem.solve(solver=cp.SCS, eps=1e-6)
        assert _is_solved(result.dual_problem.status)
        assert abs(pval - result.dual_problem.value) < 5e-3

        for dv in result.dual_variables:
            assert dv.variable.value is not None
            for con in dv.cone_constraints:
                assert np.max(con.violation()) < 1e-3, f"{dv.name}: cone violation {con.violation()}"
        assert np.max(result.dual_constraints.violation()) < 1e-3
