from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from cvxpy import settings as s
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver

from .utils import build_cone_constraints, extract_cone_values


@dataclass
class DualizationResult:
    dual_problem: cp.Problem
    _primal_problem: cp.Problem
    _y: cp.Variable
    _w: cp.Variable | None
    _K_dir: dict
    _eq_constraint: Any
    _inv_data: dict
    _chain: Chain
    _chain_inv_data: list[Any]

    def to_primal_solution(self) -> Solution:
        status = self.dual_problem.status
        opt_val = self.dual_problem.value if status in s.SOLUTION_PRESENT else np.nan
        primal_vars: dict = {}
        dual_vars: dict = {}

        if status in s.SOLUTION_PRESENT:
            y_val = self._y.value
            assert y_val is not None
            w_val = self._w.value if self._w is not None else None
            primal_vars = extract_cone_values(y_val, self._K_dir, w_val)
            dual_vars = {s.EQ_DUAL: self._eq_constraint.dual_value}

        dual_sol = Solution(status, opt_val, primal_vars, dual_vars, {})
        cone_sol = a2d.Dualize.invert(dual_sol, self._inv_data)
        return self._chain.invert(cone_sol, self._chain_inv_data)

    def unpack_primal(self) -> Solution:
        sol = self.to_primal_solution()
        self._primal_problem.unpack(sol)
        return sol


def dualize(problem: cp.Problem) -> DualizationResult:
    if not problem.is_dcp():
        raise ValueError("cvxdual requires a DCP problem.")
    if problem.is_mixed_integer():
        raise ValueError("cvxdual does not support mixed-integer problems.")

    has_quad = (
        problem.objective.expr.is_quadratic()
        and not problem.objective.expr.is_affine()
    )

    reductions: list[Any] = []
    if isinstance(problem.objective, cp.Maximize):
        reductions.append(FlipObjective())
    reductions.extend([
        Dcp2Cone(quad_obj=has_quad),
        CvxAttr2Constr(reduce_bounds=True),
        ConeMatrixStuffing(quad_obj=has_quad),
    ])

    chain = Chain(None, reductions)
    cone_prog, chain_inv_data = chain.apply(problem)
    cone_prog = ConicSolver.format_constraints(cone_prog, exp_cone_order=[0, 1, 2])
    data, inv_data = a2d.Dualize.apply(cone_prog)
    A, b, c = data[s.A], data[s.B], data[s.C]
    K_dir = data['K_dir']
    P = data.get(s.P)
    d = inv_data[s.OBJ_OFFSET]

    y = cp.Variable(A.shape[1], name="y")
    if P is not None:
        w = cp.Variable(P.shape[0], name="w")
        eq_constraint = A @ y - P @ w == b
        objective = cp.Maximize(c @ y - 0.5 * cp.quad_form(w, P) + d)
    else:
        w = None
        eq_constraint = A @ y == b
        objective = cp.Maximize(c @ y + d)

    cone_constraints = build_cone_constraints(y, K_dir)
    dual_problem = cp.Problem(objective, [eq_constraint] + cone_constraints)

    return DualizationResult(
        dual_problem=dual_problem,
        _primal_problem=problem,
        _y=y,
        _w=w,
        _K_dir=K_dir,
        _eq_constraint=eq_constraint,
        _inv_data=inv_data,
        _chain=chain,
        _chain_inv_data=chain_inv_data,
    )


def solve_dual(problem: cp.Problem, **solve_kwargs: Any) -> DualizationResult:
    result = dualize(problem)
    result.dual_problem.solve(**solve_kwargs)
    result.unpack_primal()
    return result
