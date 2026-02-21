from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from cvxpy import settings as s

from cvxpy.constraints.psd import PSD
from cvxpy.constraints.zero import Zero
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.exponential import ExpCone

from cvxpy.reductions.chain import Chain
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing

_CONE_NAMES = {
    Zero: "zero", NonNeg: "nonneg", SOC: "soc",
    PSD: "psd", ExpCone: "exp", PowCone3D: "pow3d",
}

_STATUS_MAP = {  # FIXME
    s.INFEASIBLE: s.UNBOUNDED,
    s.INFEASIBLE_INACCURATE: s.UNBOUNDED_INACCURATE,
    s.UNBOUNDED: s.INFEASIBLE,
    s.UNBOUNDED_INACCURATE: s.INFEASIBLE_INACCURATE,
}


@dataclass
class DualVariable:
    name: str
    cone_type: str
    constraint: cp.Constraint
    variable: cp.Variable
    cone_constraints: list[cp.Cone] = field(default_factory=list)
    A_block: Any = None
    b_block: Any = None

    @property
    def objective_term(self) -> cp.Expression:
        return (-self.b_block) @ self.variable

    @property
    def constraint_term(self) -> cp.Expression:
        return self.A_block.T @ self.variable


@dataclass
class DualMeta:
    primal_problem: cp.Problem
    dual_variables: list[DualVariable]
    dual_constraints: cp.Constraint
    quad_slack: cp.Variable | None
    _chain: Chain
    _chain_inv_data: list[Any]
    _x_id: int = 0

    def cone_dimensions(self) -> dict[str, list[int]]:
        dims: dict[str, list[int]] = {}
        for v in self.dual_variables:
            dims.setdefault(v.cone_type, []).append(v.variable.size)
        return dims


@dataclass
class DualizationResult:
    dual_problem: cp.Problem
    meta: DualMeta

    @property
    def dual_variables(self) -> list[DualVariable]:
        return self.meta.dual_variables

    @property
    def dual_constraints(self) -> cp.Constraint:
        return self.meta.dual_constraints

    @property
    def quad_slack(self) -> cp.Variable | None:
        return self.meta.quad_slack

    def cone_dimensions(self) -> dict[str, list[int]]:
        return self.meta.cone_dimensions()

    def dual_for_constraint(self, constraint: cp.Constraint) -> Any | None:
        for con in self.meta.primal_problem.constraints:
            if con.id == constraint.id:
                return con.dual_value
        return None

    def to_primal_solution(self) -> Solution:
        dual_status = self.dual_problem.status
        primal_status = _STATUS_MAP.get(dual_status, dual_status)
        primal_vars = None
        dual_vars = None
        opt_val: Any = np.nan

        if dual_status in s.SOLUTION_PRESENT:
            opt_val = self.dual_problem.value
            if self.quad_slack is not None:
                x_stuffed = self.quad_slack.value
            else:
                x_stuffed = self.dual_constraints.dual_value
            primal_vars = {self.meta._x_id: x_stuffed}

            dual_vars = {}
            for dv in self.dual_variables:
                con = dv.constraint
                val = dv.variable.value
                if val is not None:
                    dual_vars[con.id] = val if val.size > 1 else val.item()
        elif dual_status in (s.INFEASIBLE, s.INFEASIBLE_INACCURATE):
            opt_val = -np.inf
        elif dual_status in (s.UNBOUNDED, s.UNBOUNDED_INACCURATE):
            opt_val = np.inf

        cone_sol = Solution(primal_status, opt_val, primal_vars, dual_vars, {})
        return self.meta._chain.invert(cone_sol, self.meta._chain_inv_data)

    def unpack_primal(self) -> Solution:
        sol = self.to_primal_solution()
        self.meta.primal_problem.unpack(sol)
        return sol

def _dual_cone(y: cp.Variable, constraint: Any) -> list[Any]:
    if isinstance(constraint, Zero):
        return []
    if isinstance(constraint, NonNeg):
        return [y >= 0]
    if isinstance(constraint, SOC):
        constraints, offset = [], 0
        for dim in constraint.cone_sizes():
            block = y[offset : offset + dim]
            constraints.append(SOC(block[0], block[1:]))
            offset += dim
        return constraints
    if isinstance(constraint, PSD):
        n = constraint.shape[0]
        return [constraint._dual_cone(cp.reshape(y, (n, n), order="F"))]
    if isinstance(constraint, (ExpCone, PowCone3D)):
        n_cones = constraint.num_cones() if isinstance(constraint, ExpCone) else y.size // 3
        yr = cp.reshape(y, (n_cones, 3), order="F")
        args = [yr[:, i] if n_cones > 1 else yr[0, i] for i in range(3)]
        return [constraint._dual_cone(*args)]
    raise ValueError(f"Unsupported constraint type: {type(constraint)}")


def dualize(problem: cp.Problem) -> DualizationResult:
    """Construct the conic dual of *problem*."""
    if not problem.is_dcp():
        raise ValueError("cvxdual requires a DCP problem.")
    if problem.is_mixed_integer():
        raise ValueError("cvxdual does not support mixed-integer problems.")

    has_quad = (
        problem.objective.expr.is_quadratic()
        and not problem.objective.expr.is_affine()
    )

    reductions: list[Reduction] = []
    if isinstance(problem.objective, cp.Maximize):
        reductions.append(FlipObjective())
    if has_quad:
        reductions.extend([Dcp2Cone(quad_obj=True), CvxAttr2Constr(reduce_bounds=True), ConeMatrixStuffing(quad_obj=True)])
    else:
        reductions.extend([Dcp2Cone(), CvxAttr2Constr(reduce_bounds=True), ConeMatrixStuffing()])

    chain = Chain(None, reductions)
    cone_prog, chain_inv_data = chain.apply(problem)

    if has_quad:
        P_matrix, c_vec, d_val, A_mat, b_vec = cone_prog.apply_parameters(quad_obj=True)
    else:
        c_vec, d_val, A_mat, b_vec = cone_prog.apply_parameters()
        P_matrix = None

    dvars: list[DualVariable] = []
    cone_type_counts: dict[str, int] = {}
    offset = 0
    for con in cone_prog.constraints:
        ct = _CONE_NAMES.get(type(con))
        if ct is None:
            raise ValueError(f"Unsupported constraint type: {type(con)}")
        idx = cone_type_counts.get(ct, 0)
        cone_type_counts[ct] = idx + 1
        A_i = A_mat[offset:offset + con.size, :]
        b_i = b_vec[offset:offset + con.size]
        name = f"dual_{ct}_{idx}"
        y = cp.Variable(con.size, name=name)
        dvars.append(DualVariable(
            name=name, cone_type=ct, constraint=con,
            variable=y, cone_constraints=_dual_cone(y, con),
            A_block=A_i, b_block=b_i,
        ))
        offset += con.size

    obj_terms = sum((v.objective_term for v in dvars), 0.0)
    feas_terms = sum((v.constraint_term for v in dvars), 0.0)

    if P_matrix is not None:
        w = cp.Variable(cone_prog.x.size, name="dual_quad_slack")
        quad_term = -0.5 * cp.quad_form(w, P_matrix)
        dual_constraints = feas_terms - P_matrix @ w == c_vec
        dual_objective = cp.Maximize(obj_terms + quad_term + d_val)
    else:
        w = None
        dual_constraints = feas_terms == c_vec
        dual_objective = cp.Maximize(obj_terms + d_val)

    cone_constraints = [c for v in dvars for c in v.cone_constraints]

    return DualizationResult(
        dual_problem=cp.Problem(
            dual_objective,
            [dual_constraints] + cone_constraints
        ),
        meta=DualMeta(
            primal_problem=problem,
            dual_variables=dvars,
            dual_constraints=dual_constraints,
            quad_slack=w,
            _chain=chain,
            _chain_inv_data=chain_inv_data,
            _x_id=cone_prog.x.id,
        ),
    )


def solve_dual(problem: cp.Problem, **solve_kwargs: Any) -> DualizationResult:
    result = dualize(problem)
    result.dual_problem.solve(**solve_kwargs)
    result.unpack_primal()
    return result
