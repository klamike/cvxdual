from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cvxpy import settings as s
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.second_order import SOC
from cvxpy.problems.objective import Maximize
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver


@dataclass
class DualizationResult:
    primal_problem: cp.Problem
    dual_problem: cp.Problem
    dual_variable: cp.Variable
    dual_data: dict[str, Any]
    dual_cone_slices: dict[str, Any]
    _direct_primal_exprs: dict[str, Any]
    _dual_eq_constraint: cp.constraints.constraint.Constraint
    _dual_inv_data: dict[str, Any]
    _chain: Chain
    _chain_inv_data: list[Any]

    def to_primal_solution(self) -> Solution:
        obj_offset = float(self._dual_inv_data.get(s.OBJ_OFFSET, 0.0))
        raw_opt_val = self.dual_problem.value
        direct_opt_val = None if raw_opt_val is None else raw_opt_val - obj_offset
        direct_prims = _evaluate_direct_primal_exprs(self._direct_primal_exprs)
        direct_duals = {s.EQ_DUAL: self._dual_eq_constraint.dual_value}
        direct_sol = Solution(
            self.dual_problem.status,
            direct_opt_val,
            direct_prims,
            direct_duals,
            {},
        )
        cone_sol = a2d.Dualize.invert(direct_sol, self._dual_inv_data)
        return self._chain.invert(cone_sol, self._chain_inv_data)

    def unpack_primal(self) -> Solution:
        primal_solution = self.to_primal_solution()
        self.primal_problem.unpack(primal_solution)
        return primal_solution

    def cone_dimensions(self) -> dict[str, Any]:
        return dict(self.dual_data.get("K_dir", {}))

    def dual_for_constraint(self, constraint: cp.Constraint) -> Any:
        if constraint not in self.primal_problem.constraints:
            raise KeyError("Constraint does not belong to primal_problem.")
        return constraint.dual_value


def dualize(problem: cp.Problem, *, dual_var_name: str = "y") -> DualizationResult:
    if not problem.is_dcp():
        raise ValueError("cvxdual currently supports DCP problems only.")
    if problem.is_mixed_integer():
        raise ValueError("cvxdual does not support mixed-integer problems.")

    reductions: list[Any] = []
    if type(problem.objective) == Maximize:
        reductions.append(FlipObjective())
    reductions.extend([Dcp2Cone(), CvxAttr2Constr(), ConeMatrixStuffing()])
    chain = Chain(None, reductions)
    cone_prog, chain_inv_data = chain.apply(problem)

    cone_prog = ConicSolver.format_constraints(cone_prog, exp_cone_order=[0, 1, 2])
    data, dual_inv_data = a2d.Dualize.apply(cone_prog)
    k_dir = data["K_dir"]

    A = _to_dense_if_sparse(data[s.A])
    b = np.asarray(data[s.B]).reshape(-1)
    c = np.asarray(data[s.C]).reshape(-1)

    y = cp.Variable(shape=(A.shape[1],), name=dual_var_name)
    eq = A @ y == b
    constraints: list[Any] = [eq]
    direct_primal_exprs, cone_slices = _add_dual_cone_constraints(y, constraints, k_dir)

    obj_offset = float(dual_inv_data.get(s.OBJ_OFFSET, 0.0))
    dual_problem = cp.Problem(cp.Maximize(c @ y + obj_offset), constraints)
    return DualizationResult(
        primal_problem=problem,
        dual_problem=dual_problem,
        dual_variable=y,
        dual_data=data,
        dual_cone_slices=cone_slices,
        _direct_primal_exprs=direct_primal_exprs,
        _dual_eq_constraint=eq,
        _dual_inv_data=dual_inv_data,
        _chain=chain,
        _chain_inv_data=chain_inv_data,
    )


def solve_dual(problem: cp.Problem, **solve_kwargs: Any) -> DualizationResult:
    result = dualize(problem)
    result.dual_problem.solve(**solve_kwargs)
    result.unpack_primal()
    return result


def _evaluate_direct_primal_exprs(exprs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ([v.value for v in value] if isinstance(value, list) else (None if value is None else value.value))
        for key, value in exprs.items()
    }


def _to_dense_if_sparse(A: Any) -> Any:
    if sp.issparse(A):
        return A.toarray()
    return A


def _add_dual_cone_constraints(
    y: cp.Variable, constraints: list[Any], k_dir: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    direct_primal_exprs: dict[str, Any] = {a2d.FREE: None, a2d.SOC: [], a2d.PSD: []}
    cone_slices: dict[str, Any] = {a2d.SOC: [], a2d.PSD: []}

    i = int(k_dir[a2d.FREE])
    direct_primal_exprs[a2d.FREE] = y[:i]
    cone_slices[a2d.FREE] = slice(0, i)

    if k_dir[a2d.NONNEG]:
        dim = int(k_dir[a2d.NONNEG])
        direct_primal_exprs[a2d.NONNEG] = y[i : i + dim]
        cone_slices[a2d.NONNEG] = slice(i, i + dim)
        constraints.append(y[i : i + dim] >= 0)
        i += dim

    for dim in map(int, k_dir[a2d.SOC]):
        block = y[i : i + dim]
        direct_primal_exprs[a2d.SOC].append(block)
        cone_slices[a2d.SOC].append(slice(i, i + dim))
        constraints.append(SOC(block[0], block[1:]))
        i += dim

    for order in map(int, k_dir[a2d.PSD]):
        block_len = order * order
        block = y[i : i + block_len]
        block_mat = cp.reshape(block, (order, order), order="F")
        constraints.append(block_mat >> 0)
        direct_primal_exprs[a2d.PSD].append(block_mat)
        cone_slices[a2d.PSD].append(slice(i, i + block_len))
        i += block_len

    if k_dir[a2d.DUAL_EXP]:
        exp_len = 3 * int(k_dir[a2d.DUAL_EXP])
        exp_block = y[i : i + exp_len]
        direct_primal_exprs[a2d.DUAL_EXP] = exp_block
        cone_slices[a2d.DUAL_EXP] = slice(i, i + exp_len)
        y_de = cp.reshape(exp_block, (exp_len // 3, 3), order="C")
        constraints.append(ExpCone(-y_de[:, 1], -y_de[:, 0], np.exp(1) * y_de[:, 2]))
        i += exp_len

    if k_dir[a2d.DUAL_POW3D]:
        alpha = np.asarray(k_dir[a2d.DUAL_POW3D], dtype=float)
        pow_block = y[i:]
        direct_primal_exprs[a2d.DUAL_POW3D] = pow_block
        cone_slices[a2d.DUAL_POW3D] = slice(i, y.shape[0])
        y_dp = cp.reshape(pow_block, (alpha.size, 3), order="C")
        constraints.append(PowCone3D(y_dp[:, 0] / alpha, y_dp[:, 1] / (1 - alpha), y_dp[:, 2], alpha))

    return direct_primal_exprs, cone_slices
