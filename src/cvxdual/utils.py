from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.cone2cone import affine2direct as a2d

QUAD_SLACK = 'w'


def build_cone_constraints(y: cp.Variable, K_dir: dict) -> list[Any]:
    constraints: list[Any] = []
    i = K_dir[a2d.FREE]
    if K_dir[a2d.NONNEG]:
        dim = K_dir[a2d.NONNEG]
        constraints.append(y[i:i + dim] >= 0)
        i += dim
    for dim in K_dir[a2d.SOC]:
        constraints.append(SOC(y[i], y[i + 1:i + dim]))
        i += dim
    for n in K_dir[a2d.PSD]:
        sz = n * n
        Y = cp.reshape(y[i:i + sz], (n, n), order='F')
        constraints.append(Y >> 0)
        i += sz
    if K_dir[a2d.DUAL_EXP]:
        exp_len = 3 * K_dir[a2d.DUAL_EXP]
        y_de = cp.reshape(y[i:i + exp_len], (exp_len // 3, 3), order='C')
        constraints.append(ExpCone(-y_de[:, 1], -y_de[:, 0], np.exp(1) * y_de[:, 2]))
        i += exp_len
    if K_dir[a2d.DUAL_POW3D]:
        alpha = np.array(K_dir[a2d.DUAL_POW3D])
        y_dp = cp.reshape(y[i:i + len(alpha) * 3], (alpha.size, 3), order='C')
        constraints.append(PowCone3D(y_dp[:, 0] / alpha, y_dp[:, 1] / (1 - alpha), y_dp[:, 2], alpha))
    return constraints


def extract_cone_values(
    y_val: np.ndarray,
    K_dir: dict,
    w_val: np.ndarray | None = None,
) -> dict:
    primal_vars: dict = {}
    i = K_dir[a2d.FREE]
    primal_vars[a2d.FREE] = y_val[:i]
    if K_dir[a2d.NONNEG]:
        dim = K_dir[a2d.NONNEG]
        primal_vars[a2d.NONNEG] = y_val[i:i + dim]
        i += dim
    primal_vars[a2d.SOC] = []
    for dim in K_dir[a2d.SOC]:
        primal_vars[a2d.SOC].append(y_val[i:i + dim])
        i += dim
    primal_vars[a2d.PSD] = []
    for n in K_dir[a2d.PSD]:
        sz = n * n
        primal_vars[a2d.PSD].append(y_val[i:i + sz].reshape((n, n), order='F'))
        i += sz
    if K_dir[a2d.DUAL_EXP]:
        exp_len = 3 * K_dir[a2d.DUAL_EXP]
        primal_vars[a2d.DUAL_EXP] = y_val[i:i + exp_len]
        i += exp_len
    if K_dir[a2d.DUAL_POW3D]:
        pow_len = len(K_dir[a2d.DUAL_POW3D]) * 3
        primal_vars[a2d.DUAL_POW3D] = y_val[i:i + pow_len]
    if w_val is not None:
        primal_vars[QUAD_SLACK] = w_val
    return primal_vars
