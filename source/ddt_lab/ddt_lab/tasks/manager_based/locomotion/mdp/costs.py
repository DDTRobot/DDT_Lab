# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost functions for NP3O constrained training.

Each function follows the Isaac Lab reward-function signature
``func(env, **params) -> torch.Tensor`` of shape ``(num_envs,)``.

Translated from ``LocomotionWithNP3O/configs/base/legged_robot.py`` (the
``_cost_*`` methods). All functions return non-negative values: NP3O's
``compute_constraint_violation_loss`` only penalises positive cost violations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum of joint-position excursions past the soft limits, per env.

    Mirrors the reference ``_cost_pos_limit``:
    ``out = (-clip(q - q_lo, max=0)) + clip(q - q_hi, min=0)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    soft_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    out_low = -(q - soft_limits[..., 0]).clamp(max=0.0)
    out_high = (q - soft_limits[..., 1]).clamp(min=0.0)
    return torch.sum(out_low + out_high, dim=1)


def joint_torque_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.9,
) -> torch.Tensor:
    """Sum of |τ| above the soft torque limit (``ratio * effort_limit``)."""
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    limit = asset.data.joint_effort_limits[:, asset_cfg.joint_ids] * soft_ratio
    return torch.sum((tau.abs() - limit).clamp(min=0.0), dim=1)


def joint_vel_limit(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    soft_ratio: float = 0.9,
) -> torch.Tensor:
    """Sum of |q̇| above the soft velocity limit, capped at 1 per joint."""
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    limit = asset.data.joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    return torch.sum((qd.abs() - limit).clamp(min=0.0, max=1.0), dim=1)


def hip_pos_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum of squared hip-joint deviation from the default pose.

    Cost-side counterpart of the reward ``hip_pos`` — reference NP3O's
    ``_cost_hip_pos`` always penalises hip drift (no lateral-command gating),
    while the reward-side ``hip_pos`` only penalises when lin_vel_y ≈ 0.

    The function name is ``hip_pos_l2`` (rather than ``hip_pos``) to avoid
    shadowing the reward-side ``mdp.hip_pos`` symbol when both modules
    re-export through ``mdp/__init__.py``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(q - q_default), dim=1)
