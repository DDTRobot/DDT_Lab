# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos - asset.data.default_joint_pos
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    joint_pos_rel = joint_pos_rel[:, asset_cfg.joint_ids]
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


# ---------------------------------------------------------------------------
# Privileged observation terms (critic-only)
# Mirrors the reference ``LocomotionWithNP3O`` priv_latent subset that can be
# obtained from Isaac Lab's Articulation data without extra PhysX API calls.
# ---------------------------------------------------------------------------


def contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary foot-contact state, centred at zero: +0.5 in contact, -0.5 not.

    Shape: ``(num_envs, num_feet)`` — typically 4 for D1.
    Mirrors reference ``contact_filt.float() - 0.5``.
    """
    from isaaclab.sensors import ContactSensor

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (B, K, 3)
    in_contact = (net_forces.norm(dim=-1) > threshold).float()  # (B, K)
    return in_contact - 0.5


def joint_kp_factor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Randomised kp (joint stiffness) as a scale factor relative to default.

    Shape: ``(num_envs, num_joints)``.
    Mirrors reference ``kp_factor`` in priv_latent; values outside [0, 2]
    are clamped to avoid very large signals from near-zero defaults.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    kp = asset.data.joint_stiffness[:, asset_cfg.joint_ids]
    default_kp = asset.data.default_joint_stiffness[:, asset_cfg.joint_ids]
    return (kp / (default_kp.abs() + 1e-6)).clamp(0.0, 2.0)


def joint_kd_factor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Randomised kd (joint damping) as a scale factor relative to default.

    Shape: ``(num_envs, num_joints)``.
    Mirrors reference ``kd_factor`` in priv_latent.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    kd = asset.data.joint_damping[:, asset_cfg.joint_ids]
    default_kd = asset.data.default_joint_damping[:, asset_cfg.joint_ids]
    return (kd / (default_kd.abs() + 1e-6)).clamp(0.0, 2.0)
