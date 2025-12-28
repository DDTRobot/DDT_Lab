# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

# Import the rough terrain config as the base (it already contains the full observation, action, reward setup)
from .rough_env_cfg import D1RoughEnvCfg


@configclass
class D1FlatEnvCfg(D1RoughEnvCfg):
    """
    Flat terrain version of the D1 locomotion task.
    This configuration overrides only the parts necessary to switch to a flat plane
    while preserving the same proprioceptive observation structure as the rough version
    (history-stacked observations → 10 × ~33 → flattened 330-dimensional input).
    """

    def __post_init__(self):
        # Call parent post-init first (sets up everything from rough config)
        super().__post_init__()

        # ----------------------------- Terrain -----------------------------
        # Switch to a completely flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None  # No terrain generator needed

        # ----------------------------- Sensors -----------------------------
        # Remove height scanner (no terrain perception needed on flat ground)
        self.scene.height_scanner = None

        # ----------------------------- Observations -----------------------------
        # Disable height scan term for both policy and critic
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # IMPORTANT: Keep observation history stacking enabled.
        # The rough config already uses history for proprioceptive terms
        # (base_ang_vel, projected_gravity, commands, joint_pos, joint_vel, last_action).
        # By default Isaac Lab stacks the last N steps (history_length is set in the
        # observation manager, usually 10 in locomotion tasks). No change needed here.

        # Ensure corruption (noise) is still applied during training
        self.observations.policy.enable_corruption = True

        # ----------------------------- Curriculum -----------------------------
        # No terrain difficulty curriculum on flat ground
        self.curriculum.terrain_levels = None

        # Optional: you may want to keep domain randomization events (push, mass, etc.)
        # for robustness – they are already defined in D1RoughEnvCfg and remain active.

        # ----------------------------- Rewards -----------------------------
        # (Optional) You can tweak rewards here if flat terrain makes some terms too easy.
        # Example:
        # self.rewards.track_lin_vel_xy_exp.weight = 4.0
        # self.rewards.action_rate_l2.weight = -0.05
        # Leave as-is to match rough training as closely as possible.


@configclass
class D1FlatEnvCfg_PLAY(D1FlatEnvCfg):
    """
    Play configuration for the flat terrain environment.
    Reduces environment count and disables all randomizations for deterministic playback.
    """

    def __post_init__(self) -> None:
        # Call parent post-init
        super().__post_init__()

        # Smaller scene for visualization / playback
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable observation noise
        self.observations.policy.enable_corruption = False

        # Remove all domain randomization events for clean playback
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
        self.events.physics_material = None

        # Optional: fix commands to zero for standing test
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)