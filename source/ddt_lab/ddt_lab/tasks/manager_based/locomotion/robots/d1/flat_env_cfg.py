# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1 flat-ground env, mirroring ``LocomotionWithNP3O/configs/d1/d1_flat_config.py``.

Reward weights and the cost set are configured in ``rough_env_cfg.py``; this
file only encodes flat-specific differences: terrain switched to a plane, the
critic's height_scan disabled, terrain curriculum disabled, looser reset poses,
and (for the play variant) zero commands + no domain randomisation.
"""

from isaaclab.utils import configclass

from .rough_env_cfg import D1RoughEnvCfg, D1RoughDreamWaQEnvCfg


@configclass
class D1FlatEnvCfg(D1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------------- terrain (plane, no height scan) ----------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # Remove the scanner ObsGroup entirely on flat terrain — no height_scanner.
        self.observations.scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # ---------------- commands (D1FlatCfg.commands.ranges) ----------------
        # Same as reference: ±1 m/s xy linear, ±1 rad/s yaw, full heading range.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)

        # ---------------- domain randomisation events (looser reset pose) ----------------
        # Matches D1FlatCfg's _reset_root_states wider initial pose distribution.
        # self.events.reset_base.params = {
        #     "pose_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (0.0, 0.2),
        #         "roll": (-0.785, 0.785),
        #         "pitch": (-1.57, 1.57),
        #         "yaw": (-3.14, 3.14),
        #     },
        #     "velocity_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (-0.5, 0.5),
        #         "roll": (-0.5, 0.5),
        #         "pitch": (-0.5, 0.5),
        #         "yaw": (-0.5, 0.5),
        #     },
        # }


@configclass
class D1FlatEnvCfg_PLAY(D1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # smaller scene for interactive playback
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable noise / external pushes / domain rand events for stable playback
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None

        # zero commands (matches D1FlatCfg_Play.commands.ranges)
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.heading = (0.0, 0.0)


##
# DreamWaQ flat variants — inherit from D1RoughDreamWaQEnvCfg and apply flat overrides.
##


@configclass
class D1FlatDreamWaQEnvCfg(D1RoughDreamWaQEnvCfg):
    """D1 flat (plane) env cfg for DreamWaQ.  Terrain overrides only."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.scanner = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        # self.disable_zero_weight_rewards()


@configclass
class D1FlatDreamWaQEnvCfg_PLAY(D1FlatDreamWaQEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
