# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NP3O env cfg variant that swaps the rough terrain for ``CUSTOM_TERRAINS_CFG``.

Kept separate from ``rough_env_cfg.py`` so the stock
``DDT-Velocity-Rough-D1-NP3O-v0`` task is untouched; this registers its own
task ID in ``robots/d1/__init__.py``.
"""
import math

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from ddt_lab.assets.terrains.platform import PLATFORM_TERRAINS_CFG
from isaaclab.utils import configclass

from ..base_env_cfg import D1RoughEnvCfg

# import dataclasses


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.15,  # 15% envs always get zero cmd → learns stand-still
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class D1PlatformNP3OEnvCfg(D1RoughEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Basic settings
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = PLATFORM_TERRAINS_CFG

        # On pit-like terrain cells (see UniformThresholdVelocityCommandCfg.pit_terrain_names),
        # restrict the velocity command to forward-only (lin_vel_x); lin_vel_y / ang_vel_z are
        # zeroed in UniformThresholdVelocityCommand._update_command.
        # base_velocity_fields = {
        #     f.name: getattr(self.commands.base_velocity, f.name)
        #     for f in dataclasses.fields(self.commands.base_velocity)
        #     if f.name != "class_type"
        # }
        # self.commands.base_velocity = UniformThresholdVelocityCommandCfg(**base_velocity_fields)

        if self.__class__.__name__ == "D1PlatformNP3OEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class D1PlatformNP3OEnvCfg_PLAY(D1PlatformNP3OEnvCfg):
    # Set to one of PLATFORM_TERRAINS_CFG.sub_terrains' keys (e.g. "pits", "rails", "boxes",
    # "hf_pyramid_slope", "hf_pyramid_slope_inv", "random_rough") to spawn only that terrain
    # type during play. Leave as None to keep the full terrain mix.

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.sub_terrains = {
                "pits": self.scene.terrain.terrain_generator.sub_terrains["pits"].replace(proportion=1.0)
            }
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        # # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (-0.0, 0.0)

        if self.__class__.__name__ == "D1PlatformNP3OEnvCfg_PLAY":
            self.disable_zero_weight_rewards()
