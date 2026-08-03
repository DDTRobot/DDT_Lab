# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1H flat-ground env config."""

from isaaclab.utils import configclass

from .rough_env_cfg import D1hRoughEnvCfg


@configclass
class D1hFlatEnvCfg(D1hRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---------------- terrain (plane, no height scan) ----------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # Remove the scanner ObsGroup entirely on flat terrain
        self.observations.scanner = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # ---------------- commands ----------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-3.14, 3.14)


@configclass
class D1hFlatEnvCfg_PLAY(D1hFlatEnvCfg):
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
