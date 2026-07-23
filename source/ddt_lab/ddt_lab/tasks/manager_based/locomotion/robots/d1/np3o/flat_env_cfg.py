# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import D1RoughNP3OEnvCfg


@configclass
class D1FlatNP3OEnvCfg(D1RoughNP3OEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # self.scene.height_scanner = None
        self.observations.scanner = None
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.curriculum.terrain_levels = None
        # # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # ------------------------------Curriculums------------------------------
        
        # self.curriculum.command_levels_lin_vel = None
        # self.curriculum.command_levels_ang_vel = None
        # reward 
        self.rewards.hip_pos.weight = -20.0
        self.rewards.foot_clearance.weight = 0.5
        self.rewards.gait_trot.weight = 0.2
        self.rewards.joint_mirror = -1.0

@configclass
class D1FlatNP3OEnvCfg_PLAY(D1FlatNP3OEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 1e9
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.add_base_inertia = None
        self.events.add_base_com = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
        # # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # self.events.reset_base.params = {
        #     "pose_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (0.0, 0.2),
        #         "roll": (-0.0, 0.0),
        #         "pitch": (-0, 0),
        #         "yaw": (-3.14, 3.14),
        #     },
        #     "velocity_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (-0.5, 0.5),
        #         "roll": (-0.0, 0.0),
        #         "pitch": (-0.0, 0.0),
        #         "yaw": (-0.0, 0.0),
        #     },
        # }
        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None