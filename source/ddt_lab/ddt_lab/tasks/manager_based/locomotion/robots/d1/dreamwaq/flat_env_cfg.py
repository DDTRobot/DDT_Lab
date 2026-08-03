# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import D1RoughDreamWaQEnvCfg


@configclass
class D1FlatDreamWaQEnvCfg(D1RoughDreamWaQEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.scanner = None
        self.curriculum.terrain_levels = None
        if self.__class__.__name__ == "D1FlatDreamWaQEnvCfg":
            self.disable_zero_weight_rewards()


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
