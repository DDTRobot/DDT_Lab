# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""D1 NP3O flat-terrain task with a commanded base height."""

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .flat_env_cfg import D1FlatNP3OEnvCfg

HEIGHT_RANGE = (0.15, 0.50)


@configclass
class D1HeightFlatNP3OEnvCfg(D1FlatNP3OEnvCfg):
    """The existing D1 flat task plus a uniformly sampled base-height command."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_height = mdp.UniformHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 10.0),
            ranges=mdp.UniformHeightCommandCfg.Ranges(height=HEIGHT_RANGE),
        )
        self.observations.policy.height_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_height"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.height_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_height"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.rewards.base_height_l2 = None
        self.rewards.track_base_height_exp = RewTerm(
            func=mdp.track_base_height_exp,
            weight=2.0,
            params={"command_name": "base_height", "std": 0.1},
        )


@configclass
class D1HeightFlatNP3OEnvCfg_PLAY(D1HeightFlatNP3OEnvCfg):
    def __post_init__(self):
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
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
