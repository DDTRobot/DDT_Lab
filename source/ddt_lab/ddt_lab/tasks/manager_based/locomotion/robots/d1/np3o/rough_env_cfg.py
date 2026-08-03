# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
from ddt_lab.managers import CostTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..base_env_cfg import D1RoughEnvCfg, ObservationsCfg, RewardsCfg


@configclass
class PrivilegedObservationsCfg(ObservationsCfg):
    """NP3O observation cfg.

    Inherits ``PolicyCfg`` and ``policy`` from ``ObservationsCfg`` (identical).
    Overrides ``CriticCfg`` to remove ``height_scan`` (provided by ``ScannerCfg``).
    Adds ``PrivCfg`` and ``ScannerCfg``.
    """

    @configclass
    class CriticCfg(ObsGroup):
        """Critic obs — no height_scan (comes from ScannerCfg)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=(2.0, 2.0, 0.25),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

    @configclass
    class PrivCfg(ObsGroup):
        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        joint_kp_factor = ObsTerm(
            func=mdp.joint_kp_factor,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(0.0, 2.0),
            scale=1.0,
        )
        joint_kd_factor = ObsTerm(
            func=mdp.joint_kd_factor,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(0.0, 2.0),
            scale=1.0,
        )

    @configclass
    class ScannerCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

    # policy: PolicyCfg inherited from ObservationsCfg (identical)
    critic: CriticCfg = CriticCfg()
    priv: PrivCfg = PrivCfg()
    scanner: ScannerCfg | None = ScannerCfg()

@configclass
class RoughRewardsCfg(RewardsCfg):
    # -- wheel-legged gait shaping (core trio for lin_x rolling / lin_y+ang_z stepping)
    wheel_scrub_penalty = RewTerm(
        func=mdp.wheel_scrub_penalty,
        weight=-0.0,  # penalise lateral foot contact during lin_y/ang_z
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "command_name": "base_velocity",
            "command_threshold": 0.10,
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"]),
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.0,  # reward swing-foot lift height during lin_y / ang_z; penalise it otherwise
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "command_name": "base_velocity",
            "target_height": 0.02,
            "std": 0.04,
            "wheel_radius": 0.1,
            "command_threshold": 0.10,  # match hip_pos/wheel_scrub_penalty gating
            "max_air_time": 0.3,  # decay reward if a foot hovers past ~1 swing phase
            "min_contact": 2,  # never let more than 2 feet leave the ground at once
            "lift_penalty_scale": 200.0,  # penalise clearance above wheel_radius when cmd≈0 or pure lin_x
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot"]),
        },
    )
    gait_trot = RewTerm(
        func=mdp.GaitReward,
        weight=0.0,  # suggested: +1.0 ~ +5.0; trot = diagonal pairs synchronized
        params={
            "std": 0.1,
            "command_name": "base_velocity",
            "max_err": 0.5,
            "velocity_threshold": 0.2,
            "command_threshold": 0.1,
            "lateral_only": True,
            "penalize_forward": True,  # penalise trot during pure lin_x, reward during lin_y/ang_z
            "synced_feet_pair_names": [["FL_foot", "RR_foot"], ["FR_foot", "RL_foot"]],
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
        },
    )
    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.0,  # suggested: -0.1 ~ -0.5; penalise L/R asymmetry
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # bilateral (L-R) symmetry: FL↔FR, RL↔RR for thigh and calf
            "mirror_joints": [
                ["FR_(thigh|calf).*", "RL_(thigh|calf).*"],
                ["FL_(thigh|calf).*", "RR_(thigh|calf).*"],
            ],
        },
    )


@configclass
class CostsCfg:
    joint_pos_limit = CostTermCfg(
        func=mdp.joint_pos_limit,
        scale=10.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )
    joint_vel_limit = CostTermCfg(
        func=mdp.joint_vel_limit,
        scale=5.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_torque_limit = CostTermCfg(
        func=mdp.joint_torque_limit,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class D1RoughNP3OEnvCfg(D1RoughEnvCfg):
    observations: PrivilegedObservationsCfg = PrivilegedObservationsCfg()
    rewards: RoughRewardsCfg = RoughRewardsCfg()
    costs: CostsCfg = CostsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.history_length = 10
        self.observations.policy.flatten_history_dim = False
        # ---- rewards ----
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.hip_pos.weight = -10.0
        self.rewards.foot_clearance.weight = 0.0
        self.rewards.gait_trot.weight = 0.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.wheel_scrub_penalty.weight = -3.0
        self.rewards.action_rate_l2.weight = -0.01
        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        if self.__class__.__name__ == "D1RoughNP3OEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class D1RoughNP3OEnvCfg_PLAY(D1RoughNP3OEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        if self.__class__.__name__ == "D1RoughNP3OEnvCfg_PLAY":
            self.disable_zero_weight_rewards()