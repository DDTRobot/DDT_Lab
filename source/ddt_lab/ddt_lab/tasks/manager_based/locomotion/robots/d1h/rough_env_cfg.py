# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math

import ddt_lab.tasks.manager_based.locomotion.mdp as mdp
import isaaclab.sim as sim_utils
from ddt_lab.managers import CostTermCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from ddt_lab.assets.ddt_robot import DDT_D1H_CFG  # isort: skip

##
# Scene definition
##


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = DDT_D1H_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


# from isaaclab import mdp
# from isaaclab.utils import configclass


@configclass
class D1HActionsCfg:

    # 左腿：髋、大腿、小腿
    fl_leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        scale={
            "FL_hip_joint": 0.25,
            "FL_thigh_joint": 0.25,
            "FL_calf_joint": 0.25,
        },
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )

    # 左足轮：速度控制
    fl_foot_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FL_foot_joint"],
        scale=5.0,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )

    # 右腿：髋、大腿、小腿
    fr_leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        scale={
            "FR_hip_joint": 0.25,
            "FR_thigh_joint": 0.25,
            "FR_calf_joint": 0.25,
        },
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )

    # 右足轮：速度控制
    fr_foot_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FR_foot_joint"],
        scale=5.0,
        clip={".*": (-100.0, 100.0)},
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=0.25
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0
        )
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
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
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

        def __post_init__(self):
            pass

    @configclass
    class PrivCfg(ObsGroup):
        """Privileged physical parameters for the critic."""

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
        """Height-scan input for the critic / scan encoder."""

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    priv: PrivCfg = PrivCfg()
    scanner: ScannerCfg = ScannerCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 2.75),
            "dynamic_friction_range": (0.2, 2.75),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_base_link"),
            "mass_distribution_params": (-0.5, 2.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    add_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_base_link"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_base_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.5, 1.0),
            "velocity_range": (-0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # General
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- root penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.10)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.01)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.5)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.5,
        params={
            "target_height": 0.45,
        },
    )

    # -- joint penalties
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_foot_joint"), "soft_ratio": 0.9},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"]),
        },
    )

    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FL_hip_joint", "FR_hip_joint"],
                ["FL_thigh_joint", "FR_thigh_joint"],
                ["FL_calf_joint", "FR_calf_joint"],
            ],
        },
    )

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"]),
            "stand_still_scale": 10,
            "velocity_threshold": 0.02,
            "command_threshold": 0.02,
        },
    )

    #     stand_still = RewTerm(
    #     func=mdp.stand_still,
    #     weight=-2.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "command_threshold": 0.06,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # -- action penalties
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

    # -- Contact sensor
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*_foot).*"]), "threshold": 1.0},
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"]),
            "threshold": 400.0,
        },
    )

    # -- optional penalties
    upward = RewTerm(
        func=mdp.upward,
        weight=1.1,
    )

    # -- pose regularisation
    default_joint_l2 = RewTerm(
        func=mdp.default_joint_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )

    hip_pos = RewTerm(
        func=mdp.default_joint_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"]),
        },
    )

    # 腾空时间奖励
    feet_air_time = RewTerm(
        func=mdp.reward_feet_air_time,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot"])},
    )

    # 头部碰撞惩罚
    collision_head = RewTerm(
        func=mdp.reward_collision_head,
        weight=-0.0,
    )

    #  大腿关节速度惩罚
    dof_thigh_vel = RewTerm(
        func=mdp.reward_dof_thigh_vel,
        weight=-0.0,
    )

    # 机身不前后偏移两脚中点
    body_pos_to_feet_x = RewTerm(
        func=mdp.reward_body_pos_to_feet_x,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.01},
    )

    # 禁止两腿前后错开
    body_feet_distance_x = RewTerm(
        func=mdp.reward_body_feet_distance_x,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.1},
    )

    # 固定两腿横向间距
    body_feet_distance_y = RewTerm(
        func=mdp.reward_body_feet_distance_y,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.1, "desired_feet_distance": 0.4},
    )

    # 左右腿横向对称
    body_symmetry_y = RewTerm(
        func=mdp.reward_body_symmetry_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.1},
    )

    # 左右腿离地高度一致
    body_symmetry_z = RewTerm(
        func=mdp.reward_body_symmetry_z,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.1},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Cost configuration (NP3O constrained training)
##


@configclass
class CostsCfg:
    """NP3O cost terms."""

    joint_pos_limit = CostTermCfg(
        func=mdp.joint_pos_limit,
        scale=1.0,
        d_value=0.0,
        k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*(hip|thigh|calf)_joint"])},
    )
    joint_vel_limit = CostTermCfg(
        func=mdp.joint_vel_limit,
        scale=1.0,
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


##
# Environment configuration
##


@configclass
class D1hRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the D1H locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: D1HActionsCfg = D1HActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    costs: CostsCfg = CostsCfg()

    # fmt: off
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
    ]
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint",
    ]
    # fmt: on

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # NP3O / BarlowTwins-PPO: policy obs must be 3D (B, T, D)
        self.observations.policy.history_length = 10
        self.observations.policy.flatten_history_dim = False

        # domain randomisation events (looser reset pose)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-0.0, 0.0),
                "pitch": (-0, 0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.disable_zero_weight_rewards()

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)


@configclass
class D1hRoughEnvCfg_PLAY(D1hRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
