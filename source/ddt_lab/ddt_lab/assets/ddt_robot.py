# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import (  # noqa: F401
    ActuatorNetMLPCfg,
    DCMotorCfg,
    DelayedPDActuatorCfg,
    ImplicitActuatorCfg,
)
from isaaclab.assets.articulation import ArticulationCfg

DDT_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/robots"))


DDT_TITA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{DDT_MODEL_DIR}/tita/urdf/robot.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.40),
        joint_pos={
            "joint_left_leg_1": 0.1,
            "joint_left_leg_2": 0.8,
            "joint_left_leg_3": -1.5,
            "joint_left_leg_4": 0.0,
            "joint_right_leg_1": 0.1,
            "joint_right_leg_2": 0.8,
            "joint_right_leg_3": -1.5,
            "joint_right_leg_4": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # "legs": DCMotorCfg(
        #     joint_names_expr=[".*"],
        #     effort_limit=60.0,
        #     saturation_effort=80.0,
        #     velocity_limit=30.0,
        #     stiffness=25.0,
        #     damping=0.5,
        #     friction=0.0,
        # ),
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=["^(?!.*_leg_4).*"],
            effort_limit=60.0,
            # saturation_effort=100.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*_leg_4"],
            effort_limit=20.0,
            # saturation_effort=10.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
    },
)


DDT_D1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{DDT_MODEL_DIR}/d1/urdf/robot.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.60),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # "legs": DCMotorCfg(
        #     joint_names_expr=[".*"],
        #     effort_limit=60.0,
        #     saturation_effort=80.0,
        #     velocity_limit=30.0,
        #     stiffness=25.0,
        #     damping=0.5,
        #     friction=0.0,
        # ),
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*(hip|thigh|calf)_joint"],
            effort_limit=60.0,
            # saturation_effort=100.0,
            velocity_limit=20.0,
            stiffness=60.0,
            damping=1.5,
            friction=0.0,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=20.0,
            # saturation_effort=10.0,
            velocity_limit=20.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
    },
)
