# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments. All Tita tasks are wired to NP3O (BarlowTwins-PPO);
# stock PPO is no longer supported because the policy obs is now 3D.
##

gym.register(
    id="DDT-Velocity-Flat-Tita-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:tita_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-Tita-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TitaFlatEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:tita_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Tita-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TitaRoughEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:tita_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-Tita-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.TitaRoughEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:tita_rough_np3o_runner_cfg",
    },
)
