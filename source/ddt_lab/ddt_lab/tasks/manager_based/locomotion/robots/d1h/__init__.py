# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments. Each task is wired to NP3O (BarlowTwins-PPO).
##

gym.register(
    id="DDT-Velocity-Flat-D1H-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.D1hFlatEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1h_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-D1H-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.D1hFlatEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1h_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1H-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.D1hRoughEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1h_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1H-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.D1hRoughEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1h_rough_np3o_runner_cfg",
    },
)
