# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments. Each task is wired to NP3O (BarlowTwins-PPO).
# Stock PPO is no longer supported on these tasks because the policy obs is now
# 3D (history_length=10, flatten_history_dim=False) — the upstream
# rsl_rl ``ActorCritic`` cannot consume that shape.
##

gym.register(
    id="DDT-Velocity-Flat-D1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.D1FlatEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-D1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.D1FlatEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.D1RoughEnvCfg,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.D1RoughEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{agents.__name__}.np3o_cfg:d1_rough_np3o_runner_cfg",
    },
)

##
# DreamWaQ variants — CeNet VAE actor, history_length=5 (matches reference num_obs_hist=5).
##

gym.register(
    id="DDT-Velocity-Flat-D1-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.D1FlatDreamWaQEnvCfg,
        "dreamwaq_cfg_entry_point": f"{agents.__name__}.dreamwaq_cfg:d1_flat_dreamwaq_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-D1-DreamWaQ-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.D1FlatDreamWaQEnvCfg_PLAY,
        "dreamwaq_cfg_entry_point": f"{agents.__name__}.dreamwaq_cfg:d1_flat_dreamwaq_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.D1RoughDreamWaQEnvCfg,
        "dreamwaq_cfg_entry_point": f"{agents.__name__}.dreamwaq_cfg:d1_rough_dreamwaq_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-DreamWaQ-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.D1RoughDreamWaQEnvCfg_PLAY,
        "dreamwaq_cfg_entry_point": f"{agents.__name__}.dreamwaq_cfg:d1_rough_dreamwaq_runner_cfg",
    },
)
