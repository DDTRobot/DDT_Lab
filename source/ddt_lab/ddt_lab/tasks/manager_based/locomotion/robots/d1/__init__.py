# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from .base_env_cfg import D1RoughEnvCfg  # noqa: F401 (used as entry_point below)
from .dreamwaq import agents as _dw_agents
from .dreamwaq import flat_env_cfg as _dw_flat
from .dreamwaq import rough_env_cfg as _dw_rough
from .np3o import agents as _np3o_agents
from .np3o import platform_env_cfg as _np3o_platform
from .np3o import flat_env_cfg as _np3o_flat
from .np3o import rough_env_cfg as _np3o_rough
from .rsl_rl import agents as _ppo_agents
from .rsl_rl import flat_env_cfg as _ppo_flat
from .rsl_rl import rough_env_cfg as _ppo_rough

##
# RSL-RL (stock PPO) — original task IDs, 2D policy obs.
# scripts/rsl_rl/train.py  |  logs/rsl_rl/
##

gym.register(
    id="DDT-Velocity-Flat-D1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _ppo_flat.D1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{_ppo_agents.__name__}.rsl_rl_ppo_cfg:D1FlatPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-D1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _ppo_flat.D1FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{_ppo_agents.__name__}.rsl_rl_ppo_cfg:D1FlatPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": D1RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{_ppo_agents.__name__}.rsl_rl_ppo_cfg:D1RoughPPORunnerCfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _ppo_rough.D1RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{_ppo_agents.__name__}.rsl_rl_ppo_cfg:D1RoughPPORunnerCfg",
    },
)

##
# NP3O (BarlowTwins-PPO) — 3D policy obs (history_length=10).
# scripts/np3o/train.py  |  logs/np3o/
##

gym.register(
    id="DDT-Velocity-Flat-D1-NP3O-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _np3o_flat.D1FlatNP3OEnvCfg,
        "np3o_cfg_entry_point": f"{_np3o_agents.__name__}.np3o_cfg:d1_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-D1-NP3O-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _np3o_flat.D1FlatNP3OEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{_np3o_agents.__name__}.np3o_cfg:d1_flat_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-NP3O-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _np3o_rough.D1RoughNP3OEnvCfg,
        "np3o_cfg_entry_point": f"{_np3o_agents.__name__}.np3o_cfg:d1_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-NP3O-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _np3o_rough.D1RoughNP3OEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{_np3o_agents.__name__}.np3o_cfg:d1_rough_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Platform-D1-NP3O-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _np3o_platform.D1PlatformNP3OEnvCfg,
        "np3o_cfg_entry_point": f"{_np3o_agents.__name__}.np3o_cfg:d1_platform_np3o_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Platform-D1-NP3O-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _np3o_platform.D1PlatformNP3OEnvCfg_PLAY,
        "np3o_cfg_entry_point": f"{_np3o_agents.__name__}.np3o_cfg:d1_platform_np3o_runner_cfg",
    },
)

##
# DreamWaQ (CeNet VAE + PPO) — 3D policy obs (history_length=5).
# scripts/dreamwaq/train.py  |  logs/dreamwaq/
##

gym.register(
    id="DDT-Velocity-Flat-D1-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _dw_flat.D1FlatDreamWaQEnvCfg,
        "dreamwaq_cfg_entry_point": f"{_dw_agents.__name__}.dreamwaq_cfg:d1_flat_dreamwaq_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Flat-D1-DreamWaQ-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _dw_flat.D1FlatDreamWaQEnvCfg_PLAY,
        "dreamwaq_cfg_entry_point": f"{_dw_agents.__name__}.dreamwaq_cfg:d1_flat_dreamwaq_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-DreamWaQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _dw_rough.D1RoughDreamWaQEnvCfg,
        "dreamwaq_cfg_entry_point": f"{_dw_agents.__name__}.dreamwaq_cfg:d1_rough_dreamwaq_runner_cfg",
    },
)

gym.register(
    id="DDT-Velocity-Rough-D1-DreamWaQ-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _dw_rough.D1RoughDreamWaQEnvCfg_PLAY,
        "dreamwaq_cfg_entry_point": f"{_dw_agents.__name__}.dreamwaq_cfg:d1_rough_dreamwaq_runner_cfg",
    },
)
