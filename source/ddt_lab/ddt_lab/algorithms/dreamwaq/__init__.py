# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ — self-contained RL algorithm package.

Unique to this package:
  - ``actor_critic.py``      — CeNet VAE + ActorCriticDreamWaQ
  - ``ppo_dreamwaq.py``      — PPO_DreamWaQ with dual optimizer (no cost)
  - ``rollout_storage.py``   — RolloutStorageDreamWaQ (no cost fields, live_batch)
  - ``runner.py``            — OnPolicyRunner for DreamWaQ

Shared utilities (imported from ``algorithms/utils``, no np3o dependency):
  - MLP factories, EmpiricalNormalization

"""

from .actor_critic import ActorCriticDreamWaQ, CeNet
from .ppo_dreamwaq import PPO_DreamWaQ
from .rollout_storage import RolloutStorageDreamWaQ
from .runner import OnPolicyRunner
from .wrapper import IsaacLabDreamWaQWrapper

__all__ = [
    "ActorCriticDreamWaQ",
    "CeNet",
    "IsaacLabDreamWaQWrapper",
    "OnPolicyRunner",
    "PPO_DreamWaQ",
    "RolloutStorageDreamWaQ",
]
