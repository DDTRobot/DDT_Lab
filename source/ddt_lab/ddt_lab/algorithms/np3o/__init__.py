# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NP3O (BarlowTwins-PPO) algorithm package, ported from
``LocomotionWithNP3O`` and adapted to Isaac Lab's two-ObsGroup setup.

CostManager / CostTermCfg now live with the rest of the locomotion MDP at
``ddt_lab.tasks.manager_based.locomotion.mdp`` (re-exported via that package's
``__init__`` so ``mdp.CostTermCfg`` works inside env cfgs).
"""

from .actor_critic import ActorCriticBarlowTwins
from .np3o import NP3O
from .rollout_storage import RolloutStorageWithCost
from .runner import OnConstraintPolicyRunner
from .wrapper import IsaacLabNP3OWrapper

__all__ = [
    "ActorCriticBarlowTwins",
    "IsaacLabNP3OWrapper",
    "NP3O",
    "OnConstraintPolicyRunner",
    "RolloutStorageWithCost",
]
