# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command terms for the locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformHeightCommand(CommandTerm):
    """Sample a target base height."""

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"UniformHeightCommand: range={self.cfg.ranges.height}, resample={self.cfg.resampling_time_range}"

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _resample_command(self, env_ids: Sequence[int]):
        samples = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = samples.uniform_(*self.cfg.ranges.height)

    def _update_command(self):
        pass

    def _update_metrics(self):
        height = self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        self.metrics["height_error"] = torch.abs(self.height_command[:, 0] - height)


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformHeightCommand`."""

    class_type: type = UniformHeightCommand
    asset_name: str = MISSING

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
