# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ddt_lab manager-style classes (parallel to ``isaaclab.managers``)."""

from .cost_manager import CostManager, CostTermCfg

__all__ = ["CostManager", "CostTermCfg"]
