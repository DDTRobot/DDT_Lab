# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NP3O (BarlowTwins-PPO) training configs for D1.

Built on top of [agents/np3o_cfg.py](../../agents/np3o_cfg.py) base; only
fields that differ from ``LeggedRobotCfgPPO`` defaults are listed here.
Numbers mirror ``LocomotionWithNP3O/configs/d1/d1_flat_config.py``
(``D1FlatCfgPPO`` overrides).
"""

from __future__ import annotations

from ddt_lab.tasks.manager_based.locomotion.agents.np3o_cfg import base_np3o_runner_cfg


def d1_rough_np3o_runner_cfg() -> dict:
    """D1 rough-ground NP3O config — same overrides as flat, longer run."""
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "d1_rough"
    cfg["runner"]["max_iterations"] = 20000
    return cfg


def d1_flat_np3o_runner_cfg() -> dict:
    """D1 flat-ground NP3O config — matches ``D1FlatCfgPPO``."""
    cfg = d1_rough_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "d1_flat"
    cfg["runner"]["max_iterations"] = 10000
    return cfg


def d1_platform_np3o_runner_cfg() -> dict:
    """D1 flat-ground NP3O config — matches ``D1FlatCfgPPO``."""
    cfg = d1_rough_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "d1_platform"
    cfg["runner"]["max_iterations"] = 20000
    return cfg
