# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NP3O training configs for D1H.

Built on top of [agents/np3o_cfg.py](../../agents/np3o_cfg.py) base; only
fields that differ from ``LeggedRobotCfgPPO`` defaults are listed here.
"""

from __future__ import annotations

from ddt_lab.tasks.manager_based.locomotion.agents.np3o_cfg import base_np3o_runner_cfg


def d1h_flat_np3o_runner_cfg() -> dict:
    """D1H flat-ground NP3O config."""
    cfg = d1h_rough_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "d1h_flat"
    cfg["runner"]["max_iterations"] = 10000
    return cfg


def d1h_rough_np3o_runner_cfg() -> dict:
    """D1H rough-ground NP3O config."""
    cfg = base_np3o_runner_cfg()
    cfg["runner"]["experiment_name"] = "d1h_rough"
    cfg["runner"]["max_iterations"] = 10000
    return cfg
