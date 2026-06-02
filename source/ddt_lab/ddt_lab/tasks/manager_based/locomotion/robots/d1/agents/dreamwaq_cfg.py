# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ training configs for D1.

Built on top of ``locomotion/agents/dreamwaq_cfg.py`` base; only
``experiment_name`` and ``max_iterations`` differ between flat and rough.
"""

from __future__ import annotations

from ddt_lab.tasks.manager_based.locomotion.agents.dreamwaq_cfg import base_dreamwaq_runner_cfg


def d1_flat_dreamwaq_runner_cfg() -> dict:
    cfg = base_dreamwaq_runner_cfg()
    cfg["runner"]["experiment_name"] = "d1_flat_dreamwaq"
    cfg["runner"]["max_iterations"] = 10000
    return cfg


def d1_rough_dreamwaq_runner_cfg() -> dict:
    cfg = base_dreamwaq_runner_cfg()
    cfg["runner"]["experiment_name"] = "d1_rough_dreamwaq"
    cfg["runner"]["max_iterations"] = 20000
    return cfg
