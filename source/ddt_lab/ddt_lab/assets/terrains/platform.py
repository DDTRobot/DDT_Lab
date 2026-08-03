# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom terrain generator configs for the D1 robot.

Add new ``TerrainGeneratorCfg`` presets here and import them from a robot/
algorithm ``*_env_cfg.py`` to swap ``SceneCfg.terrain.terrain_generator``.
See ``isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG`` for the stock preset
that ``base_env_cfg.py`` uses by default.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# TODO: tune `proportion` (must sum to 1.0 across all sub-terrains), and the
# per-terrain parameters below to taste.
PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.1,
        #     gap_width_range=(0.05, 0.23),
        #     platform_width=3.0,
        # ),
        "boxes": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.1,
            box_height_range=(0.05, 0.7),
            platform_width=2.0,
            double_box=True,
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.2,
            rail_thickness_range=(0.05, 0.10),
            rail_height_range=(0.7, 0.05),
            platform_width=3.0,
        ),
        "pits": terrain_gen.MeshPitTerrainCfg(
            proportion=0.2,
            pit_depth_range=(0.05, 0.7),
            platform_width=3.0,
            double_pit=True,
        ),
    },
)
