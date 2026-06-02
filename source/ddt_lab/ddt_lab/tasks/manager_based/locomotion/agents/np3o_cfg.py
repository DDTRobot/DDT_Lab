# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default NP3O training cfg for the locomotion task family.

Numbers mirror ``LocomotionWithNP3O/configs/base/legged_robot_config.py``
(``LeggedRobotCfgPPO``). Robot- or task-specific overrides live next to the
env cfg, e.g. [robots/d1/agents/np3o_cfg.py] copies this base via
``base_np3o_runner_cfg()`` and tweaks ``experiment_name`` / ``max_iterations``
/ ``max_grad_norm`` / ``cost_*_coef`` / etc.

Fields irrelevant to ddt_lab's two-ObsGroup setup (``priv_encoder_dims``,
``scan_encoder_dims``, ``rnn_*``) are dropped.
"""

from __future__ import annotations

from copy import deepcopy

# fmt: off
_BASE_NP3O_RUNNER_CFG = {
    # Runner / training-loop config (LeggedRobotCfgPPO.runner)
    "runner": {
        "policy_class_name": "ActorCriticBarlowTwins",
        "algorithm_class_name": "NP3O",
        "runner_class_name": "OnConstraintPolicyRunner",
        "num_steps_per_env": 24,
        "max_iterations": 5000,
        "save_interval": 100,
        "experiment_name": "locomotion_np3o",
        "run_name": "",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": r"model_.*\.pt",
    },
    # PPO core + NP3O cost weights (LeggedRobotCfgPPO.algorithm)
    "algorithm": {
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "cost_value_loss_coef": 0.1,
        "cost_viol_loss_coef": 0.1,
    },
    # ActorCriticBarlowTwins (LeggedRobotCfgPPO.policy + reference's hard-coded BT sizes)
    "policy": {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "priv_encoder_dims": [],          # [] = Identity (D1FlatCfgPPO default)
        "scan_encoder_dims": [128, 64, 32],  # used when n_scan > 0 (rough tasks)
        "activation": "elu",
        "imi_flag": True,
        "bt_window": 5,
        "bt_mlp_encoder_dims": [128, 64],
        "bt_latent_encoder_dims": [32, 16],
        "bt_projector_dims": [64],
    },
}
# fmt: on


def base_np3o_runner_cfg() -> dict:
    """Return a deep-copied base NP3O runner cfg dict.

    Always copy — robot cfgs mutate ``experiment_name``/``max_iterations``/
    etc. in place, and we don't want those edits to leak across robots.
    """
    return deepcopy(_BASE_NP3O_RUNNER_CFG)
