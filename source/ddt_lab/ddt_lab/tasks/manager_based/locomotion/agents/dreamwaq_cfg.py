# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Default DreamWaQ training cfg for the locomotion task family.

Mirrors the structure of ``np3o_cfg.py``: robot-specific overrides live next
to the env cfg (e.g. ``robots/d1/agents/dreamwaq_cfg.py``) and call
``base_dreamwaq_runner_cfg()`` to inherit these defaults.
"""

from __future__ import annotations

from copy import deepcopy

_BASE_DREAMWAQ_RUNNER_CFG = {
    "runner": {
        "policy_class_name": "ActorCriticDreamWaQ",
        "algorithm_class_name": "PPO_DreamWaQ",
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "max_iterations": 5000,
        "save_interval": 100,
        "experiment_name": "locomotion_dreamwaq",
        "run_name": "",
        "resume": False,
        "load_run": ".*",
        "load_checkpoint": r"model_.*\.pt",
    },
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
    },
    "policy": {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        # CeNet (VAE) architecture — matches DreamWaQ reference
        "num_vel_dims": 3,
        "num_latent_dims": 16,
        "cenet_encoder_dims": [128, 64],
        "cenet_decoder_dims": [128, 128],
        "kl_weight": 1.0,
        "activation": "elu",
    },
}


def base_dreamwaq_runner_cfg() -> dict:
    """Return a deep-copied base DreamWaQ runner cfg dict."""
    return deepcopy(_BASE_DREAMWAQ_RUNNER_CFG)
