# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter that exposes ``isaaclab.envs.ManagerBasedRLEnv`` through the API
``OnConstraintPolicyRunner`` expects:

* ``get_observations()`` -> ``(B, history_len, n_proprio)`` (the policy ObsGroup)
* ``get_privileged_observations()`` -> ``(B, n_critic)`` (the critic ObsGroup,
  with base_lin_vel as the first 3 dims for the BT vel supervision head)
* ``step(actions)`` -> ``(policy_obs, critic_obs, rewards, costs, dones, infos)``

NP3O is constraint-aware. By default we expose ``num_costs=1`` with all-zero
``costs / k_value / d_value`` so the cost head and Lagrangian loss run but
contribute zero — i.e. NP3O degrades to PPO + BarlowTwins. Subclass and override
``_compute_costs`` and update ``cost_k_values`` / ``cost_d_values_tensor`` to
plug in real constraints.

Env requirements:

* ObservationsCfg.policy: ``history_length=N``, ``flatten_history_dim=False`` so
  the policy obs has shape ``(B, N, D)``.
* ObservationsCfg.critic.base_lin_vel must be the first term (3 dims).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from ddt_lab.managers import CostManager


class IsaacLabNP3OWrapper:
    """Bridges Isaac Lab's ManagerBasedRLEnv to the reference NP3O VecEnv API."""

    def __init__(self, env, num_costs: int = 1, device: str | None = None):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = device or str(self.unwrapped.device)

        # ---- discover dims from the env's observation manager
        obs_dict = self.unwrapped.observation_manager.compute()
        if "policy" not in obs_dict:
            raise KeyError("policy ObsGroup is required")
        if "critic" not in obs_dict:
            raise KeyError("critic ObsGroup is required for NP3O privileged obs")

        policy_obs = obs_dict["policy"]
        critic_obs = obs_dict["critic"]

        if policy_obs.dim() != 3:
            raise ValueError(
                f"policy obs must be 3D (B, T, D); got {tuple(policy_obs.shape)}. "
                "Set ObservationsCfg.policy.history_length=N + flatten_history_dim=False."
            )
        if critic_obs.dim() != 2:
            raise ValueError(f"critic obs must be 2D (B, D); got shape {tuple(critic_obs.shape)}")

        self.history_len = int(policy_obs.shape[1])
        self.n_proprio = int(policy_obs.shape[-1])

        # Each segment is its own ObsGroup so dims are inferred from shape, not
        # hard-coded.  ``priv`` contains privileged physical params; ``scanner``
        # contains height-scan data (None / missing on flat terrain).
        priv_obs = obs_dict.get("priv")  # optional: None if env has no PrivCfg
        scan_obs = obs_dict.get("scanner")  # optional: None on flat terrain

        self.n_priv_latent = int(priv_obs.shape[-1]) if priv_obs is not None and priv_obs.shape[-1] > 0 else 0
        self.n_scan = int(scan_obs.shape[-1]) if scan_obs is not None and scan_obs.shape[-1] > 0 else 0
        # Full critic dim = shared prop + priv + scan.  No tensor allocation needed.
        self.n_critic = int(critic_obs.shape[-1]) + self.n_priv_latent + self.n_scan

        self.num_envs = int(self.unwrapped.num_envs)
        self.num_actions = int(self.unwrapped.action_manager.action.shape[-1])
        self.max_episode_length = float(self.unwrapped.max_episode_length)

        # ---- cost setup ------------------------------------------------------
        # If env_cfg.costs is declared, instantiate a CostManager from it.
        # Otherwise use a 1-dim zero placeholder so NP3O's cost path runs but
        # contributes 0 (algorithm reduces to PPO+BarlowTwins).
        cost_cfg = getattr(self.unwrapped.cfg, "costs", None)
        if cost_cfg is not None:
            self.cost_manager = CostManager(cost_cfg, self.unwrapped, self.device)
            self.num_costs = self.cost_manager.num_costs
            self.cost_k_values = self.cost_manager.k_values
            self.cost_d_values_tensor = self.cost_manager.d_values_tensor
        else:
            self.cost_manager = None
            self.num_costs = max(num_costs, 1)
            self.cost_k_values = torch.zeros(1, self.num_costs, device=self.device)
            self.cost_d_values_tensor = torch.zeros(1, 1, self.num_costs, device=self.device)
            print(
                "[IsaacLabNP3OWrapper] no env_cfg.costs declared — running with zero-cost placeholder (NP3O ≡"
                " PPO+BarlowTwins)."
            )

        # ---- runner-facing shapes (kept as plain attributes; runner reads them)
        self.policy_obs_shape = (self.history_len, self.n_proprio)
        self.critic_obs_shape = (self.n_critic,)

        # cfg.env namespace (mirrors LeggedRobotCfg.env entries the runner expects)
        self.cfg = SimpleNamespace(
            env=SimpleNamespace(
                n_proprio=self.n_proprio,
                n_critic=self.n_critic,
                n_priv_latent=self.n_priv_latent,
                n_scan=self.n_scan,
                history_len=self.history_len,
            )
        )

    # ------------------------------------------------------------ properties
    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    # ----------------------------------------------------------------- API
    def _concat_critic_obs(self, obs_dict) -> torch.Tensor:
        """Concatenate critic + priv + scanner ObsGroups into one flat tensor."""
        parts = [obs_dict["critic"].to(self.device)]
        if self.n_priv_latent > 0 and "priv" in obs_dict:
            parts.append(obs_dict["priv"].to(self.device))
        if self.n_scan > 0 and "scanner" in obs_dict:
            parts.append(obs_dict["scanner"].to(self.device))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def get_observations(self) -> torch.Tensor:
        return self.unwrapped.observation_manager.compute()["policy"].to(self.device)

    def get_privileged_observations(self) -> torch.Tensor:
        return self._concat_critic_obs(self.unwrapped.observation_manager.compute())

    def reset(self, env_ids=None):
        obs_dict, _ = self.env.reset()
        return obs_dict["policy"].to(self.device)

    def step(self, actions: torch.Tensor):
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions.to(self.unwrapped.device))

        policy_obs = obs_dict["policy"].to(self.device)
        critic_obs = self._concat_critic_obs(obs_dict)
        rewards = rewards.to(self.device)
        dones = (terminated | truncated).to(self.device)

        costs = self._compute_costs(obs_dict, rewards, terminated, truncated, extras)

        # Mirror RewardManager: write per-episode cost averages into extras['log']
        # whenever an episode ends, so the runner picks them up as
        # ``Mean episode cost_<name>`` lines.
        if self.cost_manager is not None:
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                cost_log = self.cost_manager.log_episode(done_ids, self.unwrapped.max_episode_length_s)
                log_dict = extras.setdefault("log", {})
                log_dict.update(cost_log)

        infos = dict(extras)
        infos["time_outs"] = truncated.to(self.device)

        return policy_obs, critic_obs, rewards, costs, dones, infos

    def _compute_costs(self, obs_dict, rewards, terminated, truncated, extras) -> torch.Tensor:
        """Per-step costs. If a CostManager was built from ``env_cfg.costs``,
        delegate to it; otherwise return zeros (NP3O ≡ PPO+BarlowTwins)."""
        if self.cost_manager is not None:
            return self.cost_manager.compute().to(self.device)
        return torch.zeros(self.num_envs, self.num_costs, device=self.device)
