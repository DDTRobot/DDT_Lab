# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab → DreamWaQ VecEnv adapter.

Exposes ``isaaclab.envs.ManagerBasedRLEnv`` through the API expected by
``OnPolicyRunner``:

* ``get_observations()``            → policy obs ``(B, history_len, n_proprio)``
* ``get_privileged_observations()`` → critic obs ``(B, n_critic)``
* ``step(actions)``                 → ``(policy_obs, critic_obs, rewards, dones, infos)``

No cost support — DreamWaQ is pure PPO + CeNet VAE.
"""

from __future__ import annotations

import torch
from types import SimpleNamespace


class IsaacLabDreamWaQWrapper:
    """Bridges Isaac Lab's ManagerBasedRLEnv to the DreamWaQ VecEnv API."""

    def __init__(self, env, device: str | None = None):
        self.env = env
        self.unwrapped = env.unwrapped
        self.device = device or str(self.unwrapped.device)

        # ---- discover dims from ObsGroups -----------------------------------
        obs_dict = self.unwrapped.observation_manager.compute()
        if "policy" not in obs_dict:
            raise KeyError("policy ObsGroup is required")
        if "critic" not in obs_dict:
            raise KeyError("critic ObsGroup is required for DreamWaQ privileged obs")

        policy_obs = obs_dict["policy"]
        critic_obs = obs_dict["critic"]

        if policy_obs.dim() != 3:
            raise ValueError(
                f"policy obs must be 3D (B, T, D); got {tuple(policy_obs.shape)}. "
                "Set ObservationsCfg.policy.history_length=N + flatten_history_dim=False."
            )
        if critic_obs.dim() != 2:
            raise ValueError(f"critic obs must be 2D (B, D); got {tuple(critic_obs.shape)}")

        self.history_len = int(policy_obs.shape[1])
        self.n_proprio = int(policy_obs.shape[-1])

        priv_obs = obs_dict["priv"]              # always present
        scan_obs = obs_dict.get("scanner")       # optional: None on flat terrain
        self.n_priv_latent = int(priv_obs.shape[-1]) if priv_obs is not None and priv_obs.shape[-1] > 0 else 0
        self.n_scan = int(scan_obs.shape[-1]) if scan_obs is not None and scan_obs.shape[-1] > 0 else 0
        self.n_critic = int(critic_obs.shape[-1]) + self.n_priv_latent + self.n_scan

        self.num_envs = int(self.unwrapped.num_envs)
        self.num_actions = int(self.unwrapped.action_manager.action.shape[-1])
        self.max_episode_length = float(self.unwrapped.max_episode_length)

        self.policy_obs_shape = (self.history_len, self.n_proprio)
        self.critic_obs_shape = (self.n_critic,)

        self.cfg = SimpleNamespace(
            env=SimpleNamespace(
                n_proprio=self.n_proprio,
                n_critic=self.n_critic,
                n_priv_latent=self.n_priv_latent,
                n_scan=self.n_scan,
                history_len=self.history_len,
            )
        )

    # ---- properties --------------------------------------------------------

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    # ---- API ---------------------------------------------------------------

    def _concat_critic_obs(self, obs_dict) -> torch.Tensor:
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
        """Returns ``(policy_obs, critic_obs, rewards, dones, infos)`` — no costs."""
        obs_dict, rewards, terminated, truncated, extras = self.env.step(
            actions.to(self.unwrapped.device)
        )
        policy_obs = obs_dict["policy"].to(self.device)
        critic_obs = self._concat_critic_obs(obs_dict)
        rewards = rewards.to(self.device)
        dones = (terminated | truncated).to(self.device)

        infos = dict(extras)
        infos["time_outs"] = truncated.to(self.device)

        return policy_obs, critic_obs, rewards, dones, infos
