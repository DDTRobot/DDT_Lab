# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ rollout storage.

Ported from ``Dreamwaq/rsl_rl/rsl_rl/storage/rollout_storage_dreamwaq.py``
and adapted to the ddt_lab two-ObsGroup convention:

* ``actor_obs``  — ``(B, history_len, n_proprio)``  (3-D; IS the obs_hist_buf)
* ``critic_obs`` — ``(B, n_critic)``               (2-D; includes vel GT in first dims)

The reference stored ``obs_hist_buf`` and ``estimation`` (vel GT) as flat
separate tensors. In ddt_lab we fold them into the existing obs tensors:

    obs_hist_buf  ≡  actor_obs                 (already 3-D history)
    estimation    ≡  critic_obs[:, :vel_dims]  (derived at update time, not stored)

``live_batch = 1 - dones`` is computed inside ``mini_batch_generator`` as in
the reference and passed to the algorithm so the VAE can mask terminal frames.
"""

import torch


class RolloutStorageDreamWaQ:
    """Rollout storage for DreamWaQ — no cost fields."""

    class Transition:
        def __init__(self):
            self.observations = None          # actor obs  (B, T_hist, D)
            self.critic_observations = None   # critic obs (B, n_critic)
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list,    # e.g. [history_len, n_proprio]
        critic_obs_shape: list,   # e.g. [n_critic]
        actions_shape: list,
        device: str = "cpu",
    ):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # Core tensors
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *actor_obs_shape, device=device
        )
        self.privileged_observations = torch.zeros(
            num_transitions_per_env, num_envs, *critic_obs_shape, device=device
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )

        # PPO fields
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)

        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.step = 0

    def add_transitions(self, transition: "RolloutStorageDreamWaQ.Transition"):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device)
                for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device)
                for i in range(len(hid_c))
            ]
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = (
                last_values if step == self.num_transitions_per_env - 1
                else self.values[step + 1]
            )
            not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + not_terminal * gamma * next_values - self.values[step]
            advantage = delta + not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Yield mini-batches.

        Yields (14 fields, matching the reference ``rollout_storage_dreamwaq.py``):
            obs_batch         (B, T_hist, D)
            critic_obs_batch  (B, n_critic)
            actions_batch
            target_values_batch
            advantages_batch
            returns_batch
            old_log_prob_batch
            old_mu_batch
            old_sigma_batch
            live_batch        (B, 1) = 1 - dones   ← used by VAE to mask terminals
            hid_states_batch  = (None, None)
            masks_batch       = None
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size,
                                 requires_grad=False, device=self.device)

        obs = self.observations.flatten(0, 1)
        critic_obs = self.privileged_observations.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                live = 1.0 - dones[idx].float()
                yield (
                    obs[idx],
                    critic_obs[idx],
                    actions[idx],
                    values[idx],
                    advantages[idx],
                    returns[idx],
                    log_prob[idx],
                    old_mu[idx],
                    old_sigma[idx],
                    live,           # live_batch: 1 for non-terminal, 0 for terminal
                    (None, None),   # hid_states (no RNN)
                    None,           # masks
                )
