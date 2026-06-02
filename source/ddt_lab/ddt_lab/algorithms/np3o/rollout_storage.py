# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storages for NP3O.

Verbatim port of ``RolloutStorageWithCost`` from ``LocomotionWithNP3O/runner/rollout_storage.py``.
The ``utils.split_and_pad_trajectories`` import is rewired to the vendored
``rsl_rl.utils`` (same implementation upstream).
"""

import torch

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorageWithCost:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.costs = None
            self.dones = None
            self.values = None
            self.cost_values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, cost_shape, cost_d_values, device='cpu'):
        self.device = device
        self.obs_shape = obs_shape
        self.cost_shape = cost_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.costs = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.cost_values = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.cost_returns = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.cost_advantages = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        self.cost_violation = torch.zeros(num_transitions_per_env, num_envs, *cost_shape, device=self.device)
        self.cost_d_values = cost_d_values

        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None
        self.step = 0

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError('Rollout buffer overflow')
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.costs[self.step].copy_(transition.costs)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.cost_values[self.step].copy_(transition.cost_values)
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
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def compute_cost_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.cost_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.costs[step] + next_is_not_terminal * gamma * next_values - self.cost_values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.cost_returns[step] = advantage + self.cost_values[step]

        self.cost_advantages = self.cost_returns - self.cost_values
        cost_adv_mean = self.cost_advantages.view(self.num_envs * self.num_transitions_per_env, -1).mean(0)
        cost_adv_std = self.cost_advantages.view(self.num_envs * self.num_transitions_per_env, -1).std(0)
        self.cost_advantages = (self.cost_advantages - cost_adv_mean.view(1, 1, -1)) / (cost_adv_std.view(1, 1, -1) + 1e-8)
        self.cost_violation = ((1.0 - gamma) * (self.cost_returns - self.cost_d_values) + cost_adv_mean.view(1, 1, -1)) / (cost_adv_std.view(1, 1, -1) + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        cost_values = self.cost_values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        cost_returns = self.cost_returns.flatten(0, 1)
        cost_violation = self.cost_violation.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                    cost_values[batch_idx],
                    cost_advantages[batch_idx],
                    cost_returns[batch_idx],
                    cost_violation[batch_idx],
                )

    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                cost_returns_batch = self.cost_returns[:, start:stop]
                cost_violation_batch = self.cost_violation[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                cost_advantage_batch = self.cost_advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                cost_values_batch = self.cost_values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [s.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous() for s in self.saved_hidden_states_a]
                hid_c_batch = [s.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous() for s in self.saved_hidden_states_c]
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_a_batch

                yield (
                    obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch,
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch,
                    cost_values_batch, cost_advantage_batch, cost_returns_batch, cost_violation_batch,
                )
                first_traj = last_traj
