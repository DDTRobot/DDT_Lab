# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NP3O algorithm (verbatim port from LocomotionWithNP3O).

Imports rewritten to relative paths; other behavior preserved (cost critic,
k_value Lagrangian schedule, BarlowTwins imitation loss, adaptive-KL LR).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .actor_critic import ActorCriticBarlowTwins
from .rollout_storage import RolloutStorageWithCost


class NP3O:
    actor_critic: ActorCriticBarlowTwins

    def __init__(
        self,
        actor_critic,
        k_value,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        cost_value_loss_coef=1.0,
        cost_viol_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        dagger_update_freq=20,
        priv_reg_coef_schedual=[0, 0, 0],
        **kwargs,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        self.imi_flag = hasattr(self.actor_critic, "imitation_learning_loss") and self.actor_critic.imi_flag
        print(f'Running with imitation: {"ON" if self.imi_flag else "OFF"}')
        self.imi_weight = 1

        self.transition = RolloutStorageWithCost.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_viol_loss_coef = cost_viol_loss_coef
        self.entropy_coef = entropy_coef

        self.k_value = k_value
        self.substeps = 1

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
        cost_shape,
        cost_d_values,
    ):
        self.storage = RolloutStorageWithCost(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            cost_shape,
            cost_d_values,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def set_imi_flag(self, flag):
        self.imi_flag = flag
        print(f'Imitation mode: {"ON" if self.imi_flag else "OFF"}')

    def set_imi_weight(self, value):
        self.imi_weight = value

    def act(self, obs, critic_obs, infos):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.cost_values = self.actor_critic.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, costs, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()
        self.transition.dones = dones

        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
            self.transition.costs += self.gamma * (self.transition.costs * time_outs)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, obs):
        last_cost_values = self.actor_critic.evaluate_cost(obs).detach()
        self.storage.compute_cost_returns(last_cost_values, self.gamma, self.lam)

    def _compute_importance_ratio(self, actions_log_prob_batch, old_actions_log_prob_batch):
        return torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))

    def compute_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch):
        ratio = self._compute_importance_ratio(actions_log_prob_batch, old_actions_log_prob_batch)
        advantages = torch.squeeze(advantages_batch)
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        return torch.max(surrogate, surrogate_clipped).mean()

    def compute_cost_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch):
        ratio = self._compute_importance_ratio(actions_log_prob_batch, old_actions_log_prob_batch)
        ratio = ratio.view(-1, 1)
        surrogate = cost_advantages_batch * ratio
        surrogate_clipped = cost_advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        return torch.max(surrogate, surrogate_clipped).mean(0)

    def compute_value_loss(self, target_values_batch, value_batch, returns_batch):
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            return torch.max(value_losses, value_losses_clipped).mean()
        return (returns_batch - value_batch).pow(2).mean()

    def update_k_value(self, iteration):
        self.k_value = torch.min(torch.ones_like(self.k_value), self.k_value * (1.0004**iteration))
        return self.k_value

    def compute_constraint_violation_loss(
        self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch, cost_violation_batch
    ):
        cost_surrogate_loss = self.compute_cost_surrogate_loss(
            actions_log_prob_batch=actions_log_prob_batch,
            old_actions_log_prob_batch=old_actions_log_prob_batch,
            cost_advantages_batch=cost_advantages_batch,
        )
        cost_violation_loss = cost_violation_batch.mean()
        combined_loss = cost_surrogate_loss + cost_violation_loss
        return torch.sum(self.k_value * F.relu(combined_loss))

    def _adaptive_lr(self, kl_mean):
        if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def update(self):
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_viol_loss = 0
        mean_surrogate_loss = 0
        mean_imitation_loss = 0
        obs_batch_max = -math.inf
        obs_batch_min = math.inf

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch_count, (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            target_cost_values_batch,
            cost_advantages_batch,
            cost_returns_batch,
            cost_violation_batch,
        ) in enumerate(generator, start=1):
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            cost_value_batch = self.actor_critic.evaluate_cost(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )

            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    self._adaptive_lr(kl_mean)

            surrogate_loss = self.compute_surrogate_loss(
                actions_log_prob_batch=actions_log_prob_batch,
                old_actions_log_prob_batch=old_actions_log_prob_batch,
                advantages_batch=advantages_batch,
            )
            viol_loss = self.compute_constraint_violation_loss(
                actions_log_prob_batch=actions_log_prob_batch,
                old_actions_log_prob_batch=old_actions_log_prob_batch,
                cost_advantages_batch=cost_advantages_batch,
                cost_violation_batch=cost_violation_batch,
            )
            value_loss = self.compute_value_loss(
                target_values_batch=target_values_batch, value_batch=value_batch, returns_batch=returns_batch
            )
            cost_value_loss = self.compute_value_loss(
                target_values_batch=target_cost_values_batch,
                value_batch=cost_value_batch,
                returns_batch=cost_returns_batch,
            )
            entropy_loss = -self.entropy_coef * entropy_batch.mean()

            main_loss = surrogate_loss + self.cost_viol_loss_coef * viol_loss
            combine_value_loss = self.cost_value_loss_coef * cost_value_loss + self.value_loss_coef * value_loss

            if self.imi_flag:
                imitation_loss = self.actor_critic.imitation_learning_loss(obs_batch, critic_obs_batch, self.imi_weight)
                loss = main_loss + combine_value_loss + entropy_loss + imitation_loss
            else:
                loss = main_loss + combine_value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_viol_loss += viol_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            obs_batch_max = max(obs_batch_max, obs_batch.max().item())
            obs_batch_min = min(obs_batch_min, obs_batch.min().item())

            if self.imi_flag:
                mean_imitation_loss += imitation_loss.item()

        num_updates = batch_count
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_viol_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_imitation_loss /= num_updates

        self.storage.clear()

        return (
            mean_value_loss,
            mean_cost_value_loss,
            mean_viol_loss,
            mean_surrogate_loss,
            mean_imitation_loss,
            obs_batch_min,
            obs_batch_max,
        )
