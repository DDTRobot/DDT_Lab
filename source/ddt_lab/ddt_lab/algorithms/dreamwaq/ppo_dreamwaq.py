# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ PPO algorithm with dual optimizer.

Faithful port of ``Dreamwaq/rsl_rl/rsl_rl/algorithms/ppo_dreamwaq.py``.

No cost constraints — DreamWaQ is pure PPO + CeNet VAE.

Dual optimizer:
  - ``self.optimizer``      — actor MLP + critic MLP + std
  - ``self.vae_optimizer``  — CeNet encoder + decoder only

Two sequential backward passes per mini-batch (same as reference):
  1. RL loss (surrogate + value + entropy)
  2. VAE loss (vel_estimation + reconstruction + KL)

``live_batch`` (1 = non-terminal, 0 = terminal) is provided by
``RolloutStorageDreamWaQ.mini_batch_generator`` and used to mask terminal
transitions in the VAE loss — same as the reference implementation.

Framework mapping (reference → ddt_lab)
---------------------------------------
``obs``              ``policy_obs[:, -1, :]``       current proprio frame
``obs_hist_buf``     ``obs_batch``                  (B, T_hist, D), flattened inside CeNet
``estimation``       ``critic_obs_batch[:, :3]``    base_lin_vel GT
``decode_target``    ``obs_batch[:, -1, :].detach()``
``live_batch``       yielded by RolloutStorageDreamWaQ.mini_batch_generator
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .actor_critic import ActorCriticDreamWaQ
from .rollout_storage import RolloutStorageDreamWaQ


class PPO_DreamWaQ:
    """DreamWaQ PPO with dual-optimizer VAE training."""

    actor_critic: ActorCriticDreamWaQ

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        vae_learning_rate=None,
        vae_kl_weight=1.0,
        **kwargs,
    ):
        if kwargs:
            print(f"[PPO_DreamWaQ] unexpected kwargs ignored: {list(kwargs.keys())}")

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.vae_kl_weight = vae_kl_weight

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage: RolloutStorageDreamWaQ = None
        self.transition = RolloutStorageDreamWaQ.Transition()

        # ---- Dual optimizer ---------------------------------------------------
        vae_ids = {id(p) for p in self.actor_critic.cenet.parameters()}
        rl_params = [p for p in self.actor_critic.parameters() if id(p) not in vae_ids]
        self.optimizer = optim.Adam(rl_params, lr=learning_rate)
        self.vae_optimizer = optim.Adam(
            self.actor_critic.cenet.parameters(),
            lr=vae_learning_rate if vae_learning_rate is not None else learning_rate,
        )

        # ---- PPO hyperparameters ---------------------------------------------
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    # =========================================================================
    # Storage
    # =========================================================================

    def init_storage(self, num_envs, num_transitions_per_env,
                     actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageDreamWaQ(
            num_envs, num_transitions_per_env,
            actor_obs_shape, critic_obs_shape,
            action_shape, device=self.device,
        )

    # =========================================================================
    # Rollout collection
    # =========================================================================

    def act(self, obs, critic_obs):
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            time_outs = infos["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * time_outs, 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    # =========================================================================
    # Update — dual optimizer: RL then VAE
    # =========================================================================

    def update(self):
        mean_value_loss = mean_surrogate_loss = 0.0
        mean_vae_loss = mean_vel_loss = mean_recon_loss = mean_kl_loss = 0.0
        obs_batch_max = -float("inf")
        obs_batch_min = float("inf")

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,           # (B, T_hist, D) — actor obs WITH full history
            critic_obs_batch,    # (B, n_critic)
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            live_batch,          # (B, 1) — 1 for non-terminal, 0 for terminal
            hid_states_batch,
            masks_batch,
        ) in generator:

            # ---- RL forward pass ----
            self.actor_critic.act(obs_batch, masks=masks_batch,
                                  hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # Adaptive LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2))
                        / (2.0 * sigma_batch.pow(2))
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            # Surrogate
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            adv = torch.squeeze(advantages_batch)
            surrogate_loss = torch.max(
                -adv * ratio,
                -adv * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
            ).mean()

            # Value loss
            if self.use_clipped_value_loss:
                v_clip = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (v_clip - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            rl_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # ---- RL backward (reference: optimizer.zero_grad; loss.backward; optimizer.step) ----
            self.optimizer.zero_grad()
            rl_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for pg in self.optimizer.param_groups for p in pg["params"]],
                self.max_grad_norm,
            )
            self.optimizer.step()

            # ---- VAE backward (reference: separate vae_optimizer) ----
            num_vel = self.actor_critic.num_vel_dims
            vel_target = critic_obs_batch[:, :num_vel].detach()
            decode_target = obs_batch[:, -1, :].detach()

            code, decode, (_, _, mean_lat, logvar_lat) = (
                self.actor_critic.cenet.forward(obs_batch)
            )
            code_vel = code[:, :num_vel]

            vel_loss = nn.MSELoss()(code_vel * live_batch, vel_target * live_batch)
            recon_loss = nn.MSELoss()(decode * live_batch, decode_target * live_batch)
            kl_loss = -0.5 * torch.mean(
                torch.sum(
                    (1 + logvar_lat - mean_lat.pow(2) - logvar_lat.exp()) * live_batch,
                    dim=-1,
                )
            )
            vae_loss = vel_loss + recon_loss + self.vae_kl_weight * kl_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.cenet.parameters(), self.max_grad_norm
            )
            self.vae_optimizer.step()

            mean_vae_loss += vae_loss.item()
            mean_vel_loss += vel_loss.item()
            mean_recon_loss += recon_loss.item()
            mean_kl_loss += kl_loss.item()

            mean_surrogate_loss += surrogate_loss.item()
            mean_value_loss += value_loss.item()
            obs_batch_max = max(obs_batch_max, obs_batch.max().item())
            obs_batch_min = min(obs_batch_min, obs_batch.min().item())

        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        return {
            "value_function": mean_value_loss / n,
            "surrogate": mean_surrogate_loss / n,
            "vae": mean_vae_loss / n,
            "vel_pred": mean_vel_loss / n,
            "recon": mean_recon_loss / n,
            "kl": mean_kl_loss / n,
            "learning_rate": self.learning_rate,
        }
