# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ actor-critic for ddt_lab, ported from the DreamWaQ wheel-legged gym.

Architecture:

* **CeNet (Context Estimation Network)** — a VAE that takes the proprioceptive
  history buffer and produces two latent codes:
    - ``code_vel``    (``num_vel_dims`` = 3) : explicitly supervised to match
      the ground-truth base linear velocity (taken from ``critic_obs[:, :3]``).
    - ``code_latent`` (``num_latent_dims`` = 16): implicit code regularised with
      a KL divergence term; encodes terrain, mass, friction, etc.

* **Actor MLP** — receives ``cat(code.detach(), current_obs)`` so that RL
  gradients cannot reach the CeNet encoder (same detach trick as NP3O).

* **Critic / Cost** — identical to ``ActorCriticBarlowTwins``; shares the same
  ``_get_critic_backbone_input`` normalises critic_obs directly (no pre-encoder).

The ``imitation_learning_loss`` interface is intentionally identical to
``ActorCriticBarlowTwins`` so that **no other file needs to change** — the same
``NP3O`` algorithm, runner, wrapper, and rollout storage work unchanged.

VAE loss:
    L = MSE(code_vel, vel_GT)
      + MSE(decode, current_obs)
      + kl_weight * KL(posterior || N(0,I))

References:
    DreamWaQ wheel-legged gym:
    https://github.com/XinLang2019/Wheel_Legged_Gym
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..utils.common_modules import get_activation, mlp_batchnorm_factory, mlp_factory
from ..utils.normalizer import EmpiricalNormalization

# ============================================================================
# CeNet — Context Estimation Network (VAE)
# ============================================================================


class CeNet(nn.Module):
    """VAE that maps proprioceptive history to (code_vel, code_latent).

    Encoder architecture (matches DreamWaQ reference):
        Linear(history_len*num_prop → 128) → BN → ELU
        Linear(128 → 64)                   → BN → ELU
        ↓ four separate heads from 64-dim bottleneck ↓
        mean_vel / logvar_vel   (num_vel_dims each)
        mean_latent / logvar_latent  (num_latent_dims each)

    Decoder:
        Linear(code_dim → 128) → BN → ELU
        Linear(128 → 128)      → BN → ELU
        Linear(128 → num_prop)   → reconstructed current obs
    """

    def __init__(
        self,
        num_prop: int,
        history_len: int,
        num_vel_dims: int,
        num_latent_dims: int,
        encoder_dims: tuple = (128, 64),
        decoder_dims: tuple = (128, 128),
        activation=None,
    ):
        super().__init__()
        self.num_prop = num_prop
        self.history_len = history_len
        self.num_vel_dims = num_vel_dims
        self.num_latent_dims = num_latent_dims

        cenet_in = num_prop * history_len
        enc_out = encoder_dims[-1]
        code_dim = num_vel_dims + num_latent_dims

        # shared encoder
        self.encoder = nn.Sequential(*mlp_batchnorm_factory(activation, cenet_in, None, list(encoder_dims)))

        # explicit head — supervised to predict base_lin_vel
        self.mean_vel = nn.Linear(enc_out, num_vel_dims)
        self.logvar_vel = nn.Sequential(nn.Linear(enc_out, num_vel_dims), nn.Hardtanh(-5, 5))

        # implicit head — regularised with KL
        self.mean_latent = nn.Linear(enc_out, num_latent_dims)
        self.logvar_latent = nn.Sequential(nn.Linear(enc_out, num_latent_dims), nn.Hardtanh(-5, 5))

        # decoder — reconstructs current proprioceptive frame
        dec_layers = mlp_batchnorm_factory(activation, code_dim, None, list(decoder_dims))
        dec_layers.append(nn.Linear(decoder_dims[-1], num_prop))
        self.decoder = nn.Sequential(*dec_layers)

        # per-dim running normalizer (normalises each frame independently)
        self.obs_normalizer = EmpiricalNormalization(num_prop)

    # ----- internal helpers --------------------------------------------------

    def _normalize(self, policy_obs: torch.Tensor) -> torch.Tensor:
        """Normalise all frames and return flattened ``(B, T*D)``."""
        b, t, d = policy_obs.shape
        normed = self.obs_normalizer(policy_obs.reshape(-1, d)).reshape(b, t, d)
        return normed.reshape(b, -1)

    def _encode(self, flat: torch.Tensor):
        h = self.encoder(flat)
        return (
            self.mean_vel(h),
            self.logvar_vel(h),
            self.mean_latent(h),
            self.logvar_latent(h),
        )

    @staticmethod
    def _reparameterise(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mean + torch.randn_like(mean) * std

    # ----- public forward passes ---------------------------------------------

    def forward(self, policy_obs: torch.Tensor):
        """Train-time forward: sample from both posteriors.

        Returns:
            code    (B, num_vel_dims + num_latent_dims) — sampled codes
            decode  (B, num_prop)                       — reconstructed current frame
            stats   tuple(mean_vel, logvar_vel, mean_lat, logvar_lat)
        """
        flat = self._normalize(policy_obs)
        mean_vel, logvar_vel, mean_lat, logvar_lat = self._encode(flat)

        code_vel = self._reparameterise(mean_vel, logvar_vel)
        code_lat = self._reparameterise(mean_lat, logvar_lat)
        code = torch.cat([code_vel, code_lat], dim=-1)

        decode = self.decoder(code)
        return code, decode, (mean_vel, logvar_vel, mean_lat, logvar_lat)

    def encode_inference(self, policy_obs: torch.Tensor) -> torch.Tensor:
        """Inference-time: deterministic mean (no sampling)."""
        flat = self._normalize(policy_obs)
        mean_vel, _, mean_lat, _ = self._encode(flat)
        return torch.cat([mean_vel, mean_lat], dim=-1)

    def compute_vae_loss(
        self,
        policy_obs: torch.Tensor,
        vel_target: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict]:
        """Compute three-term VAE loss.

        Args:
            policy_obs:  ``(B, history_len, num_prop)``
            vel_target:  ``(B, 3)`` — ground-truth base_lin_vel from critic_obs
            kl_weight:   weight for KL term (beta-VAE style)

        Returns:
            (total_loss, info_dict)
        """
        current = policy_obs[:, -1, :].detach()  # reconstruction target

        code, decode, (mean_vel, logvar_vel, mean_lat, logvar_lat) = self.forward(policy_obs)
        code_vel = code[:, : self.num_vel_dims]

        vel_loss = F.mse_loss(code_vel, vel_target)
        recon_loss = F.mse_loss(decode, current)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_lat - mean_lat.pow(2) - logvar_lat.exp(), dim=-1))

        total = vel_loss + recon_loss + kl_weight * kl_loss
        info = {
            "bt/vel_pred": vel_loss.item(),
            "bt/recon": recon_loss.item(),
            "bt/kl": kl_loss.item(),
        }
        return total, info


# ============================================================================
# ActorCriticDreamWaQ
# ============================================================================


class ActorCriticDreamWaQ(nn.Module):
    """DreamWaQ actor-critic.

    Drop-in replacement for ``ActorCriticBarlowTwins`` — identical external
    interface so that the NP3O runner / algorithm / wrapper need no changes.
    """

    is_recurrent = False

    def __init__(
        self,
        num_prop: int,
        num_critic_obs: int,
        history_len: int,
        num_actions: int,
        # ---- CeNet (VAE) ----
        num_vel_dims: int = 3,
        num_latent_dims: int = 16,
        cenet_encoder_dims: tuple = (128, 64),
        cenet_decoder_dims: tuple = (128, 128),
        kl_weight: float = 1.0,
        # ---- Actor / Critic ----
        actor_hidden_dims: tuple = (512, 256, 128),
        critic_hidden_dims: tuple = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(
                f"[ActorCriticDreamWaQ] unexpected kwargs ignored: {list(kwargs.keys())}",
                stacklevel=2,
            )
        super().__init__()
        if history_len < 2:
            raise ValueError(f"history_len ({history_len}) must be >= 2 for DreamWaQ.")

        self.num_prop = num_prop
        self.num_critic_obs = num_critic_obs
        self.history_len = history_len
        self.num_vel_dims = num_vel_dims
        self.num_latent_dims = num_latent_dims
        self.kl_weight = kl_weight

        act = get_activation(activation)

        # ---- CeNet -----------------------------------------------------------
        self.cenet = CeNet(
            num_prop=num_prop,
            history_len=history_len,
            num_vel_dims=num_vel_dims,
            num_latent_dims=num_latent_dims,
            encoder_dims=cenet_encoder_dims,
            decoder_dims=cenet_decoder_dims,
            activation=act,
        )
        code_dim = num_vel_dims + num_latent_dims

        # ---- Actor MLP -------------------------------------------------------
        actor_input_dim = code_dim + num_prop  # [code | current_obs]
        actor_layers = mlp_factory(act, actor_input_dim, num_actions, list(actor_hidden_dims), last_act=False)
        self.actor = nn.Sequential(*actor_layers)

        # ---- Critic V(s) ----
        critic_input_dim = num_critic_obs
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)

        critic_layers = mlp_factory(act, critic_input_dim, 1, list(critic_hidden_dims), last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # ---- Action distribution --------------------------------------------
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print(f"[ActorCriticDreamWaQ] cenet:  {self.cenet}")
        print(f"[ActorCriticDreamWaQ] actor:  {self.actor}")
        print(f"[ActorCriticDreamWaQ] critic: {self.critic}")

    # ----- std helper --------------------------------------------------------

    @property
    def action_noise_std(self) -> torch.Tensor:
        return self.std

    # ----- distribution properties -------------------------------------------

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    # ----- actor -------------------------------------------------------------

    def update_distribution(self, policy_obs: torch.Tensor):
        current = policy_obs[:, -1, :]
        with torch.no_grad():
            code, _, _ = self.cenet(policy_obs)
        actor_input = torch.cat([code.detach(), current.detach()], dim=-1)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, policy_obs: torch.Tensor, **kwargs):
        self.update_distribution(policy_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_obs: torch.Tensor) -> torch.Tensor:
        """Deterministic inference — uses CeNet mean (no sampling)."""
        current = policy_obs[:, -1, :]
        code = self.cenet.encode_inference(policy_obs)
        actor_input = torch.cat([code, current], dim=-1)
        return self.actor(actor_input)

    # ----- critic / cost -----------------------------------------------------

    def _get_critic_backbone_input(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """Normalise critic_obs directly — matches reference (no pre-encoder)."""
        return self.critic_obs_normalizer(critic_obs)

    def evaluate(self, critic_obs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(self._get_critic_backbone_input(critic_obs))

    # ----- DreamWaQ VAE loss (same interface as BarlowTwins) ----------------

    def imitation_learning_loss(
        self, policy_obs: torch.Tensor, critic_obs: torch.Tensor, _imi_weight=1
    ) -> torch.Tensor:
        """VAE loss: vel_estimation + reconstruction + KL.

        ``critic_obs[:, :3]`` must be ``base_lin_vel × 2.0`` (the first term in
        ``CriticCfg``), which is the explicit velocity supervision target.
        """
        vel_target = critic_obs[:, : self.num_vel_dims]
        total, _ = self.cenet.compute_vae_loss(policy_obs, vel_target, kl_weight=self.kl_weight)
        return total

    # ----- ONNX / JIT export ------------------------------------------------

    def save_torch_jit_policy(self, path: str, device: str) -> dict:
        """Export inference policy as TorchScript + ONNX.

        Input:
            ``nn_input`` : ``(1, history_len, num_prop)`` — rolling history buffer
            (current frame = last slice ``[:, -1, :]``)

        Output:
            ``nn_output`` : ``(1, num_actions)``           — deterministic action mean
        """
        import os

        os.makedirs(path, exist_ok=True)

        wrapper = _InferenceDreamWaQ(self.cenet, self.actor)
        wrapper.eval()

        history = torch.randn(1, self.history_len, self.num_prop, device=device)

        jit_path = os.path.join(path, "policy.pt")
        torch.jit.save(torch.jit.trace(wrapper, (history,)), jit_path)
        print(f"[ActorCriticDreamWaQ] saved TorchScript → {jit_path}")

        onnx_path = os.path.join(path, "policy.onnx")
        torch.onnx.export(
            wrapper,
            (history,),
            onnx_path,
            input_names=["nn_input"],
            output_names=["nn_output"],
            opset_version=13,
            export_params=True,
            verbose=False,
        )
        print(f"[ActorCriticDreamWaQ] saved ONNX        → {onnx_path}")
        return {"jit": os.path.abspath(jit_path), "onnx": os.path.abspath(onnx_path)}


# ============================================================================
# Inference wrapper (for ONNX export)
# ============================================================================


class _InferenceDreamWaQ(nn.Module):
    """Wraps CeNet + Actor into a single traceable forward.

    Single input: ``history (B, history_len, num_prop)`` — rolling buffer
    where the last frame ``[:, -1, :]`` is the current proprio observation.
    """

    def __init__(self, cenet: CeNet, actor: nn.Module):
        super().__init__()
        self.cenet = cenet
        self.actor = actor

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        code = self.cenet.encode_inference(history)
        current = history[:, -1, :]
        return self.actor(torch.cat([code, current], dim=-1))
