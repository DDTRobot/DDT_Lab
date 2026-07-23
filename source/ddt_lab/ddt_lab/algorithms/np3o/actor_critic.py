# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""BarlowTwins actor-critic for NP3O, ported from LocomotionWithNP3O and adapted
to Isaac Lab's two-ObsGroup convention.

Differences vs. the upstream reference:

* The upstream actor-critic packs everything into a single 1D ``obs`` and slices
  by hard-coded offsets (``obs[:, :3]`` is vel, ``obs[:, 3:n_proprio]`` is the
  rest of proprio, ``obs[:, -n_hist*n_prop:]`` is history, etc.). Isaac Lab
  cleanly separates ``policy`` and ``critic`` ObservationGroups, so we drop the
  slicing and consume two distinct tensors:
    - ``policy_obs``: ``(B, history_len, n_proprio)`` — what the actor sees.
    - ``critic_obs``: ``(B, n_critic)`` — full privileged obs for V(s) / C(s).
* Vel head supervision now reads ``critic_obs[:, :3]`` (base_lin_vel must be the
  first term in CriticCfg). The actor never sees vel directly.
* Optional ``priv_encoder`` / ``scan_encoder`` heads are dropped — Isaac Lab's
  CriticCfg is already compact and goes straight into the critic MLP.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ..utils.common_modules import get_activation, mlp_batchnorm_factory, mlp_factory
from ..utils.normalizer import EmpiricalNormalization


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class MlpBarlowTwinsActor(nn.Module):
    """Teacher actor with BarlowTwins SSL on the history encoder."""

    def __init__(
        self,
        num_prop,
        bt_window,
        num_actions,
        mlp_encoder_dims,
        latent_encoder_dims,
        actor_dims,
        projector_encoder_dims,
        activation,
    ):
        super().__init__()
        self.num_prop = num_prop
        self.bt_window = bt_window  # number of history frames consumed by the encoder

        self.obs_normalizer = EmpiricalNormalization(shape=num_prop)

        self.mlp_encoder = nn.Sequential(
            *mlp_batchnorm_factory(
                activation=activation,
                input_dims=num_prop * bt_window,
                out_dims=None,
                hidden_dims=mlp_encoder_dims,
            )
        )
        self.latent_layer = nn.Sequential(
            *mlp_batchnorm_factory(
                activation=activation,
                input_dims=mlp_encoder_dims[-1],
                out_dims=latent_encoder_dims[-1],
                hidden_dims=[latent_encoder_dims[-2]],
            )
        )
        self.vel_layer = nn.Linear(mlp_encoder_dims[-1], 3)

        latent_dim = latent_encoder_dims[-1]
        actor_input_dim = latent_dim + num_prop + 3  # latent + current proprio + predicted vel
        self.actor = nn.Sequential(
            *mlp_factory(
                activation=activation,
                input_dims=actor_input_dim,
                out_dims=num_actions,
                hidden_dims=actor_dims,
            )
        )

        self.projector = nn.Sequential(
            *mlp_batchnorm_factory(
                activation=activation,
                input_dims=latent_dim,
                out_dims=projector_encoder_dims[-1],
                hidden_dims=projector_encoder_dims,
                bias=False,
            ),
            nn.BatchNorm1d(projector_encoder_dims[-1], affine=False),
        )

    def _normalize(self, current, hist):
        current = self.obs_normalizer(current)
        b, t, d = hist.shape
        hist = self.obs_normalizer(hist.reshape(-1, d)).reshape(b, t, d)
        return current, hist

    def forward(self, current, hist):
        """Forward pass.

        Args:
            current: (B, num_prop) — proprio at time t.
            hist:    (B, bt_window, num_prop) — proprio at [t-bt_window, ..., t-1].
        """
        current, hist = self._normalize(current, hist)

        # Reference's "view 1": shift hist by one and append current → ends at t.
        full = torch.cat([hist[:, 1:, :], current.unsqueeze(1)], dim=1)
        b = current.shape[0]

        with torch.no_grad():
            latent = self.mlp_encoder(full.reshape(b, -1))
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)

        actor_input = torch.cat([vel.detach(), z.detach(), current.detach()], dim=-1)
        return self.actor(actor_input)

    def barlow_twins_loss(self, current, hist, priv_vel, weight=5e-3):
        """Two-view BT loss with the vel-prediction privileged supervision."""
        current, hist = self._normalize(current, hist)
        current = current.detach()
        hist = hist.detach()

        full = torch.cat([hist[:, 1:, :], current.unsqueeze(1)], dim=1)
        b = current.shape[0]

        z1 = self.mlp_encoder(full.reshape(b, -1))  # ends at t
        z2 = self.mlp_encoder(hist.reshape(b, -1))  # ends at t-1

        z1_l = self.latent_layer(z1)
        z1_v = self.vel_layer(z1)
        z2_l = self.latent_layer(z2)

        z1_l = self.projector(z1_l)
        z2_l = self.projector(z2_l)

        c = z1_l.T @ z2_l
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        priv_loss = F.mse_loss(z1_v, priv_vel)

        return on_diag + weight * off_diag + priv_loss


class ActorCriticBarlowTwins(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_prop,
        num_critic_obs,
        history_len,
        num_actions,
        num_costs,
        num_priv_latent=0,
        num_scan=0,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        priv_encoder_dims=(),
        scan_encoder_dims=(128, 64, 32),
        bt_window=5,
        bt_mlp_encoder_dims=(128, 64),
        bt_latent_encoder_dims=(32, 16),
        bt_projector_dims=(64,),
        activation="elu",
        init_noise_std=1.0,
        imi_flag=True,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(
                f"[ActorCriticBarlowTwins] unexpected kwargs will be ignored: {list(kwargs.keys())}. "
                "Check your policy cfg for typos.",
                stacklevel=2,
            )
        super().__init__()
        if history_len < bt_window + 1:
            raise ValueError(
                f"history_len ({history_len}) must be >= bt_window + 1 ({bt_window + 1}); "
                "BarlowTwins needs an extra past frame for the second view."
            )

        self.num_prop = num_prop
        self.num_critic_obs = num_critic_obs
        self.num_priv_latent = num_priv_latent
        self.num_scan = num_scan
        self.history_len = history_len
        self.bt_window = bt_window
        self.imi_flag = imi_flag

        # CriticCfg layout: [prop_for_critic | priv_latent | height_scan]
        self.num_prop_for_critic = num_critic_obs - num_priv_latent - num_scan

        activation_module = get_activation(activation)

        # ---- actor (BarlowTwins teacher) ----
        self.actor_teacher_backbone = MlpBarlowTwinsActor(
            num_prop=num_prop,
            bt_window=bt_window,
            num_actions=num_actions,
            mlp_encoder_dims=list(bt_mlp_encoder_dims),
            latent_encoder_dims=list(bt_latent_encoder_dims),
            actor_dims=list(actor_hidden_dims),
            projector_encoder_dims=list(bt_projector_dims),
            activation=activation_module,
        )

        # ---- scan encoder (Identity / no-op when num_scan=0) ----
        # Mirrors reference: scan_encoder compresses height_scan before critic.
        # D1FlatCfgPPO uses scan_encoder_dims=[128,64,32] with n_scan=187.
        # Flat tasks set n_scan=0, encoder is bypassed entirely.
        if num_scan > 0 and len(scan_encoder_dims) > 0:
            scan_enc_layers = mlp_factory(
                activation_module,
                num_scan,
                scan_encoder_dims[-1],
                list(scan_encoder_dims[:-1]),
                last_act=False,
            )
            self.scan_encoder = nn.Sequential(*scan_enc_layers)
            scan_encoder_output_dim = scan_encoder_dims[-1]
            print(f"[ActorCriticBarlowTwins] scan_encoder: {self.scan_encoder}")
        else:
            self.scan_encoder = None
            scan_encoder_output_dim = num_scan  # 0 for flat (no scan)
            if num_scan > 0:
                print(f"[ActorCriticBarlowTwins] scan_encoder: Identity ({num_scan} dims)")
            else:
                print("[ActorCriticBarlowTwins] scan_encoder: None (n_scan=0)")

        # ---- privileged encoder (Identity if priv_encoder_dims=[]) ----
        if len(priv_encoder_dims) > 0:
            priv_enc_layers = mlp_factory(
                activation_module,
                num_priv_latent,
                None,
                list(priv_encoder_dims),
                last_act=True,
            )
            self.priv_encoder = nn.Sequential(*priv_enc_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
            print(f"[ActorCriticBarlowTwins] priv_encoder: {self.priv_encoder}")
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent
            print(f"[ActorCriticBarlowTwins] priv_encoder: Identity ({num_priv_latent} dims pass-through)")

        # ---- critic V(s) and cost C(s) ----
        # Input = prop_for_critic + scan_encoded + priv_encoded
        # Matches reference: [obs_prop | scan_latent | priv_encoded]
        critic_input_dim = self.num_prop_for_critic + scan_encoder_output_dim + priv_encoder_output_dim
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        critic_layers = mlp_factory(activation_module, critic_input_dim, 1, list(critic_hidden_dims), last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        cost_layers = mlp_factory(
            activation_module, critic_input_dim, max(num_costs, 1), list(critic_hidden_dims), last_act=False
        )
        cost_layers.append(nn.Softplus())
        self.cost = nn.Sequential(*cost_layers)

        # ---- action distribution ----
        # ``log`` parameterization: std = exp(log_std) — always > 0, SGD cannot
        # collapse it to zero even without entropy regularisation.  Use ``scalar``
        # only for backwards-compat with pre-trained checkpoints.
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print(f"[ActorCriticBarlowTwins] actor: {self.actor_teacher_backbone}")
        print(f"[ActorCriticBarlowTwins] critic: {self.critic}")
        print(f"[ActorCriticBarlowTwins] cost: {self.cost}")

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def action_noise_std(self) -> torch.Tensor:
        """Current scalar std per action (for logging). Works for both parameterisations."""
        return self.std

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # ----- helpers
    def _split_policy_obs(self, policy_obs):
        """Split policy obs ``(B, history_len, num_prop)`` into ``(current, hist)``.

        ``current`` is the latest frame; ``hist`` is the bt_window frames
        preceding it (the slot just before current is the "second view").
        """
        if policy_obs.dim() != 3:
            raise ValueError(f"policy_obs must be 3D (B, T, D); got {tuple(policy_obs.shape)}")
        current = policy_obs[:, -1, :]
        hist = policy_obs[:, -self.bt_window - 1 : -1, :]
        return current, hist

    # ----- actor

    def update_distribution(self, policy_obs):
        current, hist = self._split_policy_obs(policy_obs)
        mean = self.actor_teacher_backbone(current, hist)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, policy_obs, **kwargs):
        self.update_distribution(policy_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, policy_obs):
        current, hist = self._split_policy_obs(policy_obs)
        return self.actor_teacher_backbone(current, hist)

    # ----- critic / cost
    def _get_critic_backbone_input(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """Normalise and split critic_obs into three segments, then encode each.

        Layout (matches reference ``get_critic_obs``):

            critic_obs = [prop_for_critic | priv_latent | height_scan]
                                  ↓               ↓             ↓
                              pass-through   priv_encoder   scan_encoder
                                  ↓               ↓             ↓
            backbone_input  =  [prop  |  scan_latent  |  priv_encoded]

        When ``num_scan = 0`` (flat tasks) the scan segment and encoder are
        skipped entirely; when ``priv_encoder = Identity`` the priv segment is
        a pass-through (D1FlatCfgPPO default: ``priv_encoder_dims=[]``).
        """
        normed = self.critic_obs_normalizer(critic_obs)
        prop = normed[:, : self.num_prop_for_critic]

        # Critic layout: [prop | priv | scan]
        # Matches reference: backbone_input = cat([obs_prop, priv_latent, scan_latent])
        parts = [prop]

        if self.num_priv_latent > 0:
            priv_start = self.num_prop_for_critic
            priv = normed[:, priv_start : priv_start + self.num_priv_latent]
            parts.append(self.priv_encoder(priv))

        if self.num_scan > 0:
            scan_start = self.num_prop_for_critic + self.num_priv_latent
            scan = normed[:, scan_start : scan_start + self.num_scan]
            if self.scan_encoder is not None:
                parts.append(self.scan_encoder(scan))
            else:
                parts.append(scan)

        return torch.cat(parts, dim=-1)

    def evaluate(self, critic_obs, **kwargs):
        return self.critic(self._get_critic_backbone_input(critic_obs))

    def evaluate_cost(self, critic_obs, **kwargs):
        return self.cost(self._get_critic_backbone_input(critic_obs))

    # ----- BarlowTwins SSL
    def imitation_learning_loss(self, policy_obs, critic_obs, _imi_weight=1):
        """Vel supervision uses ``critic_obs[:, :3]`` (must be base_lin_vel)."""
        current, hist = self._split_policy_obs(policy_obs)
        priv_vel = critic_obs[:, :3]
        return self.actor_teacher_backbone.barlow_twins_loss(current, hist, priv_vel, weight=5e-3)

    def imitation_mode(self):
        pass

    def save_torch_jit_policy(self, path: str, device: str) -> dict:
        """Export the inference policy as TorchScript (+ ONNX).

        The exported policy takes two inputs (matches the legacy LocomotionWithNP3O
        export shape so deployment runtimes can stay unchanged):

        * ``nn_input0``: ``(B, num_prop)`` — proprio at time t
        * ``nn_input1``: ``(B, history_len, num_prop)`` — full history buffer

        Internally the wrapper slices the last ``bt_window`` frames (excluding
        the current step) from ``nn_input1`` before calling the BarlowTwins
        actor backbone, so callers always feed the same fixed-shape history
        buffer regardless of ``bt_window``.

        Returns ``{"jit": <abs path>, "onnx": <abs path>}``.
        """
        import os

        os.makedirs(path, exist_ok=True)

        backbone = self.actor_teacher_backbone
        was_training = backbone.training
        backbone.eval()
        try:
            wrapper = _InferenceWrapper(backbone, history_len=self.history_len, bt_window=self.bt_window)
            wrapper.eval()

            current = torch.randn(1, self.num_prop, device=device)
            history = torch.randn(1, self.history_len, self.num_prop, device=device)

            jit_path = os.path.join(path, "policy.pt")
            traced = torch.jit.trace(wrapper, (current, history))
            traced.save(jit_path)
            print(f"[ActorCriticBarlowTwins] saved TorchScript policy to {jit_path}")

            onnx_path = os.path.join(path, "policy.onnx")
            torch.onnx.export(
                wrapper,
                (current, history),
                onnx_path,
                input_names=["nn_input0", "nn_input1"],
                output_names=["nn_output"],
                opset_version=13,
                export_params=True,
                verbose=False,
            )
            print(f"[ActorCriticBarlowTwins] saved ONNX policy to {onnx_path}")
        finally:
            if was_training:
                backbone.train()

        return {"jit": os.path.abspath(jit_path), "onnx": os.path.abspath(onnx_path)}


class _InferenceWrapper(torch.nn.Module):
    """Inference-only wrapper: receives the full history buffer and reproduces
    the reference NP3O ONNX graph shape.

    Layout of ``history`` is assumed to be ``[t-history_len+1, ..., t-1, t]``
    (latest frame at the end — same as the env-side ObsGroup history). The
    forward path mirrors the reference pipeline exactly:

        history → Slice [:, 1:]         (drop oldest frame)
                → Concat with current   (append current at the end)
                → Slice [:, -bt_window:] (take last ``bt_window`` frames)
                → Reshape → MLP encoder → ...

    This is mathematically equivalent to slicing ``history[:, -bt_window-1:-1]``
    upfront, but emits the same node sequence as the legacy export so deployment
    runtimes that pattern-match on the graph keep working.
    """

    def __init__(self, backbone: torch.nn.Module, history_len: int, bt_window: int):
        super().__init__()
        self.backbone = backbone
        self.history_len = history_len
        self.bt_window = bt_window

    def forward(self, current: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        # 1) normalize (use the same EmpiricalNormalization as the backbone)
        current_n = self.backbone.obs_normalizer(current)
        b, t, d = history.shape
        history_n = self.backbone.obs_normalizer(history.reshape(-1, d)).reshape(b, t, d)

        # 2) reference-style slice → concat → slice
        drop_first = history_n[:, 1:, :]  # (B, history_len-1, num_prop)
        full = torch.cat([drop_first, current_n.unsqueeze(1)], dim=1)  # (B, history_len, num_prop)
        window = full[:, -self.bt_window :, :]  # (B, bt_window, num_prop)

        # 3) BarlowTwins MLP encoder + vel/latent heads + actor MLP
        latent = self.backbone.mlp_encoder(window.reshape(current_n.shape[0], -1))
        z = self.backbone.latent_layer(latent)
        vel = self.backbone.vel_layer(latent)
        actor_input = torch.cat([vel, z, current_n], dim=-1)
        return self.backbone.actor(actor_input)
