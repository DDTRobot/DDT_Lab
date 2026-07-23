# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost manager for NP3O constrained training, in Isaac Lab manager style.

``CostTermCfg`` extends ``isaaclab.managers.ManagerTermBaseCfg`` (the same base
``RewardTermCfg`` / ``EventTermCfg`` use), so the standard ``func`` + ``params``
+ ``SceneEntityCfg`` resolution plumbing is inherited — only the NP3O-specific
fields (``scale`` / ``d_value`` / ``k_value``) are added.

Usage in env_cfg::

    from ddt_lab.managers import CostTermCfg

    @configclass
    class MyCostsCfg:
        pos_limit = CostTermCfg(
            func=mdp.joint_pos_limits,
            scale=1.0,
            d_value=0.0,
            k_value=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip.*"])},
        )

    @configclass
    class MyEnvCfg(D1RoughEnvCfg):
        costs: MyCostsCfg = MyCostsCfg()

The wrapper auto-detects ``env_cfg.costs`` and instantiates a ``CostManager``;
if the attribute is missing or ``None``, NP3O's cost path stays dormant
(zeros + k=0) and the algorithm degrades to PPO+BarlowTwins.

Cost functions follow the Isaac Lab reward-function signature:
``func(env, **params) -> torch.Tensor`` of shape ``(num_envs,)``.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from isaaclab.utils import configclass
from prettytable import PrettyTable


@configclass
class CostTermCfg(ManagerTermBaseCfg):
    """Configuration for a single cost term.

    Inherits ``func`` and ``params`` from ``ManagerTermBaseCfg`` (same as
    ``RewardTermCfg`` / ``EventTermCfg``); adds NP3O-specific scaling and
    Lagrangian fields.
    """

    scale: float = 1.0
    """Multiplied with the raw cost value before storage / loss computation."""

    d_value: float = 0.0
    """Per-term safety budget. ``cost_violation = (1-γ)*(cost_returns - d_value) + ...``"""

    k_value: float = 0.01
    """Initial Lagrangian multiplier for this term. Updated by ``NP3O.update_k_value``."""


class CostManager:
    """Computes cost vectors per step. Each cost term is a ``CostTermCfg``.

    Attributes mirror what ``OnConstraintPolicyRunner`` reads off the env wrapper:

    * ``num_costs``        — int
    * ``active_terms``     — list[str]
    * ``k_values``         — ``(1, num_costs)``
    * ``d_values_tensor``  — ``(1, 1, num_costs)`` (broadcasts over T, num_envs)

    Per-episode cost statistics are tracked the same way ``RewardManager``
    handles per-episode reward stats: ``compute()`` accumulates into
    ``_episode_sums`` and ``log_episode(env_ids, max_episode_length_s)`` returns
    ``{"Episode_Cost/<name>": ...}`` for the wrapper to merge into
    ``extras['log']`` at episode boundaries.
    """

    def __init__(self, cfg, env, device: str):
        self._cfg = cfg
        self._env = env
        self._device = device

        self._terms: list[tuple[str, CostTermCfg]] = []
        for name in dir(cfg):
            if name.startswith("_"):
                continue
            term = getattr(cfg, name)
            if not isinstance(term, CostTermCfg):
                continue
            # resolve SceneEntityCfg in params (same as RewardManager does)
            for value in term.params.values():
                if isinstance(value, SceneEntityCfg):
                    value.resolve(env.scene)
            self._terms.append((name, term))

        if not self._terms:
            raise ValueError(
                f"CostsCfg '{type(cfg).__name__}' has no CostTermCfg attributes; "
                "either remove the cfg or add at least one cost term."
            )

        self.num_costs: int = len(self._terms)
        self.active_terms: list[str] = [n for n, _ in self._terms]

        k_init = [term.k_value for _, term in self._terms]
        d_init = [term.d_value for _, term in self._terms]
        self.k_values = torch.tensor(k_init, device=device).view(1, -1)
        self.d_values_tensor = torch.tensor(d_init, device=device).view(1, 1, -1)

        # per-episode accumulators (one scalar per env, per term)
        self._episode_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(env.num_envs, device=device) for name in self.active_terms
        }

        # mirror Isaac Lab's "[INFO] Reward Manager: <RewardManager> ..." line
        print("[INFO] Cost Manager: ", self)

    def __str__(self) -> str:
        """PrettyTable representation, matching Isaac Lab manager style."""
        msg = f"<CostManager> contains {self.num_costs} active terms.\n"
        table = PrettyTable()
        table.title = "Active Cost Terms"
        table.field_names = ["Index", "Name", "Scale", "d_value", "k_value"]
        table.align["Name"] = "l"
        for align_col in ("Scale", "d_value", "k_value"):
            table.align[align_col] = "r"
        for idx, (name, term) in enumerate(self._terms):
            table.add_row([idx, name, f"{term.scale:.4g}", f"{term.d_value:.4g}", f"{term.k_value:.4g}"])
        msg += table.get_string()
        msg += "\n"
        return msg

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        """Return ``(num_envs, num_costs)`` per-step costs (clamped to >=0).

        Side-effect: accumulates each term into ``self._episode_sums``,
        multiplied by ``env.step_dt`` so the episode totals represent
        time-integrated cost (e.g. cost-seconds)."""
        cols = []
        for name, term in self._terms:
            val = (term.func(self._env, **term.params) * term.scale).clamp_min_(0.0)
            self._episode_sums[name] += val * self._env.step_dt
            cols.append(val.unsqueeze(-1))
        return torch.cat(cols, dim=-1)

    @torch.no_grad()
    def log_episode(self, env_ids: torch.Tensor, max_episode_length_s: float) -> dict[str, torch.Tensor]:
        """Pop and return per-term mean cost over the episodes that just ended.

        Mirrors RewardManager: each value is divided by ``max_episode_length_s``
        so it is interpretable as a per-second average.
        """
        log: dict[str, torch.Tensor] = {}
        for name in self.active_terms:
            sums = self._episode_sums[name]
            log[f"Episode_Cost/{name}"] = sums[env_ids].mean() / max_episode_length_s
            sums[env_ids] = 0.0
        return log
