# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner for NP3O (port of LocomotionWithNP3O OnConstraintPolicyRunner).

Adapted to read a flat dict cfg (the reference reads ``train_cfg['runner' /
'algorithm' / 'policy']``) instead of pulling per-env attributes through
``train_cfg``. The env API matches the reference's ``VecEnv`` (see
``IsaacLabNP3OWrapper``).
"""

import os
import statistics
import sys
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

import rsl_rl
from rsl_rl.utils import store_code_state

from .actor_critic import ActorCriticBarlowTwins
from .np3o import NP3O


def _console_write(msg: str) -> None:
    """Write to the *real* stdout so Isaac Sim's carb logger can't swallow it."""
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()


def _short_episode_key(key: str) -> str:
    """Map Isaac Lab's slashed group names to the reference's flat prefixes."""
    if "/" not in key:
        return key
    head, tail = key.split("/", 1)
    head_lower = head.lower()
    if "reward" in head_lower:
        return f"rew_{tail}"
    if "cost" in head_lower:
        return f"cost_{tail}"
    if "termination" in head_lower:
        return f"term_{tail}"
    if "metric" in head_lower:
        return f"metric_{tail}"
    return key.replace("/", "_")


class OnConstraintPolicyRunner:
    """Training runner for cost-constrained policy optimization."""

    def _get_actor_critic_class(self, name: str):
        """Return the actor-critic class for *name*.

        Override in subclasses to register additional policy classes without
        modifying this file (e.g. DreamWaQ, RMA).
        """
        registry = {'ActorCriticBarlowTwins': ActorCriticBarlowTwins}
        if name not in registry:
            raise KeyError(
                f"Unknown policy_class_name '{name}'. "
                f"Available: {list(registry.keys())}. "
                "Subclass OnConstraintPolicyRunner and override "
                "_get_actor_critic_class() to register new policy classes."
            )
        return registry[name]

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg['runner']
        self.alg_cfg = dict(train_cfg['algorithm'])
        self.policy_cfg = dict(train_cfg['policy'])
        self.device = device
        self.env = env

        actor_critic_class = self._get_actor_critic_class(self.cfg['policy_class_name'])
        actor_critic = actor_critic_class(
            num_prop=self.env.cfg.env.n_proprio,
            num_critic_obs=self.env.cfg.env.n_critic,
            num_priv_latent=self.env.cfg.env.n_priv_latent,
            num_scan=self.env.cfg.env.n_scan,
            history_len=self.env.cfg.env.history_len,
            num_actions=self.env.num_actions,
            num_costs=self.env.num_costs,
            **self.policy_cfg,
        ).to(self.device)

        self.alg_cfg['k_value'] = self.env.cost_k_values
        alg_class = {'NP3O': NP3O}[self.cfg['algorithm_class_name']]
        self.alg: NP3O = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg['num_steps_per_env']
        self.save_interval = self.cfg['save_interval']

        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=list(self.env.policy_obs_shape),
            critic_obs_shape=list(self.env.critic_obs_shape),
            action_shape=[self.env.num_actions],
            cost_shape=[self.env.num_costs],
            cost_d_values=self.env.cost_d_values_tensor,
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # repos to snapshot via ``store_code_state`` at the start of learn().
        # Always include vendored rsl_rl; train script appends its own __file__.
        self.git_status_repos = [rsl_rl.__file__]

        self.env.reset()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register an extra source file whose git repo should be snapshotted.

        Mirrors ``rsl_rl.runners.OnPolicyRunner.add_git_repo_to_log``.
        """
        self.git_status_repos.append(repo_file_path)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            # snapshot git status / diff for every registered repo
            store_code_state(self.log_dir, self.git_status_repos)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        infos = {}
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        if self.alg.actor_critic.imi_flag and self.cfg.get('resume', False):
            self.alg.actor_critic.imitation_mode()

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, infos)
                    obs, privileged_obs, rewards, costs, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    costs = costs.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, costs, dones, infos)

                    if self.log_dir is not None:
                        # Isaac Lab puts per-episode stats under extras['log'];
                        # reference codebase uses extras['episode']. Accept both.
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        elif 'log' in infos:
                            ep_infos.append(infos['log'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            self.alg.compute_returns(critic_obs)
            self.alg.compute_cost_returns(critic_obs)
            self.alg.update_k_value(it)

            (mean_value_loss, mean_cost_value_loss, mean_viol_loss, mean_surrogate_loss, mean_imitation_loss, obs_batch_min, obs_batch_max) = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0 and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
            ep_infos.clear()

        self.current_learning_iteration = tot_iter
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f'model_{self.current_learning_iteration}.pt'))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = ''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # Tensorboard: keep the original (possibly slashed) key.
                self.writer.add_scalar(f'Episode/{key}', value, locs['it'])
                # Console: shorten Isaac Lab's slash-prefixed names to the
                # reference's `rew_` / `cost_` / `term_` / `metric_` style.
                ep_string += f"""{f'Mean episode {_short_episode_key(key)}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.action_noise_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # Immediate per-step stats over the just-collected rollout (always available,
        # independent of whether any episode finished this iteration).
        step_reward = self.alg.storage.rewards.mean().item()
        step_cost = self.alg.storage.costs.mean().item()

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/cost_value_function', locs['mean_cost_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_viol_loss', locs['mean_viol_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_imitation_loss', locs['mean_imitation_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection_time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        self.writer.add_scalar('Data/obs_max', locs['obs_batch_max'], locs['it'])
        self.writer.add_scalar('Data/obs_min', locs['obs_batch_min'], locs['it'])
        self.writer.add_scalar('Train/step_reward_mean', step_reward, locs['it'])
        self.writer.add_scalar('Train/step_cost_mean', step_cost, locs['it'])

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])

        header = f' \033[1m Learning iteration {locs["it"]}/{locs["tot_iter"]} \033[0m '
        log_string = (
            f"""{'#' * width}\n"""
            f"""{header.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
            f"""{'Cost value function loss:':>{pad}} {locs['mean_cost_value_loss']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
            f"""{'Viol loss:':>{pad}} {locs['mean_viol_loss']:.4f}\n"""
            f"""{'Imitation loss:':>{pad}} {locs['mean_imitation_loss']:.4f}\n"""
            f"""{'Step reward (mean):':>{pad}} {step_reward:.4f}\n"""
            f"""{'Step cost (mean):':>{pad}} {step_cost:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
        )
        if len(locs['rewbuffer']) > 0:
            log_string += f"""{'Episode reward (mean):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Episode length (mean):':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        log_string += ep_string
        _start = locs['tot_iter'] - locs['num_learning_iterations']
        iters_done = locs['it'] - _start + 1
        iters_remaining = max(locs['tot_iter'] - locs['it'] - 1, 0)
        eta_seconds = self.tot_time / max(iters_done, 1) * iters_remaining

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}\n"""
        )
        _console_write(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'iter': self.current_learning_iteration,
                'infos': infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
