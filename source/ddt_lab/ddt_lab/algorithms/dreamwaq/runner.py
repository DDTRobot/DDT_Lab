# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ on-policy training runner.

No cost constraints.  Uses ``PPO_DreamWaQ`` (dual-optimizer VAE+PPO) and ``RolloutStorageDreamWaQ``.
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

from .actor_critic import ActorCriticDreamWaQ
from .ppo_dreamwaq import PPO_DreamWaQ


def _console_write(msg: str) -> None:
    sys.__stdout__.write(msg + "\n")
    sys.__stdout__.flush()


def _short_episode_key(key: str) -> str:
    if "/" not in key:
        return key
    head, tail = key.split("/", 1)
    h = head.lower()
    if "reward" in h:
        return f"rew_{tail}"
    if "cost" in h:
        return f"cost_{tail}"
    if "termination" in h:
        return f"term_{tail}"
    if "metric" in h:
        return f"metric_{tail}"
    return key.replace("/", "_")


class OnPolicyRunner:
    """DreamWaQ on-policy training runner (no cost)."""

    def _get_actor_critic_class(self, name: str):
        registry = {"ActorCriticDreamWaQ": ActorCriticDreamWaQ}
        if name not in registry:
            raise KeyError(f"Unknown policy_class_name '{name}'. Available: {list(registry.keys())}")
        return registry[name]

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])
        self.device = device
        self.env = env

        actor_critic_class = self._get_actor_critic_class(self.cfg["policy_class_name"])
        actor_critic = actor_critic_class(
            num_prop=self.env.cfg.env.n_proprio,
            num_critic_obs=self.env.cfg.env.n_critic,
            history_len=self.env.cfg.env.history_len,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        alg_class = {"PPO_DreamWaQ": PPO_DreamWaQ}[self.cfg["algorithm_class_name"]]
        self.alg: PPO_DreamWaQ = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=list(self.env.policy_obs_shape),
            critic_obs_shape=list(self.env.critic_obs_shape),
            action_shape=[self.env.num_actions],
        )

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        self.env.reset()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            store_code_state(self.log_dir, self.git_status_repos)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        critic_obs = self.env.get_privileged_observations().to(self.device)

        self.alg.actor_critic.train()
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    # wrapper step returns (policy_obs, critic_obs, rewards, dones, infos)
                    obs, critic_obs, rewards, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
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
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0 and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration = tot_iter
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar(f"Episode/{key}", value, locs["it"])
                ep_string += f"{f'Mean episode {_short_episode_key(key)}:':>{pad}} {value:.4f}\n"

        loss_dict = locs["loss_dict"]
        mean_std = self.alg.actor_critic.action_noise_std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time)
        step_reward = self.alg.storage.rewards.mean().item()

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        self.writer.add_scalar("Train/step_reward_mean", step_reward, locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"{'#' * width}\n{header.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
        )
        for key, value in loss_dict.items():
            log_string += f"{f'{key}:':>{pad}} {value:.4f}\n"
        log_string += f"{'Step reward (mean):':>{pad}} {step_reward:.4f}\n"
        log_string += f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"{'Episode reward (mean):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Episode length (mean):':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        log_string += ep_string

        _start = locs["tot_iter"] - locs["num_learning_iterations"]
        iters_done = locs["it"] - _start + 1
        iters_remaining = max(locs["tot_iter"] - locs["it"] - 1, 0)
        eta = self.tot_time / max(iters_done, 1) * iters_remaining
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta))}\n"
        )
        _console_write(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "vae_optimizer_state_dict": self.alg.vae_optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True):
        loaded = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded:
            self.alg.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        if load_optimizer and "vae_optimizer_state_dict" in loaded:
            self.alg.vae_optimizer.load_state_dict(loaded["vae_optimizer_state_dict"])
        if "iter" in loaded:
            self.current_learning_iteration = loaded["iter"]
        return loaded.get("infos")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
