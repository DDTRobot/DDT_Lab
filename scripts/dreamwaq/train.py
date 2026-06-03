# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an Isaac Lab task with the DreamWaQ algorithm (CeNet VAE + PPO)."""

import argparse
import importlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Train an Isaac Lab task with DreamWaQ.')
parser.add_argument('--num_envs', type=int, default=None, help='Number of parallel envs.')
parser.add_argument('--task', type=str, required=True, help='Gym task ID.')
parser.add_argument('--seed', type=int, default=None, help='Environment seed.')
parser.add_argument('--max_iterations', type=int, default=None, help='Override iterations.')
parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint.')
parser.add_argument('--load_run', type=str, default=None, help='Run dir regex when resuming.')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint filename regex.')
parser.add_argument('--experiment_name', type=str, default=None, help='Override experiment name.')
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from datetime import datetime

import ddt_lab.tasks  # noqa: F401  -- registers tasks
import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from ddt_lab.algorithms.dreamwaq import IsaacLabDreamWaQWrapper, OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _resolve_cfg(entry_point: str) -> dict:
    module_name, attr = entry_point.split(':')
    obj = getattr(importlib.import_module(module_name), attr)
    cfg = obj() if callable(obj) else obj
    if not isinstance(cfg, dict):
        raise TypeError(f"DreamWaQ cfg entry point '{entry_point}' must return a dict")
    return cfg


def main():
    spec = gym.spec(args_cli.task)
    entry = spec.kwargs.get('dreamwaq_cfg_entry_point')
    if entry is None:
        raise KeyError(
            f"Task '{args_cli.task}' has no 'dreamwaq_cfg_entry_point'. "
            'Register a dreamwaq_cfg_entry_point alongside env_cfg_entry_point.'
        )
    runner_cfg = _resolve_cfg(entry)

    env_cfg_entry = spec.kwargs['env_cfg_entry_point']
    env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.max_iterations is not None:
        runner_cfg['runner']['max_iterations'] = args_cli.max_iterations
    if args_cli.resume:
        runner_cfg['runner']['resume'] = True
    if args_cli.load_run is not None:
        runner_cfg['runner']['load_run'] = args_cli.load_run
    if args_cli.checkpoint is not None:
        runner_cfg['runner']['load_checkpoint'] = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        runner_cfg['runner']['experiment_name'] = args_cli.experiment_name

    log_root = os.path.abspath(
        os.path.join('logs', 'dreamwaq', runner_cfg['runner']['experiment_name'])
    )
    log_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if runner_cfg['runner'].get('run_name'):
        log_dir += f"_{runner_cfg['runner']['run_name']}"
    log_dir = os.path.join(log_root, log_dir)
    env_cfg.log_dir = log_dir
    print(f'[INFO] Logging experiment in directory: {log_dir}')

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = IsaacLabDreamWaQWrapper(env, device=args_cli.device or 'cuda:0')

    runner = OnPolicyRunner(
        env, runner_cfg, log_dir=log_dir, device=args_cli.device or 'cuda:0'
    )
    runner.add_git_repo_to_log(__file__)

    if runner_cfg['runner'].get('resume'):
        resume_path = get_checkpoint_path(
            log_root,
            runner_cfg['runner']['load_run'],
            runner_cfg['runner']['load_checkpoint'],
        )
        print(f'[INFO] Loading checkpoint: {resume_path}')
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, 'params', 'env.yaml'), env_cfg)
    dump_yaml(os.path.join(log_dir, 'params', 'agent.yaml'), runner_cfg)

    runner.learn(
        num_learning_iterations=runner_cfg['runner']['max_iterations'],
        init_at_random_ep_len=True,
    )
    env.env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
