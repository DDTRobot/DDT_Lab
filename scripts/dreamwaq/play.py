# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate a DreamWaQ checkpoint."""

import argparse
import importlib
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Play a DreamWaQ checkpoint.')
parser.add_argument('--task', type=str, required=True, help='Gym task ID.')
parser.add_argument('--num_envs', type=int, default=50)
parser.add_argument('--checkpoint', type=str, default=None, help='Absolute path to model_*.pt.')
parser.add_argument('--load_run', type=str, default='.*')
parser.add_argument('--load_checkpoint', type=str, default=r'model_.*\.pt')
parser.add_argument('--export_policy', action='store_true',
                    help='Export JIT + ONNX and exit without rollout.')
parser.add_argument('--export_dir', type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import ddt_lab.tasks  # noqa: F401
import gymnasium as gym
import torch
from isaaclab_tasks.utils import get_checkpoint_path

from ddt_lab.algorithms.dreamwaq import IsaacLabDreamWaQWrapper, OnPolicyRunner


def _resolve_cfg(entry_point: str) -> dict:
    module_name, attr = entry_point.split(':')
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if callable(obj) else obj


def main():
    spec = gym.spec(args_cli.task)
    runner_cfg = _resolve_cfg(spec.kwargs['dreamwaq_cfg_entry_point'])

    env_cfg_entry = spec.kwargs['env_cfg_entry_point']
    env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabDreamWaQWrapper(env, device=args_cli.device or 'cuda:0')

    runner = OnPolicyRunner(
        env, runner_cfg, log_dir=None, device=args_cli.device or 'cuda:0'
    )

    if args_cli.checkpoint is not None:
        ckpt = args_cli.checkpoint
    else:
        log_root = os.path.abspath(
            os.path.join('logs', 'dreamwaq', runner_cfg['runner']['experiment_name'])
        )
        ckpt = get_checkpoint_path(log_root, args_cli.load_run, args_cli.load_checkpoint)
    print(f'[INFO] loading checkpoint: {ckpt}')
    runner.load(ckpt, load_optimizer=False)

    export_dir = args_cli.export_dir or os.path.join(os.path.dirname(ckpt), 'exported')
    runner.alg.actor_critic.save_torch_jit_policy(export_dir, args_cli.device or 'cuda:0')

    if args_cli.export_policy:
        env.env.close()
        return

    policy = runner.get_inference_policy(args_cli.device or 'cuda:0')
    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
    env.env.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
