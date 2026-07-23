# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate a checkpoint trained with NP3O."""

import argparse
import importlib
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a checkpoint trained with NP3O.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of envs for playback.")
parser.add_argument("--checkpoint", type=str, default=None, help="Absolute checkpoint path (overrides auto-resolve).")
parser.add_argument("--load_run", type=str, default=".*", help="Run dir regex when --checkpoint is omitted.")
parser.add_argument("--load_checkpoint", type=str, default=r"model_.*\.pt", help="Checkpoint filename regex.")
parser.add_argument(
    "--export_policy",
    action="store_true",
    help="Export the loaded policy as TorchScript + ONNX next to the checkpoint and exit.",
)
parser.add_argument(
    "--export_dir",
    type=str,
    default=None,
    help="Override export directory (defaults to <checkpoint_dir>/exported).",
)
parser.add_argument(
    "--keyboard",
    action="store_true",
    default=False,
    help="Enable keyboard control (W/S=fwd/bwd  A/D=strafe  Q/E=turn  Space=stop).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import ddt_lab.tasks  # noqa: F401  -- registers tasks
import gymnasium as gym
import torch
from ddt_lab.algorithms.np3o import IsaacLabNP3OWrapper, OnConstraintPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path


def _resolve_runner_cfg(entry_point: str) -> dict:
    module_name, attr = entry_point.split(":")
    obj = getattr(importlib.import_module(module_name), attr)
    cfg = obj() if callable(obj) else obj
    if not isinstance(cfg, dict):
        raise TypeError(f"NP3O cfg entry point '{entry_point}' must return a dict")
    return cfg


def main():
    spec = gym.spec(args_cli.task)
    runner_cfg = _resolve_runner_cfg(spec.kwargs["np3o_cfg_entry_point"])
    env_cfg_entry = spec.kwargs["env_cfg_entry_point"]
    env_cfg = env_cfg_entry() if callable(env_cfg_entry) else env_cfg_entry
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.heading_command = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabNP3OWrapper(env, device=args_cli.device or "cuda:0")

    runner = OnConstraintPolicyRunner(env, runner_cfg, log_dir=None, device=args_cli.device or "cuda:0")

    log_root = os.path.abspath(os.path.join("logs", "np3o", runner_cfg["runner"]["experiment_name"]))
    ckpt_key = args_cli.checkpoint if args_cli.checkpoint is not None else args_cli.load_checkpoint
    ckpt = get_checkpoint_path(log_root, args_cli.load_run, ckpt_key)
    print(f"[INFO] loading checkpoint: {ckpt}")
    runner.load(ckpt, load_optimizer=False)

    # Always export the JIT/ONNX policy next to the checkpoint (matches the
    # pre-NP3O scripts/rsl_rl/play.py behavior). Skip on --export_policy off
    # would be a one-line guard; we keep it on by default since it's cheap.
    export_dir = args_cli.export_dir or os.path.join(os.path.dirname(ckpt), "exported")
    runner.alg.actor_critic.save_torch_jit_policy(export_dir, args_cli.device or "cuda:0")
    if args_cli.export_policy:
        # --export_policy explicitly requested: skip rollout, just export.
        env.env.close()
        return

    policy = runner.get_inference_policy(args_cli.device or "cuda:0")
    obs = env.get_observations()

    keyboard = None
    if args_cli.keyboard:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from keyboard_vel_controller import VelocityKeyboardController

        keyboard = VelocityKeyboardController(env_cfg, sim_device=args_cli.device or "cuda:0")

    while simulation_app.is_running():
        if keyboard is not None:
            keyboard.apply_to_env(env)
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, _, _ = env.step(actions)

    env.env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
