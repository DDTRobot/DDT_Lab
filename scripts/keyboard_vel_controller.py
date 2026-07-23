# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard velocity controller for Isaac Lab play scripts.

Wraps Se2Keyboard and writes commands into the env's command manager each step.

Numpad layout (NumLock ON):
  [7 左转] [8 前进] [9 右转]
  [4 左移]          [6 右移]
            [2 后退]
  Z=左转  X=右转  L/Space=停止

Usage:
    controller = VelocityKeyboardController(env_cfg)
    # in loop:
    controller.apply_to_env(env)
"""

from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg


class VelocityKeyboardController:
    def __init__(self, env_cfg, sim_device: str = "cuda:0"):
        ranges = env_cfg.commands.base_velocity.ranges
        cfg = Se2KeyboardCfg(
            v_x_sensitivity=ranges.lin_vel_x[1],
            v_y_sensitivity=ranges.lin_vel_y[1],
            omega_z_sensitivity=ranges.ang_vel_z[1],
            sim_device=sim_device,
        )
        self._kb = Se2Keyboard(cfg)
        self._kb.reset()
        print(self._kb)

    def apply_to_env(self, env, command_name: str = "base_velocity") -> None:
        cmd = self._kb.advance()
        term = env.env.unwrapped.command_manager._terms[command_name]
        term.vel_command_b[:, 0] = cmd[0]
        term.vel_command_b[:, 1] = cmd[1]
        term.vel_command_b[:, 2] = cmd[2]

    def close(self) -> None:
        pass
