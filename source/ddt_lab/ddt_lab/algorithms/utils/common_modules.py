# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MLP factories used by the NP3O actor-critic.

Verbatim port of the subset of ``LocomotionWithNP3O/modules/common_modules.py``
that NP3O actually depends on (``get_activation``, ``mlp_factory``,
``mlp_batchnorm_factory``). Other reference helpers (RNN encoders, VAEs, etc.)
are intentionally dropped — NP3O does not import them.
"""

import torch.nn as nn


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name in ("relu", "crelu"):
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


def mlp_factory(activation, input_dims, out_dims, hidden_dims, last_act=False):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0]))
    layers.append(activation)
    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(activation)
    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
    if last_act:
        layers.append(activation)
    return layers


def mlp_batchnorm_factory(activation, input_dims, out_dims, hidden_dims, last_act=False, bias=True):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0], bias=bias))
    layers.append(nn.BatchNorm1d(hidden_dims[0]))
    layers.append(activation)
    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias))
        layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        layers.append(activation)
    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims, bias=bias))
    if last_act:
        layers.append(activation)
    return layers
