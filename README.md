# ddt_lab — Multi-Algorithm Locomotion for Wheel-Legged Robots

Locomotion training for the **D1** (quadruped with wheels) robot, supporting
three RL algorithms built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/):

| Algorithm | Description | Script |
|---|---|---|
| **RSL-RL (PPO)** | Stock PPO baseline | `scripts/rsl_rl/` |
| **NP3O** | BarlowTwins-augmented constrained PPO | `scripts/np3o/` |
| **DreamWaQ** | CeNet VAE + PPO (implicit terrain estimation) | `scripts/dreamwaq/` |

---

## Prerequisites

| Dependency | Version |
|---|---|
| NVIDIA Isaac Sim | 5.1 |
| Isaac Lab | [v2.3.0](https://isaac-sim.github.io/IsaacLab/v2.3.0/index.html) |
| Python | 3.11 (bundled with Isaac Sim) |
| CUDA | 12.x |

---

## Installation

### 1. Install Isaac Lab

Follow the [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

```bash
conda activate env_isaaclab
```

### 2. Clone this repo

```bash
git clone https://github.com/DDTRobot/DDT_Lab ddt_lab
cd ddt_lab
```

### 3. Get robot URDF models

URDF paths are controlled by `DDT_MODEL_DIR` in
`source/ddt_lab/ddt_lab/assets/ddt_robot.py`.

**Default — clone `ddt_ros2_control` inside `ddt_lab`:**

```bash
git clone https://github.com/DDTRobot/ddt_ros2_control.git ddt_ros2_control
```

Required layout:

```
ddt_lab/
├── ddt_ros2_control/
│   └── urdfs/
│       ├── d1_description/urdf/robot.urdf
│       └── ...
├── source/
└── scripts/
```

**Custom path** — edit `DDT_MODEL_DIR` in `ddt_robot.py` directly.

### 4. Install ddt_lab

```bash
python -m pip install -e source/ddt_lab
```

### 5. Verify

```bash
# Should print all DDT-* tasks
python scripts/list_envs.py
```

---

## Available Tasks

| Algorithm | Flat train | Rough train | Flat play | Rough play |
|---|---|---|---|---|
| **RSL-RL** | `DDT-Velocity-Flat-D1-v0` | `DDT-Velocity-Rough-D1-v0` | `…-Play-v0` | `…-Play-v0` |
| **NP3O** | `DDT-Velocity-Flat-D1-NP3O-v0` | `DDT-Velocity-Rough-D1-NP3O-v0` | `…-Play-v0` | `…-Play-v0` |
| **DreamWaQ** | `DDT-Velocity-Flat-D1-DreamWaQ-v0` | `DDT-Velocity-Rough-D1-DreamWaQ-v0` | `…-Play-v0` | `…-Play-v0` |

---

## Training

### RSL-RL (stock PPO)

```bash
python scripts/rsl_rl/train.py --task=DDT-Velocity-Flat-D1-v0 \
    --num_envs 4096 --headless

python scripts/rsl_rl/train.py --task=DDT-Velocity-Rough-D1-v0 \
    --num_envs 4096 --headless
```

Logs → `logs/rsl_rl/<experiment_name>/<timestamp>/`

### NP3O (BarlowTwins-PPO)

```bash
python scripts/np3o/train.py --task=DDT-Velocity-Flat-D1-NP3O-v0 \
    --num_envs 4096 --headless

python scripts/np3o/train.py --task=DDT-Velocity-Rough-D1-NP3O-v0 \
    --num_envs 4096 --headless
```

Logs → `logs/np3o/<experiment_name>/<timestamp>/`

### DreamWaQ (CeNet VAE + PPO)

```bash
python scripts/dreamwaq/train.py --task=DDT-Velocity-Rough-D1-DreamWaQ-v0 \
    --num_envs 4096 --headless

python scripts/dreamwaq/train.py --task=DDT-Velocity-Flat-D1-DreamWaQ-v0 \
    --num_envs 4096 --headless
```

Logs → `logs/dreamwaq/<experiment_name>/<timestamp>/`

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--num_envs` | (from cfg) | Number of parallel environments |
| `--max_iterations` | (from cfg) | Override total training iterations |
| `--headless` | False | Disable rendering (recommended for training) |
| `--seed` | None | Random seed |
| `--device` | `cuda:0` | Training device |
| `--experiment_name` | (from cfg) | Override log directory name (NP3O / DreamWaQ) |

### Resume training

```bash
# NP3O / DreamWaQ
python scripts/np3o/train.py --task=DDT-Velocity-Rough-D1-NP3O-v0 \
    --num_envs 4096 --headless \
    --resume \
    --load_run ".*" \
    --checkpoint "model_.*\.pt"

# RSL-RL
python scripts/rsl_rl/train.py --task=DDT-Velocity-Rough-D1-v0 \
    --num_envs 4096 --headless \
    --resume --load_run ".*" --checkpoint "model_.*\.pt"
```

### Monitor training

```bash
tensorboard --logdir logs/np3o       # NP3O
tensorboard --logdir logs/dreamwaq   # DreamWaQ
tensorboard --logdir logs/rsl_rl     # RSL-RL
```

Key metrics (NP3O / DreamWaQ):

| Metric | Healthy sign |
|---|---|
| `Train/mean_reward` | Steadily increasing |
| `Policy/mean_noise_std` | Gradually decreases from 1.0 → ~0.5 |
| `Loss/surrogate` | Negative, small magnitude |
| `Loss/vae` (DreamWaQ) | Decreasing (CeNet VAE converging) |
| `Loss/mean_imitation_loss` (NP3O) | Decreasing (BarlowTwins SSL converging) |

---

## Play / Evaluate

### RSL-RL

```bash
python scripts/rsl_rl/play.py --task=DDT-Velocity-Flat-D1-Play-v0
python scripts/rsl_rl/play.py --task=DDT-Velocity-Flat-D1-Play-v0 \
    --checkpoint /path/to/model_5000.pt
```

### NP3O

```bash
python scripts/np3o/play.py --task=DDT-Velocity-Flat-D1-NP3O-Play-v0
python scripts/np3o/play.py --task=DDT-Velocity-Flat-D1-NP3O-Play-v0 \
    --checkpoint /path/to/model_5000.pt
python scripts/np3o/play.py --task=DDT-Velocity-Flat-D1-NP3O-Play-v0 \
    --export_policy --export_dir /tmp/d1_deploy
```

### DreamWaQ

```bash
python scripts/dreamwaq/play.py --task=DDT-Velocity-Flat-D1-DreamWaQ-Play-v0
python scripts/dreamwaq/play.py --task=DDT-Velocity-Flat-D1-DreamWaQ-Play-v0 \
    --checkpoint /path/to/model_5000.pt
python scripts/dreamwaq/play.py --task=DDT-Velocity-Flat-D1-DreamWaQ-Play-v0 \
    --export_policy
```

### Exported policy format (NP3O / DreamWaQ)

| | NP3O | DreamWaQ |
|---|---|---|
| Input name | `nn_input` | `nn_input` |
| Input shape | `(1, 10, n_proprio)` | `(1, 5, n_proprio)` |
| Output name | `nn_output` | `nn_output` |
| Output shape | `(1, n_actions)` | `(1, n_actions)` |

The last slice `[:, -1, :]` of the history buffer is used as the current frame internally.

---

## Sanity-check environments

```bash
python scripts/zero_agent.py   --task=DDT-Velocity-Flat-D1-v0
python scripts/random_agent.py --task=DDT-Velocity-Flat-D1-v0
```

---

## Project structure

```
ddt_lab/
├── scripts/
│   ├── rsl_rl/          # train.py, play.py, cli_args.py
│   ├── np3o/            # train.py, play.py
│   └── dreamwaq/        # train.py, play.py
└── source/ddt_lab/ddt_lab/
    ├── algorithms/
    │   ├── np3o/        # ActorCriticBarlowTwins, NP3O, runner, rollout_storage
    │   └── dreamwaq/    # CeNet, ActorCriticDreamWaQ, PPO_DreamWaQ, runner
    ├── managers/
    │   └── cost_manager.py
    └── tasks/manager_based/locomotion/
        ├── mdp/         # rewards, costs, observations, events, curriculums
        └── robots/d1/
            ├── base_env_cfg.py        # shared MDP base (SceneCfg, RewardsCfg, …)
            ├── rsl_rl/
            │   ├── rough_env_cfg.py   # D1RoughEnvCfg_PLAY
            │   ├── flat_env_cfg.py    # D1FlatEnvCfg, D1FlatEnvCfg_PLAY
            │   └── agents/rsl_rl_ppo_cfg.py
            ├── np3o/
            │   ├── rough_env_cfg.py   # PrivilegedObsCfg + CostsCfg + NP3O cfgs
            │   ├── flat_env_cfg.py    # D1FlatNP3OEnvCfg
            │   └── agents/np3o_cfg.py
            └── dreamwaq/
                ├── rough_env_cfg.py   # PrivilegedObsCfg + DreamWaQ cfgs
                ├── flat_env_cfg.py    # D1FlatDreamWaQEnvCfg
                └── agents/dreamwaq_cfg.py
```

---

## Algorithm overview

### RSL-RL (stock PPO)

Standard on-policy PPO using Isaac Lab's `RslRlVecEnvWrapper`. Policy obs is
flat 2-D `(B, D)`. No history, no privileged observations at inference.

### NP3O

Extends PPO with:
- **BarlowTwins SSL** — self-supervised history encoder predicts velocity from
  10-frame proprio history; actor uses the learned code without privileged obs
  at inference.
- **Constrained optimization** — cost terms (joint limits, torque limits) are
  enforced via a Lagrangian multiplier. Remove `CostsCfg` from the env cfg to
  fall back to unconstrained BarlowTwins-PPO.
- **Privileged critic** — critic sees contact state, actuator gain factors,
  body mass / CoM offsets, and applied forces; policy cannot.

### DreamWaQ

Extends PPO with:
- **CeNet (Context Estimation Network)** — a VAE that maps 5-frame proprio
  history to an explicit velocity code (`code_vel`, 3-D) and an implicit latent
  code (`code_latent`, 16-D). The actor concatenates this 19-D code with the
  current frame.
- **Dual optimizer** — RL loss and VAE loss are optimized separately; CeNet
  gradients do not flow through the RL objective.
- No cost constraints.

---

## Adding a new cost term (NP3O)

```python
# np3o/rough_env_cfg.py — add to CostsCfg
@configclass
class CostsCfg:
    my_limit = CostTermCfg(
        func=mdp.joint_pos_limit,
        scale=1.0, d_value=0.0, k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[...])},
    )
```

---

## Code formatting

```bash
pip install pre-commit
pre-commit run --all-files
```

---

## Troubleshooting

**`FileNotFoundError` / URDF not found**

Make sure `ddt_ros2_control` is cloned inside `ddt_lab` or update `DDT_MODEL_DIR`
in `source/ddt_lab/ddt_lab/assets/ddt_robot.py`.

**Pylance missing extension indexing**

Add to `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/ddt_lab"
    ]
}
```
