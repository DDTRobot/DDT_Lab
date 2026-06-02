# ddt_lab — NP3O Locomotion for Wheel-Legged Robots

Locomotion training for the **D1** (quadruped with wheels) and **Tita**
(wheel-legged biped) robots, using **NP3O** (BarlowTwins-augmented
constrained PPO) built on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

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
The conda-based install is recommended:

```bash
# After cloning IsaacLab:
conda activate env_isaaclab
```

### 2. Clone this repo (outside the IsaacLab directory)

```bash
git clone https://github.com/DDTRobot/DDT_Lab/tree/np3o ddt_lab
cd ddt_lab
```

### 3. Get robot URDF models

URDF paths are controlled by `DDT_MODEL_DIR` in
`source/ddt_lab/ddt_lab/assets/ddt_robot.py`:

```python
# source/ddt_lab/ddt_lab/assets/ddt_robot.py (line ~28)
DDT_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../ddt_ros2_control/urdfs")
)
```

This resolves to `<ddt_lab_root>/ddt_ros2_control/urdfs/` at runtime.

**Default — clone `ddt_ros2_control` inside `ddt_lab`:**

```bash
# Run from the ddt_lab directory
git clone https://github.com/DDTRobot/ddt_ros2_control.git ddt_ros2_control
```

Required layout:

```
ddt_lab/
├── ddt_ros2_control/
│   └── urdfs/
│       ├── d1_description/urdf/robot.urdf
│       ├── tita_description/urdf/robot.urdf
│       └── ...
├── source/
└── scripts/
```

**Custom path** — edit `DDT_MODEL_DIR` in `ddt_robot.py` directly:

```python
DDT_MODEL_DIR = "/absolute/path/to/your/urdfs"
```

### 4. Install ddt_lab in editable mode

```bash
# Use the same Python that has Isaac Lab installed
python -m pip install -e source/ddt_lab
```

### 5. Verify installation

```bash
# Should print 8 DDT-* tasks
python scripts/list_envs.py
```

Expected output:

```
+----------------------------------+-----------------------------------------+
| Task Name                        | Config                                  |
+----------------------------------+-----------------------------------------+
| DDT-Velocity-Flat-D1-v0          | D1FlatEnvCfg (NP3O, history=10)         |
| DDT-Velocity-Rough-D1-v0         | D1RoughEnvCfg (NP3O, history=10)        |
| DDT-Velocity-Flat-Tita-v0        | TitaFlatEnvCfg                          |
| DDT-Velocity-Rough-Tita-v0       | TitaRoughEnvCfg                         |
| DDT-Velocity-Flat-D1-DreamWaQ-v0  | D1FlatDreamWaQEnvCfg (DreamWaQ, hist=5)|
| DDT-Velocity-Rough-D1-DreamWaQ-v0 | D1RoughDreamWaQEnvCfg (DreamWaQ, hist=5)|
+----------------------------------+-----------------------------------------+
```

`*-Play-v0` variants exist for all tasks above.

---

## Training

### NP3O (BarlowTwins-PPO)

```bash
# D1 — flat ground
python scripts/np3o/train.py --task=DDT-Velocity-Flat-D1-v0 \
    --num_envs 4096 --headless

# D1 — rough terrain (trimesh, terrain curriculum)
python scripts/np3o/train.py --task=DDT-Velocity-Rough-D1-v0 \
    --num_envs 4096 --headless

# Tita — flat ground
python scripts/np3o/train.py --task=DDT-Velocity-Flat-Tita-v0 \
    --num_envs 4096 --headless
```

### DreamWaQ (CeNet VAE + PPO)

```bash
# D1 — flat ground  (history=5 frames, matches reference num_obs_hist=5)
python scripts/dreamwaq/train.py --task=DDT-Velocity-Flat-D1-DreamWaQ-v0 \
    --num_envs 4096 --headless

# D1 — rough terrain
python scripts/dreamwaq/train.py --task=DDT-Velocity-Rough-D1-DreamWaQ-v0 \
    --num_envs 4096 --headless
```

Logs go to `logs/dreamwaq/<experiment_name>/` (separate from NP3O's `logs/np3o/`).

### Common flags

| Flag | Default | Description |
|---|---|---|
| `--num_envs` | (from cfg) | Number of parallel environments |
| `--max_iterations` | (from cfg) | Override total training iterations |
| `--headless` | False | Run without rendering (recommended for training) |
| `--seed` | None | Random seed |
| `--device` | `cuda:0` | Training device |
| `--experiment_name` | (from cfg) | Override the log directory name |

### Logs

Checkpoints and TensorBoard events are written to:

```
logs/np3o/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/
├── model_<iter>.pt      # policy checkpoint
├── params/
│   ├── env.yaml         # environment config snapshot
│   └── agent.yaml       # algorithm config snapshot
├── git/
│   ├── ddt_lab.diff     # git diff at training start
│   └── rsl_rl.diff
└── events.out.tfevents… # TensorBoard
```

### Monitor training

```bash
tensorboard --logdir logs/np3o
```

Key metrics to watch:

| Metric | Healthy sign |
|---|---|
| `Train/mean_reward` | Steadily increasing |
| `Policy/mean_noise_std` | Gradually decreases from 1.0 → ~0.5, doesn't collapse to 0 |
| `Loss/surrogate` | Negative, small magnitude |
| `Loss/mean_imitation_loss` | Decreasing (BarlowTwins SSL converging) |
| `Mean episode cost_*` | Decreasing toward 0 |

---

## Resume training

```bash
python scripts/np3o/train.py --task=DDT-Velocity-Flat-D1-v0 \
    --num_envs 4096 --headless \
    --resume \
    --load_run ".*" \
    --load_checkpoint "model_.*\.pt"
```

---

## Play / Evaluate

### NP3O

```bash
python scripts/np3o/play.py --task=DDT-Velocity-Flat-D1-Play-v0
python scripts/np3o/play.py --task=DDT-Velocity-Flat-D1-Play-v0 --checkpoint /path/to/model_5000.pt
python scripts/np3o/play.py --task=DDT-Velocity-Flat-D1-Play-v0 --export_policy --export_dir /tmp/d1_deploy
```

### DreamWaQ

```bash
python scripts/dreamwaq/play.py --task=DDT-Velocity-Flat-D1-DreamWaQ-Play-v0
python scripts/dreamwaq/play.py --task=DDT-Velocity-Flat-D1-DreamWaQ-Play-v0 --checkpoint /path/to/model_5000.pt
python scripts/dreamwaq/play.py --task=DDT-Velocity-Flat-D1-DreamWaQ-Play-v0 --export_policy
```

Exported policy ONNX inputs (both algorithms):

| Input | Shape | Description |
|---|---|---|
| `nn_input0` | `(1, n_proprio)` | Current proprio observation |
| `nn_input1` | `(1, history_len, n_proprio)` | Full history buffer (`history_len=10` for NP3O, `=5` for DreamWaQ) |

Output:

| Output | Shape | Description |
|---|---|---|
| `nn_output` | `(1, n_actions)` | Deterministic action mean |

---

## Sanity-check environments

These scripts require no RL libraries — useful to verify env setup:

```bash
python scripts/zero_agent.py --task=DDT-Velocity-Flat-D1-v0
python scripts/random_agent.py --task=DDT-Velocity-Flat-D1-v0
```

---

## Available robots & tasks

| Robot | Description | Flat task | Rough task |
|---|---|---|---|
| **D1** | Quadruped with wheel feet | `DDT-Velocity-Flat-D1-v0` | `DDT-Velocity-Rough-D1-v0` |
| **Tita** | Wheel-legged biped | `DDT-Velocity-Flat-Tita-v0` | `DDT-Velocity-Rough-Tita-v0` |

`*-Play-v0` variants use 50 envs, zero commands, no domain randomization — for visualization.

---

## Algorithm overview (NP3O)

NP3O extends PPO with:

- **BarlowTwins SSL** — a self-supervised history encoder learns to predict
  velocity from proprio history, giving the actor implicit state estimation
  without extra privileged obs at inference time.
- **Constrained optimization** — optional cost terms (joint limits, torque
  limits, etc.) are enforced via a Lagrangian multiplier that grows during
  training.
- **Privileged critic** — critic sees physical parameters (contact state,
  kp/kd randomization factors) invisible to the policy, improving value
  estimates during training only.

Key config files:

```
source/ddt_lab/ddt_lab/
├── algorithms/np3o/           # NP3O algorithm, BarlowTwins actor-critic, runner
├── managers/cost_manager.py   # CostManager + CostTermCfg
└── tasks/manager_based/locomotion/
    ├── mdp/                   # reward / cost / obs functions
    └── robots/
        ├── d1/
        │   ├── rough_env_cfg.py    # full D1 env config (rewards, costs, domain rand)
        │   ├── flat_env_cfg.py     # D1 flat override (plane terrain, no height scan)
        │   └── agents/np3o_cfg.py  # D1-specific training hyperparameters
        └── tita/
            ├── rough_env_cfg.py
            ├── flat_env_cfg.py
            └── agents/np3o_cfg.py
```

---

## Adding a new cost term

```python
# rough_env_cfg.py — add to CostsCfg
from ddt_lab.managers import CostTermCfg

@configclass
class CostsCfg:
    pos_limit = CostTermCfg(
        func=mdp.joint_pos_limit,
        scale=1.0, d_value=0.0, k_value=0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[...])},
    )
    # Add more terms here — CostManager auto-detects them
```

Remove the `costs` field entirely to fall back to PPO + BarlowTwins (no constraints).

---

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

**`FileNotFoundError` / URDF not found at startup**

`ddt_robot.py` looks for URDFs at `<ddt_lab_root>/ddt_ros2_control/urdfs/`.
Make sure `ddt_ros2_control` is cloned inside `ddt_lab` (step 3):

```bash
git clone https://github.com/DDTRobot/ddt_ros2_control.git ddt_ros2_control
ls ddt_ros2_control/urdfs/    # should list d1_description/, tita_description/, etc.
```

If the URDF directory is somewhere else, edit `DDT_MODEL_DIR` directly in
`source/ddt_lab/ddt_lab/assets/ddt_robot.py`.

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/ddt_lab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
