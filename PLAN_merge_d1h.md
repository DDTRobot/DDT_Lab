# Plan: Merge d1h Training Content from d1h_Lab-np30 into DDT_Lab_D1

## Current State Analysis

### Source: `d1h_Lab-np30` (has d1h training content)
- Full Isaac Lab + NP3O training setup for d1h bipedal robot
- Has `ddt_ros2_control/` with d1h-specific URDFs, controllers, hardware bridges
- Has complete `scripts/np3o/train.py` and `play.py` with d1h training config
- Has complete `source/ddt_lab/` with NP3O algorithm, d1h robot assets, cost manager, tasks

### Target: `DDT_Lab_D1`
- Currently on branch `np3o`
- Has a minimal setup: basic `scripts/np3o/train.py` (placeholder), `play.py`, skeleton `source/ddt_lab/`
- Missing `ddt_ros2_control/` entirely
- Missing d1h-specific robot configs in assets, tasks, managers

### Reference: `DDT_Lab-np3o` (also has d1h content, very similar to d1h_Lab-np30)
- Nearly identical to d1h_Lab-np30 in structure
- Differs in train.py (different env config: `num_envs=4096` vs `num_envs=6144` in d1h_Lab-np30)
- DDT_Lab-np3o has simulation/gazebo_bridge while d1h_Lab-np30 has simulation/webots_bridge

## What Needs to Be Merged

### 1. `scripts/np3o/train.py` - **OVERWRITE**
- DDT_Lab_D1 has a placeholder (just prints messages)
- Replace with d1h_Lab-np30's full training script which includes:
  - d1h locomotion task config
  - NP3O algorithm setup
  - Domain randomization params
  - Command-line arg parsing for headless/play modes

### 2. `scripts/np3o/play.py` - **OVERWRITE**
- DDT_Lab_D1 has a placeholder
- Replace with d1h_Lab-np30's play script

### 3. `source/ddt_lab/ddt_lab/assets/ddt_robot.py` - **OVERWRITE**
- DDT_Lab_D1 has a placeholder (empty class)
- Replace with d1h_Lab-np30's full d1h robot configuration:
  - D1HRobotCfg with all joint/body configs
  - D1H_NUM_ACTIONS, D1H_NUM_OBS, D1H_NUM_ASSETS

### 4. `source/ddt_lab/ddt_lab/algorithms/np3o/__init__.py` - **CREATE**
- DDT_Lab_D1 doesn't have this file
- NP3O algorithm initialization

### 5. `source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py` - **OVERWRITE**
- DDT_Lab_D1 has a minimal version
- Replace with d1h_Lab-np30's full version with D1H-specific actor/critic networks

### 6. `source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py` - **OVERWRITE**
- DDT_Lab_D1 has a minimal version
- Replace with d1h_Lab-np30's full wrapper

### 7. `source/ddt_lab/ddt_lab/tasks/__init__.py` - **OVERWRITE**
- DDT_Lab_D1 has empty file
- Replace with d1h_Lab-np30's task registry

### 8. `source/ddt_lab/ddt_lab/managers/cost_manager.py` - **OVERWRITE**
- DDT_Lab_D1 has empty file
- Replace with d1h_Lab-np30's cost manager

### 9. `source/ddt_lab/config/extension.toml` - **OVERWRITE**
- DDT_Lab_D1 has minimal config
- Replace with d1h_Lab-np30's full extension config

### 10. `ddt_ros2_control/` - **COPY ENTIRE DIRECTORY**
- DDT_Lab_D1 doesn't have this
- Copy from d1h_Lab-np30 (includes urdfs/d1h_description, hardware bridges, controllers, simulation bridges)

### 11. `.env` - **OVERWRITE**
- DDT_Lab_D1 likely has none or different
- Replace with d1h_Lab-np30's .env (ISAAC_LAB_PATH, etc.)

### Files that DON'T need changes (already correct or identical):
- `source/ddt_lab/ddt_lab/__init__.py` - DDT_Lab_D1's version is fine
- `source/ddt_lab/ddt_lab/algorithms/__init__.py` - already identical
- `source/ddt_lab/ddt_lab/managers/__init__.py` - already identical
- `source/ddt_lab/setup.py` and `pyproject.toml` - already identical
- `source/ddt_lab/ddt_lab/ui_extension_example.py` - already identical
- `scripts/list_envs.py`, `scripts/random_agent.py`, `scripts/zero_agent.py` - already identical

## Execution Order

1. Copy `ddt_ros2_control/` directory from d1h_Lab-np30 to DDT_Lab_D1
2. Overwrite `scripts/np3o/train.py`
3. Overwrite `scripts/np3o/play.py`
4. Overwrite `source/ddt_lab/ddt_lab/assets/ddt_robot.py`
5. Create `source/ddt_lab/ddt_lab/algorithms/np3o/__init__.py`
6. Overwrite `source/ddt_lab/ddt_lab/algorithms/np3o/actor_critic.py`
7. Overwrite `source/ddt_lab/ddt_lab/algorithms/np3o/wrapper.py`
8. Overwrite `source/ddt_lab/ddt_lab/tasks/__init__.py`
9. Overwrite `source/ddt_lab/ddt_lab/managers/cost_manager.py`
10. Overwrite `source/ddt_lab/config/extension.toml`
11. Overwrite `.env`
