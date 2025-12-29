# DDT_Lab 使用教程

## Overview

这个项目参考的是[TemplateforIsaaclabprojects](https://github.com/isaac-sim/IsaacLabExtensionTemplate).项目使用Isaaclab作为训练环境，包含有Direct drive Technology的D1和Tita机器人。



## Installation
特别注意，本项目应该在**Isaalab release/v2.3.0**和**Isaacsim 5.1** 版本中运行，其他版本暂时还没有进行适配，可以使用专门提供适配本项目的docker，后期会放出来，但是文件较大有65.8G；也可以使用官方的docker，但是需要稍微手动配置一下环境。

这里给出一个安装官方环境的参考
```
# 要拉取最小的 Isaac Lab 容器，请运行:

docker pull nvcr.io/nvidia/isaac-lab:2.3.0
```
```
# 要运行带有交互式 bash 会话的 Isaac Lab 容器，请运行:

docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   nvcr.io/nvidia/isaac-lab:2.3.0
   ```
   ```
# 为了通过 X11 转发启用渲染，请运行:

xhost +
docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   nvcr.io/nvidia/isaac-lab:2.3.0
```
---
参照流程 [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
建议使用anaconda或者miniconda进行环境配置，以及使用pip进行Isaacsim的安装。为了方便起见，下面贴出对应的安装步骤，可以直接安装配置。

- python环境安装，使用python3.11
```
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install --upgrade pip
```
- 安装依赖
```
## Isaacsim 5.1安装
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```
```
## 安装torch和torchvison，一定要是这个版本
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```
针对不同的机器，需要根据不同的cuda版本进行安装。这里默认已经安装好显卡驱动了，终端中输`nvidia-smi`查看对应的cuda版本。如果查询到cuda版本为12.6，则在安装torch和torchvision时在终端中输入下面的指令：
```
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```
这边建议cuda版本不要太低，否则会因为安装对应版本的torch和torchvision而不能运行代码

安装完成后可输入验证安装
```
isaacsim
```
- **Isaaclab安装**

```
git clone git@github.com:isaac-sim/IsaacLab.git
git checkout release/2.3.0
git pull origin release/2.3.0
```
上述命令可以安装并切换Isaaclab到最新的版本，且请务必保证是这个版本。

```
./isaaclab.sh --install # or "./isaaclab.sh -i"
```
安装训练框架。为了节省空间，可以安装rsl-rl这一个库
```
./isaaclab.sh --install rsl_rl
```
安装安成后进行验证
```
# Option 1: Using the isaaclab.sh executable
# note: this works for both the bundled python and the virtual environment
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

# Option 2: Using python in your virtual environment
python scripts/tutorials/00_sim/create_empty.py
```
到此Isaaclab的环境安装完成了
## Trainning
DDT_lab提供了丰富的训练环境，首先需要安装好依赖。
```
cd ddt_lab
python -m pip install -e source/ddt_lab
```
查找对应的任务。
```
# use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python scripts/list_envs.py
```
运行对应的任务。
```
# use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
```
注意，应根据上一步列出的任务名运行，否则会报错。同时不要使用有Play的任务进行训练，这是因为这个参数比较小，用于推理，不适合用于训练。

如果在服务器上进行训练，且没有安装好图形库，可以使用**headless**和**video**两个参数
```
python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME> --headless --video
```




## dummy agents:
这里使用没有策略的智能体来验证环境是否可以运行


- Zero-action agent （零动作代理）

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/zero_agent.py --task=<TASK_NAME>
    ```
- Random-action agent （随机动作代理）

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/random_agent.py --task=<TASK_NAME>
    ```

## Evaluation

为了验证训练的效果，项目默认使用tensorboard进行参数分析。在终端中确认已经安装好tensorboard后进入ddt_lab路径输入下列指令：
```
tensorboard --logdir ./logs
```
tensorboard会自动加载对应的训练参数，打开http://localhost:6006/便可以查看训练结果。

除了数据验证，项目中也写了推理的脚本，进入ddt_lab路径，在终端中输入：
```
python scripts/rsl_rl/play.py \
    --task DDT-Velocity-Flat-Tita-Play-v0 \
    --num_envs 50 \
    --checkpoint "./logs/rsl_rl/<path-to-your-file>"
```

其中，task输入对应的训练任务的play版本，load_run输入对应的时间日期，checkpoint输入对应的权重文件,num_envs是推理的环境数量。

对于服务器推理，可以使用`--headless`和`--video`的参数进行推理。
```
python scripts/rsl_rl/play.py \
    --task DDT-Velocity-Flat-Tita-Play-v0 \
    --num_envs 50 \
    --checkpoint "./logs/rsl_rl/<path-to-your-file>" \
    --headless \
    --video
```

推理完成后，在**logs/rsl_rl/\<data-data-path\>/video/play**中会有对应的推理视频，同时每次推理的onnx文件也都会保存在**logs/rsl_rl/\<robot\>/<data-data-path\>/exported/policy.onnx**

### 真机部署

真机部署请参考[ddt_sim2sim2real](https://github.com/DDTRobot/tita_rl_sim2sim2real)
---

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/ddt_lab/ddt_lab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

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
