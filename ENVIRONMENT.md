# Environment configuration

Use one Python 3.11 environment for simulator examples, demonstration conversion,
training, evaluation, and tests. You may choose any environment name; the code
uses the active Python interpreter.

## Create an environment

Replace `YOUR_ENV_NAME` with your preferred name:

```bash
conda create --name YOUR_ENV_NAME python=3.11
conda activate YOUR_ENV_NAME
python -m pip install --upgrade pip
```

An equivalent Python 3.11 virtual environment also works. Run the remaining
repository commands from the repository root with your environment activated.

## Install the simulator and policy packages

```bash
python -m pip install -r requirements.txt
```

The root dependency file installs the repository in editable mode, its Isaac Sim
and MuJoCo extras, and the shared FLASH policy dependencies. Important version
constraints are:

| Component | Version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.7.0 with CUDA 12.8 |
| torchvision | 0.22.0 with CUDA 12.8 |
| Isaac Sim | 5.0.0 |
| Isaac Lab | source release v2.2.1, installed below |
| NumPy | >=1.26, <2 |
| Zarr | 2.18.3 |
| numcodecs | 0.13.0 |
| Weights & Biases | 0.17.9 for the combined simulator environment |

`pyproject.toml` defines the repository package and simulator extras;
`roboverse_learn/il/policies/flash/requirements.txt` defines the shared policy
packages. Zarr remains on version 2 for the dataset APIs used by this repository.
Some packages are unpinned, so these files are an installation specification,
rather than a complete lockfile of the environment used for the paper.
The root file constrains Weights & Biases to a version compatible with Isaac Sim
5.0.0's `sentry-sdk` dependency.

Isaac Sim requires an accessible NVIDIA GPU and a compatible NVIDIA driver,
including for headless camera rendering. Its pip installation also requires
GLIBC 2.34 or newer on Linux. See the [Isaac Lab v2.2.1 installation guide](https://isaac-sim.github.io/IsaacLab/v2.2.1/source/setup/installation/pip_installation.html)
for the supported platform and driver requirements.

## Install Isaac Lab

The Isaac Sim backend imports Isaac Lab. Install its v2.2.1 source release into
the same environment after installing the root requirements. Run these commands
from a directory where you want to keep the external dependency:

```bash
git clone --branch v2.2.1 --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install none
```

`none` installs the Isaac Lab extensions without its optional reinforcement
learning frameworks. On Ubuntu, install `cmake` and `build-essential` first if
they are unavailable. The upstream installation guide describes these system
dependencies. Keep the Isaac Lab checkout available because its extensions are
installed in editable mode. Return to this repository's root before continuing.

## Prepare assets and demonstrations

Follow the [RoboVerse asset instructions](https://roboverse.wiki/metasim/get_started/roboverse_data)
to obtain the robot, object, scene, and reference trajectory resources used by
your selected task. Asset paths in task configurations are relative to the
repository root, including `roboverse_data/assets/`. Some resources are downloaded
on demand; their first use requires network access.

The paper's Close Box, Pick Cube, and Stack Cube tasks use Isaac Sim. Its
Pick-Place Bowl and Open Drawer tasks use MuJoCo, with the exact identifiers
`libero_90.kitchen_scene1_put_the_black_bowl_on_the_plate` and
`libero_90.kitchen_scene1_open_bottom_drawer`, respectively. See the
[five-task reproduction guide](REPRODUCING.md) for the complete task mapping and
collection, conversion, training, and evaluation commands.
Original demonstrations, experiment outputs, and trained checkpoints are not
included in this source release. Prepare your own demonstrations and follow the
[task-specific instructions](REPRODUCING.md#select-a-task).

For the standalone 2D tasks, follow the separate
[CorridorPush and ForkReach dependency instructions](roboverse_learn/il/push2d/README.md).
Their environment and demonstration tools can run on CPU without Isaac Sim.

## Verify the installation

Check the interpreter, package versions, and CUDA visibility:

```bash
python -c "import sys, torch, zarr, numcodecs; print(sys.executable); print(torch.__version__); print(zarr.__version__, numcodecs.__version__); print(torch.cuda.is_available())"
```

The CUDA check must print `True` for Isaac Sim. If it prints `False`, check GPU
visibility and the NVIDIA driver in the process or container running the command.

Verify the Isaac Lab installation from its checkout:

```bash
python scripts/tutorials/00_sim/create_empty.py --headless
```

Then return to this repository and run a Stack Cube replay with the required
assets available:

```bash
python scripts/advanced/replay_demo.py \
    --task stack_cube --sim isaacsim --headless --stop-on-runout
```

Isaac Sim's first launch may download additional extensions and prompt for its
license agreement. Allow the initial setup to finish before evaluating the
simulator startup time.
