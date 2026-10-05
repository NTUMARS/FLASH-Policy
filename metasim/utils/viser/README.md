# TaskViserWrapper Guide

TaskViserWrapper adds real-time Viser visualization to `RLTaskEnv`.

## Key Features

- Sets up Viser visualization automatically
- Updates robot and object states in real time
- Renders only the first environment to avoid multi-environment complexity
- Transparently proxies all environment attributes and methods
- Handles visualization errors without interrupting training

## Usage

### 1. Basic Usage

```python
from metasim.task.registry import get_task_class
from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

# Create the environment
task_cls = get_task_class('reach_origin')
scenario = task_cls.scenario.update(
    robots=['franka'],
    simulator='mujoco',
    num_envs=1024,
    headless=False,  # Enable rendering for Viser
    cameras=[]
)

env = task_cls(scenario, device='cuda')

# Wrap the environment to enable visualization
viser_env = TaskViserWrapper(env, port=8080)

# Use the environment normally
obs = viser_env.reset()
for _ in range(100):
    actions = policy(obs)  # Your policy
    obs, reward, terminated, timeout, info = viser_env.step(actions)
    if terminated.any() or timeout.any():
        obs = viser_env.reset()

viser_env.close()
```

### 2. Integration with fast_td3

Enable visualization in fast_td3:

```bash
# Enable Viser visualization
python get_started/rl/fast_td3/1_fttd3.py --viser-port 8080

# Run without visualization (default)
python get_started/rl/fast_td3/1_fttd3.py
```

Example configuration:
```python
CONFIG = {
    "sim": "mujoco",
    "robots": ["franka"],
    "task": "reach_origin",
    "headless": True,
    "viser_port": 8080,  # Set to a value greater than 0 to enable visualization
    # ... other configuration
}
```

## How It Works

1. **During initialization**:
   - Creates a `ViserVisualizer` instance
   - Downloads the required URDF files
   - Visualizes all robots and objects in the scene
   - Configures camera controls

2. **At runtime**:
   - Updates the visualization after every `reset()` and `step()`
   - Extracts the state of the first environment
   - Updates the positions and orientations of all robots and objects

3. **Error handling**:
   - Visualization failures do not interrupt training
   - Import errors are handled silently

## Environment Attribute Proxying

TaskViserWrapper transparently proxies all environment attributes:

```python
wrapper = TaskViserWrapper(env)
print(wrapper.num_envs)      # Proxies env.num_envs
print(wrapper.num_actions)   # Proxies env.num_actions
print(wrapper.num_obs)       # Proxies env.num_obs
print(wrapper.action_space)  # Proxies env.action_space
```

## Technical Details

- Uses the same state-extraction logic as `viser_demo.py`
- Supports all simulator backends, including MuJoCo and Isaac Sim
- Renders only the first environment to preserve performance
- Handles indexing of multi-environment tensors automatically
