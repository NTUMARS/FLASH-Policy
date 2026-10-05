# Reproducing the five simulation tasks

This guide covers the five simulation tasks reported in the FLASH paper: three
Isaac Sim tasks and two MuJoCo tasks. Run commands from the repository root in
the environment described in [ENVIRONMENT.md](ENVIRONMENT.md).

## Paper names and code identifiers

| Paper task | Exact task identifier | Simulator | Training demonstrations |
| --- | --- | --- | --- |
| [Close Box](#close-box) | `close_box` | `isaacsim` | 99 |
| [Pick Cube](#pick-cube) | `pick_cube` | `isaacsim` | 100 |
| [Stack Cube](#stack-cube) | `stack_cube` | `isaacsim` | 100 |
| [Pick-Place Bowl](#pick-place-bowl) | `libero_90.kitchen_scene1_put_the_black_bowl_on_the_plate` | `mujoco` | 40 |
| [Open Drawer](#open-drawer) | `libero_90.kitchen_scene1_open_bottom_drawer` | `mujoco` | 40 |

**The two MuJoCo names in the paper are shortened display names.** Open Drawer
means opening the **bottom drawer of the cabinet in kitchen scene 1**.
Pick-Place Bowl means placing the **black bowl on the plate in kitchen scene 1**.
Other drawer, bowl, or kitchen-scene tasks in the repository are different
tasks. Use the complete identifiers above for both `collect_demo.py --task`
and `il_run.sh --task_name_set` throughout the pipeline.

The converter's `--task_name` additionally includes the robot, randomization
level, and observation/action representation. The conversion command below
constructs that dataset identifier from the same task name.

## Before starting

Install the environment and download the robot, object, and reference
trajectory resources as described in [ENVIRONMENT.md](ENVIRONMENT.md#prepare-assets-and-demonstrations).
The collector replays those task reference trajectories and saves successful
rollouts in the metadata format used by the converter. Original demonstrations,
experiment outputs, and trained checkpoints from the paper are not included.

For headless MuJoCo rendering, configure the rendering backend before collection
or evaluation:

```bash
export MUJOCO_GL=egl
```

This rendering route requires a working EGL setup. Isaac Sim requires an
accessible NVIDIA GPU; FLASH training uses CUDA in the shared launcher.

## Select a task

Copy **one** of the following five blocks into your shell, then run the shared
collection, conversion, training, and evaluation steps below in the same shell.
To run another task, select its block and repeat those steps.

### Close Box

Close the box lid with the Franka robot. The implementation is
[CloseBoxTask](roboverse_pack/tasks/rlbench/close_box.py).

```bash
task_name="close_box"
sim_backend="isaacsim"
demo_count=99
```

Close Box uses 99 valid expert demonstrations in the paper's experiment setup.
Request 99 successful demonstrations here; the original collection run was
stopped after 99 because its final attempt stalled when requesting 100.

### Pick Cube

Pick up the cube with the Franka robot. The implementation is
[PickCubeTask](roboverse_pack/tasks/maniskill/pick_cube.py).

```bash
task_name="pick_cube"
sim_backend="isaacsim"
demo_count=100
```

### Stack Cube

Stack the red cube on the blue cube and release it. The registered implementation
is [StackCubeTask](metasim/example/example_pack/tasks/stack_cube.py).

```bash
task_name="stack_cube"
sim_backend="isaacsim"
demo_count=100
```

### Pick-Place Bowl

The paper's **Pick-Place Bowl** is the LIBERO kitchen scene 1 task that puts the
**black bowl on the plate**. The implementation is
[LiberoKitchen1PutBowlOnPlateTask](roboverse_pack/tasks/libero_90/libero_kitchen_scene1_put_the_black_bowl_on_the_plate.py).

```bash
task_name="libero_90.kitchen_scene1_put_the_black_bowl_on_the_plate"
sim_backend="mujoco"
demo_count=40
```

Keep this complete identifier in collection, dataset conversion, training, and
evaluation. Tasks that place a bowl on a cabinet or use a different bowl or
kitchen scene are not the paper's Pick-Place Bowl task.

### Open Drawer

The paper's **Open Drawer** is the LIBERO kitchen scene 1 task that opens the
**bottom drawer of the cabinet**. The implementation is
[LiberoKitchenOpenBottomDrawerTask](roboverse_pack/tasks/libero_90/libero_kitchen_scene1_open_bottom_drawer.py).

```bash
task_name="libero_90.kitchen_scene1_open_bottom_drawer"
sim_backend="mujoco"
demo_count=40
```

Keep this complete identifier in collection, dataset conversion, training, and
evaluation. The top-drawer task and the task that opens a drawer and puts a bowl
inside it are different tasks.

## Prepare demonstrations

With one task selected above, collect successful demonstrations using its
reference trajectories:

```bash
python scripts/advanced/collect_demo.py \
    --task "${task_name}" \
    --sim "${sim_backend}" \
    --robot franka \
    --num_envs 1 \
    --demo_start_idx 0 \
    --num_demo_success "${demo_count}" \
    --cust_name test \
    --level 0 \
    --run_unfinished \
    --headless
```

Successful demonstrations are written under:

```text
roboverse_demo/demo_<sim_backend>/<task_name>-test/robot-franka/success/
```

Each `demo_XXXX/` directory used for conversion must contain `metadata.json` and
`rgb.mp4`. If you already have compatible demonstrations, skip collection and
set `metadata_dir` in the next step to their success directory.

`collect_demo.sh` is a Stack Cube preset with settings defined inside the file;
it does not parse task-selection arguments. Use the direct collector command
above to select any of the five tasks without editing that preset.

## Convert demonstrations to Zarr

Use sparse sampling stride 4 for FLASH, with joint-position observations and
actions:

```bash
metadata_dir="./roboverse_demo/demo_${sim_backend}/${task_name}-test/robot-franka/success"

python roboverse_learn/il/data2zarr_dp.py \
    --task_name "${task_name}FrankaL0_obs:joint_pos_act:joint_pos" \
    --expert_data_num "${demo_count}" \
    --metadata_dir "${metadata_dir}" \
    --observation_space joint_pos \
    --action_space joint_pos \
    --downsample_ratio 4
```

The expected dataset path is:

```text
data_policy/<task_name>FrankaL0_obs:joint_pos_act:joint_pos_ds4_<demo_count>.zarr
```

The converter replaces an existing dataset with the same name. If fewer
demonstrations are available, it reduces the count in the filename; the launcher's
`--demo_num` must match the converted dataset. Collect the target count above to
retain the documented protocol. Using fewer demonstrations changes the setup.

For example, the two MuJoCo datasets are:

```text
data_policy/libero_90.kitchen_scene1_put_the_black_bowl_on_the_plateFrankaL0_obs:joint_pos_act:joint_pos_ds4_40.zarr
data_policy/libero_90.kitchen_scene1_open_bottom_drawerFrankaL0_obs:joint_pos_act:joint_pos_ds4_40.zarr
```

## Train FLASH

The following command works for each selected task. It configures 15,000
training steps and saves a checkpoint every 1,250 steps. Table 1 evaluates the
intermediate 10,000-step checkpoint. Keep the full 15,000-step training budget:
changing it to 10,000 can also change the learning-rate schedule.

```bash
bash roboverse_learn/il/il_run.sh \
    --task_name_set "${task_name}" \
    --sim_set "${sim_backend}" \
    --demo_num "${demo_count}" \
    --policy_name flash \
    --downsample_ratio 4 \
    --exp_name flash_paper \
    --num_steps 15000 \
    --checkpoint_every_n_steps 1250 \
    --train_enable True \
    --eval_enable False
```

The shared trainer rounds the budget up to a whole number of epochs, so the final
step count can slightly exceed 15,000. Keep the same task name, demonstration
count, sampling ratio, and experiment name when evaluating this training run.

## Evaluate the Table 1 checkpoint

The launcher requests 50 evaluation rollouts at randomization level 0. These
50 rollouts are separate from the training demonstration counts above. The
commands use the saved checkpoint at 10,000 steps:

```bash
checkpoint_path="./il_outputs/flash/${task_name}/flash_paper/checkpoints/10000.ckpt"

test -f "${checkpoint_path}" && \
bash roboverse_learn/il/il_run.sh \
    --task_name_set "${task_name}" \
    --sim_set "${sim_backend}" \
    --demo_num "${demo_count}" \
    --policy_name flash \
    --downsample_ratio 4 \
    --exp_name flash_paper \
    --num_steps 15000 \
    --checkpoint_every_n_steps 1250 \
    --eval_ckpt_name 10000 \
    --train_enable False \
    --eval_enable True
```

If the checkpoint is missing, `test -f` prevents the evaluation command from
running. Confirm that training saved it and inspect the reported checkpoint
load path. The shared evaluator can otherwise fall back to the latest checkpoint
when a requested file is missing. Run outputs remain under
`il_outputs/flash/<task_name>/flash_paper/` unless optional checkpoint archive
storage is configured.

## Scope

These commands reproduce the documented task selection and FLASH workflow using
demonstrations you prepare. Newly collected data may yield different numerical
results from the paper. Baselines have their own policy settings; if a baseline
uses sampling ratio 1, convert a separate `ds1` dataset and use the same ratio
with the launcher. The additional 2D experiments have a
[separate guide](roboverse_learn/il/push2d/README.md).
