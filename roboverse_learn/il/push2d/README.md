# CorridorPush and ForkReach

This package contains two custom 2D imitation-learning tasks. The published task names are
`corridor_push_lv2` (CorridorPush level 2) and `switchshot_N_lv4` (ForkReach).

| Task | Description | Entry points |
|---|---|---|
| CorridorPush | A pusher moves a puck through a sawtooth corridor to the right-hand goal. | `gen_demos`, `push2d_eval`, `push2d_viz`, `push2d_spectrum` |
| ForkReach | A point agent reaches either of two equally valid goal regions. Demonstrations use opposite-goal pairs from identical initial states. | See [ForkReach instructions](README_SwitchShot.md). |

These task modules require Python 3.10 or newer. Run commands from the repository
root in the project's Python environment.
Install the task dependencies and the policy dependencies described in the
[imitation-learning guide](../README.md):

```bash
pip install -e '.[push2d]' -r roboverse_learn/il/push2d/requirements.txt
```

The environments, demonstration generators and standalone evaluators can run on
CPU. The training wrapper uses the selected CUDA device. Dataset generation uses
Zarr 2; video export uses ImageIO with FFmpeg.

## CorridorPush data

Level 2 is the rebuttal stress test. Its tooth period is 100 px, with a
direction reversal every 50 px. The 41 px throat leaves 8.5 px clearance on each
side of the 24 px puck. Geometry and physics are defined in
`corridor_geometry.py` and `corridor_push_env.py`.

Generate the level-2 dataset:

```bash
python -m roboverse_learn.il.push2d.gen_demos \
  --out data_policy/corridor_push_lv2_100.zarr \
  --levels 2 --eps-per-level 100 --seed 0
```

Each dataset has a matching
`*_episodes_meta.json` sidecar recording levels, episode seeds and generation
parameters. Keep the sidecar beside the Zarr directory, including when using
symbolic links.

Stored observations are 96 × 96 BGR images in CHW format (`data/head_camera`),
agent coordinates (`data/state`) and absolute 2D agent target positions
(`data/action`). Episode boundaries are stored in `meta/episode_ends`.

## Training and evaluation

The existing training wrapper selects the policy, finds the dataset by task-name
prefix, and sets the 2D image/state/action shapes. Keep exactly one matching Zarr
directory in `data_policy/` for each task name.

```bash
bash roboverse_learn/il/il_run.sh \
  --task_name_set corridor_push_lv2 --policy_name flash --exp_name k9 \
  --num_steps 3000 --checkpoint_every_n_steps 1000 \
  --override policy_config.poly_order=9 \
  --train_enable True --eval_enable True
```

Use `--policy_name` to select another configured policy and `--override` for
policy settings. Record the effective checkpoint configuration, inference-step
count and action execution interval when comparing runs.

Evaluate a checkpoint directly:

```bash
python -m roboverse_learn.il.push2d.push2d_eval \
  --checkpoint path/to/3000.ckpt \
  --zarr data_policy/corridor_push_lv2_100.zarr \
  --out outputs/corridor_push_lv2_eval --levels 2 --eps 50 --seed 10000
```

Evaluation writes `00_final_stats.{txt,json}`, the effective evaluation config,
videos and trajectory overviews. Use `--no-video` to disable the visuals. The output directory is renamed to
include success rate, inference-step count and jerk. Jerk is the RMS second-order
difference of the executed absolute action sequence, in pixels, averaged over
episodes. A prediction executes the checkpoint's `n_action_steps` before
replanning (8 in the rebuttal CorridorPush runs). `--seed` sets environment
initialization; `--torch-seed` optionally seeds policy sampling. Historical
evaluations did not record a sampler seed.

The released expert holds the agent stationary after success when completing a
minimum-length demonstration. This fixes the original generator's post-success
steering. Newly generated data therefore correspond to the corrected dataset,
rather than the original rebuttal training data. Keep dataset versions separate
when reproducing existing checkpoints.

For degree-fit diagnostics and expert projection rollouts:

```bash
python -m roboverse_learn.il.push2d.push2d_spectrum \
  --zarr data_policy/corridor_push_lv2_100.zarr \
  --oracle 6 9 --levels 2 --eps 30
```

These diagnostics measure action representation and expert rollouts; learned
policy performance comes from checkpoint evaluation.

## Expert visualization

Replay recorded expert seeds with the dataset's sidecar:

```bash
python -m roboverse_learn.il.push2d.push2d_viz \
  --sidecar data_policy/corridor_push_lv2_100_episodes_meta.json \
  --levels 2 --per-level 2 --out outputs/corridor_push_lv2_expert
```

To generate fresh expert episodes, replace `--sidecar` and `--per-level` with
`--eps 2`. ForkReach has its own replay entry point because its environment differs
from CorridorPush.
