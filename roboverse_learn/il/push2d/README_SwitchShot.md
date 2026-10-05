# ForkReach (`switchshot_N_lv4`)

ForkReach is the multimodal 2D navigation task used in the rebuttal. The retained
code name is `switchshot_N_lv4`: variant N controls a point agent directly, and
level 4 lights both goal regions from the start. Reaching either region succeeds.
The environment contains no pushed puck or changing goal signal.

Each expert pair starts from the same environment seed and initial observation;
one demonstration chooses the top goal and the other chooses the bottom goal.
The expert's goal choice is excluded from the policy observation. Consequently,
the demonstration distribution has two valid modes for the same initial state.

Run the commands below from the repository root in the project's Python
environment. Dependencies and data layout are described in [README.md](README.md).

## Generate paired demonstrations

```bash
python -m roboverse_learn.il.push2d.switchshot_gen_demos \
  --out data_policy/switchshot_N_lv4_100.zarr \
  --variant N --levels 4 --eps-per-level 100 --seed 0
```

Use an even `--eps-per-level` count so each pair contributes one top and one
bottom demonstration. The generator writes the Zarr directory and its matching
`switchshot_N_lv4_100_episodes_meta.json` sidecar. Keep them together. The sidecar
records episode seeds, expert modes and pair IDs; generated image/state/action
arrays follow the same training format as CorridorPush.

Check expert success, paired initial observations and polynomial-fit diagnostics:

```bash
python -m roboverse_learn.il.push2d.switchshot_calibrate \
  --variant N --eps 48 --out outputs/forkreach_calibration.txt
```

## Train and evaluate

Keep exactly one `switchshot_N_lv4*.zarr` dataset in `data_policy/` for automatic
dataset selection.

```bash
PUSH2D_NIS=8 PUSH2D_EXECUTE_STEPS=6 bash roboverse_learn/il/il_run.sh \
  --task_name_set switchshot_N_lv4 --policy_name flash --exp_name std07_e10000 \
  --num_steps 10000 --checkpoint_every_n_steps 1000 --eval_ckpt_name 3000 \
  --override n_action_steps=6 --override policy_config.history_noise_std=0.7 \
  --override train_config.dataloader.batch_size=64 \
  --override train_config.training_params.max_train_steps=250 \
  --train_enable True --eval_enable True
```

Other configured policies, including `fm_dit` and `act`, use the same task/data
route. For those two methods use `n_action_steps=8`, omit the FLASH-specific
`history_noise_std` override, and set `PUSH2D_NIS` to 10 and 1 respectively. To evaluate an existing checkpoint with explicit settings:

```bash
python -m roboverse_learn.il.push2d.switchshot_eval \
  --checkpoint path/to/3000.ckpt \
  --zarr data_policy/switchshot_N_lv4_100.zarr \
  --out outputs/forkreach_eval --variant N --levels 4 \
  --eps 48 --seed 10000 --execute-steps 8
```

`--execute-steps` controls how many predicted actions run before replanning.
Preserve each run's actual interval and inference-step count when reproducing
comparisons. In the audited rebuttal runs, FLASH executes 6 steps per prediction;
FM-DiT and ACT execute 8. Their inference-step counts are 8, 10 and 1 respectively,
evaluating checkpoint 3000 on 48 episodes with environment seed 10000. Set
`--num-inference-steps` along with `--execute-steps` to reproduce those settings.
The environment seed does not fix the generative policy sampling RNG; historical
evaluations did not record a sampler seed.

The evaluator reports:

- Success rate: fraction of episodes reaching either goal.
- Top/bottom committed choices: selected branches across episodes, including a
  committed branch in an episode that subsequently times out.
- Mode balance: smaller/larger committed-goal count; 1 is equal coverage and 0 is
  collapse to a single branch. Successful goal arrivals are also reported
  separately as `L4_mode_split`.
- Jerk: RMS second-order difference of executed absolute actions, in pixels.

Results include `00_final_stats.{txt,json}`, evaluation settings, rollout logs,
and per-episode videos and trajectory PNGs. Use `--no-video` or `--no-log` to
disable the optional artifacts.

## Inspect demonstrations

Replay the expert with the recorded environment seed, expert seed and mode:

```bash
python -m roboverse_learn.il.push2d.switchshot_viz \
  --sidecar data_policy/switchshot_N_lv4_100_episodes_meta.json \
  --per-level 2 --out outputs/forkreach_expert
```

For fresh expert episodes, use `--eps 2` in place of `--sidecar` and `--per-level`.
To export stored frames directly, without running the expert:

```bash
python -m roboverse_learn.il.push2d.switchshot_dump_zarr \
  --zarr data_policy/switchshot_N_lv4_100.zarr \
  --per-level 2 --out outputs/forkreach_stored
```

Expert replay produces videos and complete trajectory overviews. Direct frame
export produces videos and last-frame PNGs from the stored training images.
