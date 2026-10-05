# A2A Policy (IL)

A2A (Action-to-Action) is a flow matching policy that directly transforms history action distributions to future action distributions, conditioned on visual observations.

## Variants

- **A2A**: Base Action-to-Action flow matching policy
- **A2A-Noise**: A variant that adds Gaussian noise to history states before encoding for improved robustness

## Architecture

```
History States [s_{t-n+1}, ..., s_t] --encode--> history_latents (x_0)
Visual Obs [img_{t-n+1}, ..., img_t] --encode--> obs_latents (condition)

Flow Matching: x_0 --flow(condition)--> x_1 (future_action_latents)

x_1 --decode--> Future Actions [a_t, a_{t+1}, ..., a_{t+k}]
```

## Install

```bash
cd roboverse_learn/il/policies/a2a
pip install -r requirements.txt
cd ../../../..
```

Create a Weights & Biases account to obtain an API key for logging.

## Collect and process data

Follow the [FLASH demonstration guide](../../../../README.md#demonstrations-and-training) to collect or supply demonstrations and convert them to Zarr, using 100 valid demonstrations for Stack Cube.

## Train and eval

For the five simulation tasks, follow the [FLASH paper protocol](../../../../README.md#demonstrations-and-training): Stack Cube uses 100 demonstrations, training is configured for 15,000 steps with checkpoints every 1,250 steps, and Table 1 uses the 10,000-step checkpoint. The examples below train; for evaluation, use `--train_enable False --eval_enable True --eval_ckpt_name 10000`.

### A2A
```bash
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name a2a --train_enable True --eval_enable False
```

### A2A-Noise
```bash
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name a2a_noise --train_enable True --eval_enable False
```

Inside `il_run.sh` you can toggle `train_enable` / `eval_enable`, set task names, seeds, GPU id, and checkpoint paths for evaluation.
