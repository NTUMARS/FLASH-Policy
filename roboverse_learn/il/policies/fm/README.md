# Flow Matching Policies (IL)

Flow Matching variants (UNet and DiT) live here and use the shared IL runners under `il/policies/fm/`.

## Install

```bash
cd roboverse_learn/il/policies/fm
pip install -r requirements.txt
cd ../../../..
```

Create a Weights & Biases account to obtain an API key for logging.

## Collect and process data

Follow the [FLASH demonstration guide](../../../../README.md#demonstrations-and-training) to collect or supply demonstrations and convert them to Zarr, using 100 valid demonstrations for Stack Cube.

## Train and eval

For the five simulation tasks, follow the [FLASH paper protocol](../../../../README.md#demonstrations-and-training): Stack Cube uses 100 demonstrations, training is configured for 15,000 steps with checkpoints every 1,250 steps, and Table 1 uses the 10,000-step checkpoint. The example below trains; for evaluation, use `--train_enable False --eval_enable True --eval_ckpt_name 10000`.

```bash
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name fm_unet --train_enable True --eval_enable False # or fm_dit
```

Inside `il_run.sh` you can toggle `train_enable` / `eval_enable`, set task names, seeds, GPU id, and checkpoint paths for evaluation.

## References

- Yaron Lipman et al., "Flow Matching for Generative Modeling." (2023).
- William Peebles and Jun-Yan Zhu, "DiT: Diffusion Models with Transformers." (2023).
