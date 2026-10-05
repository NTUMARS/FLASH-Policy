# VITA Policy (IL)

VITA is a vision-to-action Flow Matching policy built on the shared IL runners under `il/policies/vita/`.

## Install

```bash
cd roboverse_learn/il/policies/vita
pip install -r requirements.txt
cd ../../../..
```

Create a Weights & Biases account to obtain an API key for logging.

## Collect and process data

Follow the [FLASH demonstration guide](../../../../README.md#demonstrations-and-training) to collect or supply demonstrations and convert them to Zarr, using 100 valid demonstrations for Stack Cube.

## Train and eval

For the five simulation tasks, follow the [FLASH paper protocol](../../../../README.md#demonstrations-and-training): Stack Cube uses 100 demonstrations, training is configured for 15,000 steps with checkpoints every 1,250 steps, and Table 1 uses the 10,000-step checkpoint. The example below trains; for evaluation, use `--train_enable False --eval_enable True --eval_ckpt_name 10000`.

```bash
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name vita --train_enable True --eval_enable False
```

Inside `il_run.sh` you can toggle `train_enable` / `eval_enable`, set task names, seeds, GPU id, and checkpoint paths for evaluation.
## References

- Dechen Gao et al., "VITA: Vision-to-Action Flow Matching Policy." (2025).
- Yaron Lipman et al., "Flow Matching for Generative Modeling." (2023).
