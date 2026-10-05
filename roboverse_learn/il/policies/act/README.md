# ACT: Action Chunking with Transformers

## 1. Install
```bash
cd roboverse_learn/il/policies/act/detr && pip install -e .

cd ../../../../../
```

## 2. Collect and process data

Follow the [FLASH demonstration and training guide](../../../../README.md#demonstrations-and-training) to collect or supply demonstrations and convert them to Zarr. Stack Cube uses 100 valid demonstrations; the other four paper tasks use the counts in that guide.

## 3. Train and eval

Run the shared launcher from the repository root after preparing the Zarr dataset. The five paper tasks use 15,000 training steps, save every 1,250 steps, and evaluate the 10,000-step checkpoint for Table 1. ACT saves that checkpoint as `step_10000.ckpt`; the launcher also accepts `--eval_ckpt_name 10000`, preferring an existing literal `10000.ckpt` if present.

The standalone `act_run.sh` has the Stack Cube paper defaults. Use the shared launcher to select another task or override the protocol.

### 3.1 Train only

```bash
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name act \
    --demo_num 100 --num_steps 15000 --checkpoint_every_n_steps 1250 \
    --train_enable True --eval_enable False
```

### 3.2 Eval only
```bash
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name act \
    --demo_num 100 --num_steps 15000 --eval_ckpt_name 10000 \
    --train_enable False --eval_enable True
```
