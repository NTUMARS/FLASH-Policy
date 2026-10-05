## Separate script for training and evaluation
train_enable=False
eval_enable=True


## Parameters
task_name_set=stack_cube
expert_data_num=100
gpu_id=0
sim_set=isaacsim
num_epochs=100
# Matches il_run.sh / data2zarr dataset naming: ..._ds{ratio}_{demo_num}.zarr
downsample_ratio=1

obs_space=joint_pos # joint_pos or ee
act_space=joint_pos # joint_pos or ee
delta_ee=0 # 0 or 1 (only matters if act_space is ee, 0 means absolute 1 means delta control)

alg_name=ACT
seed=42
collect_level=0
# Paper protocol: train for 15k steps and evaluate the intermediate 10k checkpoint.
num_steps=15000
checkpoint_every_n_steps=1250
eval_ckpt_name=10000

# ACT hyperparameters
chunk_size=8
kl_weight=10
hidden_dim=256
lr=1e-5
batch_size=32
dim_feedforward=1024

# Transformer architecture
enc_layers=2
dec_layers=4
nheads=4

# Domain Randomization parameters for evaluation
eval_level=0
eval_scene_mode=0     # 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD
eval_seed=42          # Randomization seed (optional)

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

# Optional checkpoint archive; unset keeps checkpoints in il_outputs/.
HDD_CKPT_BASE="${CHECKPOINT_ARCHIVE_DIR:-}"
if [[ -n "$HDD_CKPT_BASE" && "$HDD_CKPT_BASE" != /* ]]; then
    echo "[ERROR] CHECKPOINT_ARCHIVE_DIR must be an absolute path."
    exit 2
fi
act_ckpt_dir="il_outputs/act/${task_name_set}/ckpt/level${collect_level}"
if [ -d "$HDD_CKPT_BASE" ] && [ ! -L "$act_ckpt_dir" ] && [ ! -d "$act_ckpt_dir" ]; then
    hdd_act_ckpt_dir="${HDD_CKPT_BASE}/${act_ckpt_dir}"
    mkdir -p "$hdd_act_ckpt_dir"
    mkdir -p "$(dirname "$act_ckpt_dir")"
    ln -s "$hdd_act_ckpt_dir" "$act_ckpt_dir"
    echo "[HDD] ACT checkpoint dir symlinked: $act_ckpt_dir -> $hdd_act_ckpt_dir"
fi

zarr_ds_suffix="ds${downsample_ratio}"
dataset_zarr="data_policy/${task_name_set}FrankaL${collect_level}_${extra}_${zarr_ds_suffix}_${expert_data_num}.zarr"
# Legacy naming (without _ds* segment): fall back if new path does not exist
legacy_zarr="data_policy/${task_name_set}FrankaL${collect_level}_${extra}_${expert_data_num}.zarr"
if [ ! -d "${dataset_zarr}" ] && [ -d "${legacy_zarr}" ]; then
  dataset_zarr="${legacy_zarr}"
  echo "[ACT] Using legacy dataset path: ${dataset_zarr}"
fi

# Training (accepts True/true)
if [[ "${train_enable}" == [Tt]rue ]]; then
  echo "=== Training ==="
  export CUDA_VISIBLE_DEVICES=${gpu_id}
  python -m roboverse_learn.il.policies.act.train \
  --task_name ${task_name_set} \
  --num_episodes ${expert_data_num} \
  --dataset_dir "${dataset_zarr}" \
  --policy_class ${alg_name} --kl_weight ${kl_weight} --chunk_size ${chunk_size} \
  --hidden_dim ${hidden_dim} --batch_size ${batch_size} --dim_feedforward ${dim_feedforward} \
  --enc_layers ${enc_layers} --dec_layers ${dec_layers} --nheads ${nheads} \
  --num_epochs ${num_epochs} --num_steps ${num_steps} \
  --checkpoint_every_n_steps ${checkpoint_every_n_steps} \
  --lr ${lr} --state_dim 9 \
  --seed ${seed} \
  --level ${collect_level}
fi

# Evaluation
if [[ "${eval_enable}" == [Tt]rue ]]; then
  echo "=== Evaluation ==="
  # Build checkpoint dir directly from task name (no longer relies on ckpt_dir_path.txt written during training)
  act_ckpt_eval_dir="il_outputs/act/${task_name_set}/ckpt/level${collect_level}"

  # === HDD redirect: reuse SSD symlink if present, otherwise check if it exists on HDD ===
  HDD_CKPT_BASE="${CHECKPOINT_ARCHIVE_DIR:-}"
  if [ -L "${act_ckpt_eval_dir}" ]; then
    echo "[Eval] Using existing symlink: ${act_ckpt_eval_dir}"
  elif [ -n "${HDD_CKPT_BASE}" ] && [ -d "${HDD_CKPT_BASE}/${act_ckpt_eval_dir}" ]; then
    mkdir -p "$(dirname "${act_ckpt_eval_dir}")"
    ln -s "${HDD_CKPT_BASE}/${act_ckpt_eval_dir}" "${act_ckpt_eval_dir}"
    echo "[Eval] Created symlink: ${act_ckpt_eval_dir} -> ${HDD_CKPT_BASE}/${act_ckpt_eval_dir}"
  fi

  if [ ! -d "${act_ckpt_eval_dir}" ]; then
    echo "ERROR: Checkpoint dir not found: ${act_ckpt_eval_dir}"
    echo "Please train ACT for task '${task_name_set}' first."
    exit 1
  fi

  # Determine the checkpoint filename for evaluation
  if [ -n "${eval_ckpt_name}" ]; then
    act_ckpt_file="${eval_ckpt_name}"
    # Preserve exact filenames; numeric selections also accept ACT's native name.
    if [[ "${eval_ckpt_name}" =~ ^[0-9]+$ ]] && \
       [[ ! -f "${act_ckpt_eval_dir}/${eval_ckpt_name}.ckpt" ]] && \
       [[ -f "${act_ckpt_eval_dir}/step_${eval_ckpt_name}.ckpt" ]]; then
      act_ckpt_file="step_${eval_ckpt_name}"
    fi
  elif [ "${num_steps}" -gt 0 ] 2>/dev/null; then
    # num_steps mode: use step_{num_steps} or policy_last
    if [ -f "${act_ckpt_eval_dir}/step_${num_steps}.ckpt" ]; then
      act_ckpt_file="step_${num_steps}"
    else
      act_ckpt_file="policy_last"
      echo "[Eval] step_${num_steps}.ckpt not found, falling back to policy_last.ckpt"
    fi
  elif [ "${num_epochs}" -gt 0 ] 2>/dev/null; then
    # num_epochs for checkpoint selection: try step_{num_epochs} first, then {num_epochs}
    if [ -f "${act_ckpt_eval_dir}/step_${num_epochs}.ckpt" ]; then
      act_ckpt_file="step_${num_epochs}"
    elif [ -f "${act_ckpt_eval_dir}/${num_epochs}.ckpt" ]; then
      act_ckpt_file="${num_epochs}"
    else
      act_ckpt_file="policy_last"
      echo "[Eval] No checkpoint for num_epochs=${num_epochs}, falling back to policy_last.ckpt"
    fi
  else
    act_ckpt_file="policy_last"
  fi
  echo "[Eval] Checkpoint dir: ${act_ckpt_eval_dir}, file: ${act_ckpt_file}.ckpt"

  export CUDA_VISIBLE_DEVICES=${gpu_id}

  python -m roboverse_learn.il.policies.act.act_eval_runner \
  --task ${task_name_set} \
  --robot franka \
  --num_envs 1 \
  --sim ${sim_set} \
  --algo act \
  --ckpt_path  ./${act_ckpt_eval_dir} \
  --eval_ckpt_name ${act_ckpt_file} \
  --headless True \
  --num_eval 50 \
  --temporal_agg True \
  --chunk_size ${chunk_size} \
  --hidden_dim ${hidden_dim} \
  --dim_feedforward ${dim_feedforward} \
  --enc_layers ${enc_layers} \
  --dec_layers ${dec_layers} \
  --nheads ${nheads} \
  --level ${eval_level} \
  --scene_mode ${eval_scene_mode} \
  --randomization_seed ${eval_seed} \
  --downsample_ratio ${downsample_ratio}
fi
