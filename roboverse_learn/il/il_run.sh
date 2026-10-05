#!/bin/bash
# Usage: bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name ddpm_dit --dr_level_eval 2 --train_enable False
# Usage: bash roboverse_learn/il/il_run.sh --downsample_ratio 4   # use downsampled training data
export PYTHONPATH=$(pwd):$PYTHONPATH  # Add current dir to Python path so roboverse_learn can be imported
export MUJOCO_GL=${MUJOCO_GL:-egl}   # MuJoCo headless rendering (keep existing value if set)

# ========== Default parameters ==========
task_name_set="stack_cube" # Example task: stack_cube
policy_name="flash_g"    # IL policy, opts: flash, flash_g, mip, ddpm_unet, ddpm_dit, ddim_unet, fm_unet, fm_dit, vita, a2a, a2a_mini, a2a_reg, a2a_noise, act, score
sim_set="isaacsim"          # Simulator, e.g., mujoco, isaacsim
demo_num=""               # Defaults to the demonstration count for the selected task
exp_name="default"       # Experiment name
downsample_ratio=1       # Training data downsample ratio: 1=66.7Hz (raw), 4=16.7Hz, 7=9.5Hz, etc.
                         # Must match the --downsample_ratio used in data2zarr_dp.py
                         # At eval, FLASH-G resamples the polynomial at the simulator's native
                         # control frequency (66.67 Hz) regardless of this ratio
resume=False             # True = resume training from the latest ckpt in output_dir/checkpoints/
eval_ckpt_name=""        # Checkpoint filename for eval (no .ckpt); paper preset selects 10000

# ========== Train/eval control ==========
train_enable=False   # True = run training, False = skip training
eval_enable=True    # True = run evaluation, False = skip evaluation

# Training hyperparameters
num_epochs=100       # Total training epochs

num_steps=0          # Total training steps (>0 overrides num_epochs and recomputes epoch count)
checkpoint_every_n_steps=1250  # >0 saves checkpoint by global_step (ignores epoch-based saving)
num_steps_explicit=False
num_epochs_explicit=False
eval_ckpt_name_explicit=False
seed=42
gpu=0
obs_space=joint_pos  # Observation space: joint_pos = joint angle control / ee = end-effector control
act_space=joint_pos  # Action space
delta_ee=0           # Use delta control: 0=False(absolute), 1=True(delta)
eval_num_envs=1      # Number of parallel envs for eval
eval_max_step=300    # Max steps per eval episode

# Domain Randomization Level   0=none, 1=scene+material, 2=+lighting, 3=+camera
dr_level_collect=0  # Randomization level for data collection
dr_level_eval=0     # Randomization level for evaluation

# Hydra config overrides (generic override, accepts any yaml parameter)
# Usage: --override "policy_config.poly_order=5"
#        --override "policy_config.poly_order=5" --override "policy_config.hidden_dim=256"
hydra_overrides=""
num_steps_override=""
num_epochs_override=""

# Parse parameters
while [[ $# -gt 0 ]]; do    # Iterate over all command-line arguments
    case "$1" in
        --task_name_set)
            task_name_set="$2"   # Read task name
            shift 2         # Advance to the next two arguments
            ;;
        --policy_name)
            policy_name="$2"   # Read policy name
            shift 2
            ;;
        --sim_set)
            sim_set="$2"       # Read simulator
            shift 2
            ;;
        --demo_num)
            demo_num="$2"      # Read number of demonstrations
            shift 2
            ;;
        --train_enable)
            train_enable="$2"  # Read train enable flag
            shift 2
            ;;
        --eval_enable)
            eval_enable="$2"    # Read eval enable flag
            shift 2
            ;;
        --dr_level_collect)
            dr_level_collect="$2"  # Read randomization level for data collection
            shift 2
            ;;
        --dr_level_eval)
            dr_level_eval="$2"    # Read randomization level for evaluation
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"      # Read total training epochs
            num_epochs_explicit=True
            shift 2
            ;;
        --gpu)
            gpu="$2"             # Read GPU ID
            shift 2
            ;;
        --exp_name)
            exp_name="$2"        # Read experiment name
            shift 2
            ;;
        --downsample_ratio)
            downsample_ratio="$2"  # Read training data downsample ratio
            shift 2
            ;;
        --resume)
            resume="$2"           # True = resume training from the latest checkpoint
            shift 2
            ;;
        --num_steps)
            num_steps="$2"        # Total training steps; overrides num_epochs when >0
            num_steps_explicit=True
            shift 2
            ;;
        --eval_ckpt_name)
            eval_ckpt_name="$2"   # Explicit eval checkpoint (without .ckpt suffix)
            eval_ckpt_name_explicit=True
            shift 2
            ;;
        --checkpoint_every_n_steps)
            checkpoint_every_n_steps="$2"  # Save checkpoint by global_step
            shift 2
            ;;
        --override)
            hydra_overrides="${hydra_overrides} $2"  # Append Hydra override
            # Budget overrides must also inform task defaults and checkpoint selection.
            override_key="${2%%=*}"
            override_key="${override_key#++}"
            override_key="${override_key#+}"
            case "${override_key}" in
                train_config.training_params.num_steps)
                    num_steps_override="${2#*=}"
                    num_steps_explicit=True
                    ;;
                train_config.training_params.num_epochs)
                    num_epochs_override="${2#*=}"
                    num_epochs_explicit=True
                    ;;
            esac
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"  # Unknown parameter
            echo "Optional parameters: --task_name_set --policy_name --sim_set --demo_num --train_enable --eval_enable --num_epochs --num_steps --gpu --exp_name --downsample_ratio --resume --eval_ckpt_name --checkpoint_every_n_steps --override"
            exit 1                  # Exit
            ;;
    esac
done

# Hydra overrides are appended after the named CLI arguments in the Python call.
[[ -n "${num_steps_override}" ]] && num_steps="${num_steps_override}"
[[ -n "${num_epochs_override}" ]] && num_epochs="${num_epochs_override}"

# Defaults for the five simulation tasks in the FLASH paper. Explicit CLI
# budgets and checkpoint selections take precedence; other tasks keep epoch mode.
paper_task=True
case "${task_name_set}" in
    close_box) default_demo_num=99 ;;
    pick_cube|stack_cube) default_demo_num=100 ;;
    libero_90.kitchen_scene1_put_the_black_bowl_on_the_plate|libero_90.kitchen_scene1_open_bottom_drawer)
        default_demo_num=40 ;;
    *) default_demo_num=100; paper_task=False ;;
esac
demo_num="${demo_num:-${default_demo_num}}"
if [[ "${paper_task}" == True ]]; then
    if [[ "${num_steps_explicit}" == False && "${num_epochs_explicit}" == False ]]; then
        num_steps=15000
    fi
    if [[ "${eval_ckpt_name_explicit}" == False && "${num_steps}" == 15000 ]]; then
        eval_ckpt_name=10000
    fi
fi

# The block below collects new demonstration data; commented out since data already exists
# # Collect demo
# echo "=== Running collect_demo.sh ==="
# sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^num_demo_success=.*/num_demo_success=$demo_num/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^random_level=.*/random_level=$dr_level_collect/" ./roboverse_learn/il/collect_demo.sh
# bash ./roboverse_learn/il/collect_demo.sh

# ========== Config file mapping ==========
# Map policy_name to model config
config_name="default_runner"    # Main config filename (without .yaml suffix)
main_script="./roboverse_learn/il/train.py"     # Path to the main training script

# Standalone 2D tasks use the shared image-policy trainer and lightweight evaluators.
if [[ "${task_name_set}" == corridor_push* || "${task_name_set}" == switchshot* || "${task_name_set}" == forkpush* ]]; then
    if [[ "${task_name_set}" != corridor_push_lv2 && "${task_name_set}" != switchshot_N_lv4 ]]; then
        echo "[push2d] Published tasks: corridor_push_lv2 and switchshot_N_lv4"
        exit 1
    fi
    if [[ "${downsample_ratio}" != 1 ]]; then
        echo "[push2d] These tasks use the original 5Hz data; --downsample_ratio must be 1"
        exit 1
    fi
    export policy_name
    shopt -s nullglob
    push2d_matches=( ./data_policy/"${task_name_set}"_*.zarr )
    if [[ ${#push2d_matches[@]} -ne 1 || ! -e "${push2d_matches[0]}" ]]; then
        echo "[push2d] Expected exactly one data_policy/${task_name_set}_*.zarr dataset. See push2d/README.md"
        exit 1
    fi
    push2d_zarr="${push2d_matches[0]}"
    if [[ "${train_enable}" == True ]]; then
        python "${main_script}" --config-name="${config_name}.yaml" \
            "task_name=${task_name_set}" "exp_name=${exp_name}" \
            "dataset_config.zarr_path=${push2d_zarr}" \
            "shape_meta.obs.head_cam.shape=[3,96,96]" \
            "shape_meta.obs.agent_pos.shape=[2]" "shape_meta.action.shape=[2]" \
            "train_config.training_params.seed=${seed}" \
            "train_config.training_params.num_epochs=${num_epochs}" \
            "train_config.training_params.num_steps=${num_steps}" \
            "train_config.training_params.device=cuda:${gpu}" \
            "train_config.training_params.checkpoint_every_n_steps=${checkpoint_every_n_steps}" \
            "train_config.training_params.resume=${resume}" \
            "train_config.val_dataloader.batch_size=8" \
            train_enable=True eval_enable=False eval_path=null ${hydra_overrides} || exit $?
    fi
    if [[ "${eval_enable}" == True ]]; then
        push2d_ckpt_dir="./il_outputs/${policy_name}/${task_name_set}/${exp_name}/checkpoints"
        push2d_checkpoint="${push2d_ckpt_dir}/${eval_ckpt_name:-${num_epochs}}.ckpt"
        if [[ -z "${eval_ckpt_name}" && ${num_steps} -gt 0 ]]; then
            push2d_checkpoint=$(python - "${push2d_ckpt_dir}" <<'PYCHECKPOINT'
from pathlib import Path
import sys
paths = list(Path(sys.argv[1]).glob("*.ckpt"))
if not paths:
    raise SystemExit("No checkpoint found")
print(max(paths, key=lambda p: p.stat().st_mtime).as_posix())
PYCHECKPOINT
            ) || exit $?
        fi
        if [[ ! -f "${push2d_checkpoint}" ]]; then
            echo "[push2d] Missing checkpoint: ${push2d_checkpoint}"
            exit 1
        fi
        push2d_eval_out="./il_outputs/${policy_name}/${task_name_set}/${exp_name}/$(basename "${push2d_checkpoint}")_$(date +%F_%H-%M-%S)"
        push2d_flags=()
        [[ -n "${PUSH2D_NIS:-}" ]] && push2d_flags+=( --num-inference-steps "${PUSH2D_NIS}" )
        if [[ "${task_name_set}" == switchshot_N_lv4 ]]; then
            [[ -n "${PUSH2D_EXECUTE_STEPS:-}" ]] && push2d_flags+=( --execute-steps "${PUSH2D_EXECUTE_STEPS}" )
            python -m roboverse_learn.il.push2d.switchshot_eval \
                --checkpoint "${push2d_checkpoint}" --zarr "${push2d_zarr}" \
                --out "${push2d_eval_out}" --eps "${PUSH2D_EPS:-48}" --device "cuda:${gpu}" \
                "${push2d_flags[@]}" || exit $?
        else
            python -m roboverse_learn.il.push2d.push2d_eval \
                --checkpoint "${push2d_checkpoint}" --zarr "${push2d_zarr}" \
                --out "${push2d_eval_out}" --levels 2 --eps "${PUSH2D_EPS:-50}" --device "cuda:${gpu}" \
                "${push2d_flags[@]}" || exit $?
        fi
    fi
    exit 0
fi

# ========== Special handling for ACT policy ==========
# if policy_name is ACT
if [ "${policy_name}" = "act" ]; then
    echo "=== Running ACT training and evaluation==="
    sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^train_enable=.*/train_enable=$train_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_enable=.*/eval_enable=$eval_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^collect_level=.*/collect_level=$dr_level_collect/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_level=.*/eval_level=$dr_level_eval/" ./roboverse_learn/il/policies/act/act_run.sh
    # Same as the generic policy above: pass GPU, epoch, downsample, seed, num_steps
    sed -i "s/^gpu_id=.*/gpu_id=$gpu/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^num_epochs=.*/num_epochs=$num_epochs/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^downsample_ratio=.*/downsample_ratio=$downsample_ratio/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^seed=.*/seed=$seed/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^num_steps=.*/num_steps=$num_steps/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^checkpoint_every_n_steps=.*/checkpoint_every_n_steps=$checkpoint_every_n_steps/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_ckpt_name=.*/eval_ckpt_name=$eval_ckpt_name/" ./roboverse_learn/il/policies/act/act_run.sh
    bash ./roboverse_learn/il/policies/act/act_run.sh    # Run ACT training and evaluation
    echo "=== Completed all data collection, training, and evaluation ==="    # All data collection, training, and evaluation finished
    exit 0    # Exit
fi

# ========== Generic policy training/evaluation ==========
# Run training/evaluation for DP/FM/VITA policies
echo "=== Running ${policy_name} ==="

output_dir="./il_outputs/${policy_name}"   # Output dir, e.g. ./il_outputs/flash_g/
ckpt_dir="${output_dir}/${task_name_set}/${exp_name}/checkpoints"

# Optional checkpoint archive; unset keeps checkpoints in il_outputs/.
HDD_CKPT_BASE="${CHECKPOINT_ARCHIVE_DIR:-}"
if [[ -n "$HDD_CKPT_BASE" && "$HDD_CKPT_BASE" != /* ]]; then
    echo "[ERROR] CHECKPOINT_ARCHIVE_DIR must be an absolute path."
    exit 2
fi
if [ -n "$HDD_CKPT_BASE" ] && [ -d "$HDD_CKPT_BASE" ] && [ ! -L "$ckpt_dir" ] && [ ! -d "$ckpt_dir" ]; then
    hdd_ckpt_dir="${HDD_CKPT_BASE}/${ckpt_dir}"
    mkdir -p "$hdd_ckpt_dir"
    mkdir -p "$(dirname "$ckpt_dir")"
    ln -s "$hdd_ckpt_dir" "$ckpt_dir"
    echo "[HDD] Checkpoint dir symlinked: $ckpt_dir -> $hdd_ckpt_dir"
fi

# Determine eval checkpoint path
if [ -n "${eval_ckpt_name}" ]; then
    # Selected by the paper preset or explicitly via --eval_ckpt_name
    eval_path="${ckpt_dir}/${eval_ckpt_name}.ckpt"
    echo "[Eval] Using selected checkpoint: ${eval_path}"
elif [ "${num_steps}" -gt 0 ] 2>/dev/null; then
    # num_steps mode: epoch count is computed at Python runtime from steps_per_epoch,
    # the shell cannot know the final epoch in advance. Set eval_path=null so Python auto-detects the latest ckpt
    eval_path="null"
    echo "[num_steps=${num_steps}] eval_path=null → Python will auto-detect latest checkpoint."
elif [ "${resume}" = "True" ] && [ -d "${ckpt_dir}" ]; then
    latest_epoch=$(ls "${ckpt_dir}"/*.ckpt 2>/dev/null | sed 's/.*\///' | sed 's/\.ckpt//' | sort -n | tail -1)
    if [ -n "${latest_epoch}" ]; then
        eval_ckpt_name=$(( latest_epoch + num_epochs ))
        echo "[Resume] Latest checkpoint: ${latest_epoch}.ckpt → will train ${num_epochs} more epochs → eval ${eval_ckpt_name}.ckpt"
    else
        eval_ckpt_name=$num_epochs
        echo "[Resume] No existing checkpoints found, starting fresh → eval ${eval_ckpt_name}.ckpt"
    fi
    eval_path="${ckpt_dir}/${eval_ckpt_name}.ckpt"
else
    eval_ckpt_name=$num_epochs
    eval_path="${ckpt_dir}/${eval_ckpt_name}.ckpt"
fi

echo "Checkpoint path: $eval_path"

# ========== Build dataset path identifier ==========
extra="obs:${obs_space}_act:${act_space}"  # Base identifier: obs:joint_pos_act:joint_pos
if [ "${delta_ee}" = 1 ]; then  # Use delta control: 0=False(absolute), 1=True(delta)
  extra="${extra}_delta"
fi
# downsample_ratio embedded in path: ds1=66.7Hz raw, ds4=16.7Hz, ds7=9.5Hz
# Path format: {task}FrankaL{dr}_obs:{obs}_act:{act}_ds{ratio}_{num}.zarr
# Data is generated by data2zarr_dp.py --downsample_ratio {ratio}
zarr_ds_suffix="ds${downsample_ratio}"

# When downsample_ratio > 1 each episode has fewer frames; auto-shrink val_dataloader batch_size
# to ensure the validation set has at least 1 full batch (avoids ds=7 yielding ~16 samples / batch32 = 0 batches)
val_batch_size=$(( 32 / downsample_ratio < 1 ? 1 : 32 / downsample_ratio ))
# max_train_steps: upper bound on batches per epoch
# Not scaled with downsample_ratio -- smaller datasets (ds=4/7) already have < 250 total batches,
# so the training loop naturally ends after iterating all batches without early truncation
max_train_steps=250

# ========== Export environment variables ==========
export policy_name="${policy_name}"  # Export policy name
python ${main_script} --config-name=${config_name}.yaml \
task_name=${task_name_set} \
exp_name=${exp_name} \
"dataset_config.zarr_path=./data_policy/${task_name_set}FrankaL${dr_level_collect}_${extra}_${zarr_ds_suffix}_${demo_num}.zarr" \
train_config.training_params.seed=${seed} \
train_config.training_params.num_epochs=${num_epochs} \
train_config.training_params.num_steps=${num_steps} \
train_config.training_params.device=${gpu} \
eval_config.policy_runner.obs.obs_type=${obs_space} \
eval_config.policy_runner.action.action_type=${act_space} \
eval_config.policy_runner.action.delta=${delta_ee} \
eval_config.eval_args.task=${task_name_set} \
eval_config.eval_args.max_step=${eval_max_step} \
eval_config.eval_args.num_envs=${eval_num_envs} \
eval_config.eval_args.sim=${sim_set} \
eval_config.eval_args.level=${dr_level_eval} \
+eval_config.eval_args.max_demo=50 \
+eval_config.eval_args.downsample_ratio=${downsample_ratio} \
train_config.training_params.max_train_steps=${max_train_steps} \
train_config.training_params.checkpoint_every_n_steps=${checkpoint_every_n_steps} \
train_config.val_dataloader.batch_size=${val_batch_size} \
train_config.training_params.resume=${resume} \
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path} \
${hydra_overrides}

echo "=== Completed all data collection, training, and evaluation ==="
