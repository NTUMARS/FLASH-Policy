#!/bin/bash
# ============================================================================
# batch_run.sh — Batch automation script for experiments
#
# Reads multiple experiment configs from a file and runs training/evaluation in sequence.
# After each experiment, automatically detects completion, force-kills hanging processes,
# releases GPU memory, then launches the next experiment.
#
# Termination: press Ctrl+C to terminate the current experiment and the script
#
# Usage:
#   bash scripts/batch_run.sh scripts/experiments.txt           # Run all experiments in the config file
#   bash scripts/batch_run.sh scripts/experiments.txt --dry-run # Print the plan only, do not run
#   bash scripts/batch_run.sh scripts/experiments.txt --gpu 1   # Specify GPU
#
# If GPU memory is still held after an experiment, set the following (kills current-user GPU
# processes whose cmdline matches Isaac/Omni etc.; verify no other important tasks first):
#   ROBOVERSE_NVIDIA_CLEANUP=1 bash scripts/batch_run.sh ...
# Or run separately: bash scripts/cleanup_roboverse_gpu.sh
#
# Config file format (experiments.txt):
#   One experiment per line, contents are arguments forwarded to il_run.sh (same CLI usage)
#   Lines starting with # and blank lines are skipped
#   Example:
#     --task_name_set pick_cube --policy_name ddpm_dit --num_epochs 100
#     --task_name_set stack_cube --policy_name flash_g --num_steps 20000
# ============================================================================

set -uo pipefail

# ======================== Tunable parameters ========================
GRACE_PERIOD=20        # Seconds to wait after completion is detected (let process exit on its own)
KILL_ESCALATION=5      # Seconds to wait between escalating signals
STALE_TIMEOUT=600      # No-output timeout in seconds; treated as hung if exceeded (0=disabled)
POLL_INTERVAL=3        # Log polling interval (seconds)
CONTINUE_ON_FAILURE=true  # Whether to continue with the next experiment after a failure

# ======================== Color definitions ========================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ======================== Global state ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_BASE_DIR="${PROJECT_ROOT}/experiment_logs"
SUMMARY_FILE=""
DRY_RUN=false
GPU_OVERRIDE=""

# PID of the currently running child process (used by trap cleanup)
CURRENT_EXP_PID=""
CURRENT_TAIL_PID=""

# ======================== Signal handling ========================
# Clean up all child processes and exit on Ctrl+C / kill
cleanup_and_exit() {
    echo ""
    log_warn "Received a termination signal, cleaning up all child processes..."
    if [ -n "$CURRENT_EXP_PID" ]; then
        # Under setsid PGID = PID, kill the whole group directly
        kill -TERM -- -"$CURRENT_EXP_PID" 2>/dev/null || true
        sleep 1
        kill -KILL -- -"$CURRENT_EXP_PID" 2>/dev/null || true
        sleep 1
        # Fallback: recursively kill descendants
        kill_descendants "$CURRENT_EXP_PID" 9
    fi
    roboverse_nvidia_orphan_cleanup
    if [ -n "$CURRENT_TAIL_PID" ]; then
        kill "$CURRENT_TAIL_PID" 2>/dev/null
    fi
    log_info "Cleaning completed"
    exit 130
}
trap cleanup_and_exit INT TERM

# ======================== Function definitions ========================

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║        Batch automated experiment runner  (batch_run)        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING $*${NC}"; }
log_error() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR $*${NC}"; }
log_step()  { echo -e "${BLUE}[$(date '+%H:%M:%S')] >>> $*${NC}"; }

usage() {
    echo "Usage: bash $0 <experiments_config_file> [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run    Only print the experimental plan"
    echo "  --gpu <id>   Covering all GPU IDs of the experiments"
    echo "  --stop-on-failure  Stop after an experiment fails (resume by default)"
    echo ""
    echo "Terminate: Press Ctrl+C directly"
    echo ""
    echo "Config format: One experiment per row, with il_run.sh parameters"
    echo "Example: --task_name_set pick_cube --policy_name ddpm_dit --num_epochs 100"
    exit 1
}

gpu_memory_usage() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

# Send a signal to the process group of a given PID. Isaac/Omni often fork multiple processes
# under the same PGID, so killing only the main PID is insufficient.
kill_process_group_of() {
    local pid=$1
    local sig=${2:-15}
    [ -z "$pid" ] && return
    local pgid
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$pgid" ] && return
    kill -"$sig" -- -"$pgid" 2>/dev/null || true
}

# Recursively kill a process and all its descendants (fallback for leftovers detached from PGID)
kill_descendants() {
    local pid=$1
    local sig=${2:-15}
    [ -z "$pid" ] && return
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    for child in $children; do
        kill_descendants "$child" "$sig"
    done
    kill -"$sig" "$pid" 2>/dev/null || true
}

# Optional: based on nvidia-smi compute processes, kill current-user leftovers that look like
# Isaac/Omni/RoboVerse (e.g. when double-forked away from the PGID)
roboverse_nvidia_orphan_cleanup() {
    [ "${ROBOVERSE_NVIDIA_CLEANUP:-0}" != "1" ] && return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local pid
    while IFS= read -r line; do
        pid=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]].*$//' | tr -d ' ')
        [ -z "$pid" ] || ! [[ "$pid" =~ ^[0-9]+$ ]] && continue
        [ -r "/proc/$pid" ] || continue
        local owner
        owner=$(stat -c %U "/proc/$pid" 2>/dev/null || true)
        [ "$owner" != "$USER" ] && continue
        local cmd
        cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || echo "")
        # Avoid matching the generic *python* to prevent killing other GPU processes such as IDE/Jupyter
        case "$cmd" in
            *isaac*|*omni*|*Kit*|*carb*|*RoboVerse*|*A2A_Flow*|*POET*|*metasim*|*roboverse_learn*)
                log_warn "ROBOVERSE_NVIDIA_CLEANUP: SIGKILL pid $pid"
                kill -9 "$pid" 2>/dev/null || true
                ;;
        esac
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
}

wait_for_gpu_release() {
    local prev_mem
    prev_mem=$(gpu_memory_usage)
    local attempts=0
    while [ $attempts -lt 10 ]; do
        sleep 2
        local curr_mem
        curr_mem=$(gpu_memory_usage)
        if [ -n "$curr_mem" ] && [ "$curr_mem" -lt "${prev_mem:-99999}" ] 2>/dev/null; then
            log_info "GPU memory released: ${prev_mem}MB -> ${curr_mem}MB"
            return 0
        fi
        attempts=$((attempts + 1))
    done
    log_warn "GPU memory may not be fully released (current: $(gpu_memory_usage)MB)"
}

# After evaluation, embed the success rate into the eval directory name
# e.g. 11250.ckpt_2026-03-23_13-33-49 -> 11250_sr92.ckpt_2026-03-23_13-33-49
rename_eval_dirs() {
    local marker_file=$1

    local stats_files
    stats_files=$(find "${PROJECT_ROOT}/il_outputs" \( -name "00_final_stats.txt" -o -name "final_stats.txt" \) -newer "$marker_file" 2>/dev/null)
    [ -z "$stats_files" ] && return 0

    while IFS= read -r stats_file; do
        local eval_dir
        eval_dir=$(dirname "$stats_file")
        local dir_name
        dir_name=$(basename "$eval_dir")
        local parent_dir
        parent_dir=$(dirname "$eval_dir")

        # Skip if _sr is already present
        [[ "$dir_name" == *_sr[0-9]*.ckpt_* ]] && continue
        # Must match the {name}.ckpt_{timestamp} format
        [[ "$dir_name" == *.ckpt_* ]] || continue

        local sr
        sr=$(grep "Average Success Rate:" "$stats_file" | grep -oP '[0-9]+\.[0-9]+' | head -1)
        [ -z "$sr" ] && continue

        local sr_tag
        if [[ "$sr" == 1.* ]] || [[ "$sr" == "1" ]]; then
            sr_tag="100"
        else
            sr_tag=$(echo "$sr" | sed -n 's/.*\.\([0-9][0-9]\).*/\1/p')
        fi
        [ -z "$sr_tag" ] && continue

        local ckpt_part="${dir_name%%.ckpt_*}"
        local rest="${dir_name#*.ckpt_}"
        local new_name="${ckpt_part}_sr${sr_tag}.ckpt_${rest}"

        if [ "$dir_name" != "$new_name" ]; then
            mv "$eval_dir" "${parent_dir}/${new_name}"
            log_info "Rename the evaluation directory: ${dir_name} -> ${new_name}"
        fi
    done <<< "$stats_files"
}

# Terminate a hung experiment process (escalation: wait -> SIGINT -> SIGTERM -> SIGKILL)
# Under setsid PGID = PID, so kill -- -PID kills the whole group.
graceful_kill() {
    local pid=$1

    log_info "Wait for the process to exit (${GRACE_PERIOD}s)..."
    local i=0
    while [ $i -lt $GRACE_PERIOD ]; do
        kill -0 "$pid" 2>/dev/null || { log_info "The process has exited"; return 0; }
        sleep 1
        i=$((i + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        log_info "Send SIGINT to the process group (PGID=$pid)..."
        kill -INT -- -"$pid" 2>/dev/null || true
        i=0
        while [ $i -lt $KILL_ESCALATION ]; do
            kill -0 "$pid" 2>/dev/null || { log_info "Exit after SIGINT"; return 0; }
            sleep 1
            i=$((i + 1))
        done
    fi

    if kill -0 "$pid" 2>/dev/null; then
        log_warn "Send SIGTERM to the process group (PGID=$pid)..."
        kill -TERM -- -"$pid" 2>/dev/null || true
        i=0
        while [ $i -lt $KILL_ESCALATION ]; do
            kill -0 "$pid" 2>/dev/null || { log_info "Exit after SIGTERM"; return 0; }
            sleep 1
            i=$((i + 1))
        done
    fi

    if kill -0 "$pid" 2>/dev/null; then
        log_error "Send SIGKILL to the process group (PGID=$pid)..."
        kill -KILL -- -"$pid" 2>/dev/null || true
        sleep 2
    fi

    ! kill -0 "$pid" 2>/dev/null
}

# ======================== Run a single experiment ========================
run_single_experiment() {
    local exp_idx=$1
    local exp_total=$2
    local exp_args=$3
    local log_file=$4

    log_step "Experiment [$exp_idx/$exp_total] Starts"
    log_info "Parameters: $exp_args"
    log_info "Log: $log_file"
    log_info "GPU Memory (before start): $(gpu_memory_usage)MB"

    local start_time
    start_time=$(date +%s)

    # Timestamp marker, used to locate newly generated eval dirs after the experiment finishes
    local marker_file
    marker_file=$(mktemp)

    # Use setsid to put the experiment in its own process group (PGID = the experiment bash PID).
    # This lets us safely kill -PGID after the experiment to clear any leftovers (including Isaac Sim
    # workers) without killing batch_run.sh itself. Ctrl+C is caught and cleaned up by the trap.
    setsid bash "${PROJECT_ROOT}/roboverse_learn/il/il_run.sh" $exp_args \
        > "$log_file" 2>&1 &
    CURRENT_EXP_PID=$!
    local cmd_pid=$CURRENT_EXP_PID

    log_info "Process PID: $cmd_pid (Independent process groups PGID=$cmd_pid)"

    # Background tail to show real-time output
    tail -f "$log_file" 2>/dev/null &
    CURRENT_TAIL_PID=$!

    # ---- Main monitoring loop ----
    local completed=false
    local completion_type="unknown"
    local last_size=0
    local stale_count=0

    while kill -0 "$cmd_pid" 2>/dev/null; do
        sleep "$POLL_INTERVAL"

        # Check whether the process has exited (it may have exited during sleep)
        kill -0 "$cmd_pid" 2>/dev/null || break

        # Detect evaluation completion
        if grep -q "FINAL RESULTS:" "$log_file" 2>/dev/null; then
            completed=true
            completion_type="eval_done"
            log_info "Detected that the assessment is complete (FINAL RESULTS)"
            break
        fi

        # Detect train.py forced exit
        if grep -q "All tasks finished. Forcing process exit" "$log_file" 2>/dev/null; then
            completed=true
            completion_type="forced_exit"
            log_info "A forced exit signal is detected"
            break
        fi

        # Detect normal completion of il_run.sh
        if grep -q "=== Completed all data collection" "$log_file" 2>/dev/null; then
            completed=true
            completion_type="script_done"
            log_info "Detected that the script has been completed"
            break
        fi

        # No-output timeout detection
        if [ "$STALE_TIMEOUT" -gt 0 ]; then
            local curr_size
            curr_size=$(wc -c < "$log_file" 2>/dev/null || echo 0)
            if [ "$curr_size" -eq "$last_size" ]; then
                stale_count=$((stale_count + POLL_INTERVAL))
                if [ $stale_count -ge $STALE_TIMEOUT ]; then
                    log_warn "No new output for more than ${STALE_TIMEOUT}s is determined to be suspended"
                    completion_type="stale_timeout"
                    break
                fi
            else
                stale_count=0
                last_size=$curr_size
            fi
        fi
    done

    # ---- Terminate hung processes ----
    if kill -0 "$cmd_pid" 2>/dev/null; then
        graceful_kill "$cmd_pid"
    fi

    # **** Core fix: unconditionally clean up the experiment process group ****
    # Even after the main process exits, Isaac Sim workers may linger as orphans on the GPU.
    # setsid guarantees PGID = cmd_pid, so this kill is safe and does not affect batch_run.
    log_info "Clean up the experimental process group (PGID=$cmd_pid)..."
    kill -TERM -- -"$cmd_pid" 2>/dev/null || true
    sleep 2
    kill -KILL -- -"$cmd_pid" 2>/dev/null || true
    sleep 1

    # Stop tail
    kill "$CURRENT_TAIL_PID" 2>/dev/null
    wait "$CURRENT_TAIL_PID" 2>/dev/null || true
    CURRENT_TAIL_PID=""

    # Get exit code
    wait "$cmd_pid" 2>/dev/null || true
    local exit_code=$?
    CURRENT_EXP_PID=""

    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    local duration_min=$(( duration / 60 ))
    local duration_sec=$(( duration % 60 ))

    # Wait for GPU memory to be released (optionally clean up orphan GPU processes)
    wait_for_gpu_release
    roboverse_nvidia_orphan_cleanup

    # ---- Extract results ----
    local success_rate="N/A"
    local sr_line
    sr_line=$(grep "FINAL RESULTS: Average Success Rate" "$log_file" 2>/dev/null | tail -1)
    if [ -n "$sr_line" ]; then
        success_rate=$(echo "$sr_line" | grep -oP '= \K[0-9.]+')
    fi

    # ---- Rename eval dirs (embed success rate into dir name) ----
    if [ "$success_rate" != "N/A" ]; then
        rename_eval_dirs "$marker_file"
    fi
    rm -f "$marker_file"

    # ---- Determine status ----
    local status
    if [ "$completed" = true ] && [ "$completion_type" != "stale_timeout" ]; then
        status="SUCCESS"
        log_info "Experiment [$exp_idx/$exp_total] completed | Duration: ${duration_min}m${duration_sec}s | Success Rate: ${success_rate}"
    elif [ "$completion_type" = "stale_timeout" ]; then
        status="TIMEOUT"
        log_warn "Experiment [$exp_idx/$exp_total] timed out | Duration: ${duration_min}m${duration_sec}s"
    else
        if [ $exit_code -eq 0 ]; then
            status="SUCCESS"
            log_info "Experiment [$exp_idx/$exp_total] exited normally | Duration: ${duration_min}m${duration_sec}s"
        else
            status="FAILED"
            log_error "Experiment [$exp_idx/$exp_total] failed (exit=$exit_code) | Duration: ${duration_min}m${duration_sec}s"
        fi
    fi

    # Write summary
    printf "%-4s | %-10s | %-6s | %5dm%02ds | %s\n" \
        "$exp_idx" "$status" "$success_rate" "$duration_min" "$duration_sec" \
        "$exp_args" >> "$SUMMARY_FILE"

    if [ "$status" = "FAILED" ] || [ "$status" = "TIMEOUT" ]; then
        return 1
    fi
    return 0
}

# ======================== Argument parsing ========================

if [ $# -lt 1 ]; then
    usage
fi

CONFIG_FILE="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift ;;
        --gpu)
            GPU_OVERRIDE="$2"
            shift 2 ;;
        --stop-on-failure)
            CONTINUE_ON_FAILURE=false
            shift ;;
        *)
            log_error "Unknown argument: $1"
            usage ;;
    esac
done

if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file does not exist: $CONFIG_FILE"
    exit 1
fi

# ======================== Read experiment config ========================

experiments=()
while IFS= read -r line; do
    line=$(echo "$line" | sed 's/#.*//' | xargs)
    [ -z "$line" ] && continue
    if [ -n "$GPU_OVERRIDE" ]; then
        line=$(echo "$line" | sed "s/--gpu [0-9]*//g")
        line="$line --gpu $GPU_OVERRIDE"
    fi
    experiments+=("$line")
done < "$CONFIG_FILE"

total=${#experiments[@]}
if [ "$total" -eq 0 ]; then
    log_error "No valid experiment configuration found in the config file"
    exit 1
fi

# ======================== Start running ========================

print_banner

log_info "Config file: $CONFIG_FILE"
log_info "Total experiments: $total"
log_info "Continue on failure: $CONTINUE_ON_FAILURE"
log_info "Termination: Ctrl+C"
[ -n "$GPU_OVERRIDE" ] && log_info "GPU override: $GPU_OVERRIDE"
echo ""

echo -e "${CYAN}=================== Experiment Plan ===================${NC}"
for i in "${!experiments[@]}"; do
    echo -e "  ${BLUE}[$((i+1))/$total]${NC} ${experiments[$i]}"
done
echo -e "${CYAN}=================================================${NC}"
echo ""

if [ "$DRY_RUN" = true ]; then
    log_info "Dry-run mode, not actually running"
    exit 0
fi

# Create log directory and summary file
BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG_DIR="${LOG_BASE_DIR}/batch_${BATCH_TIMESTAMP}"
mkdir -p "$BATCH_LOG_DIR"
SUMMARY_FILE="${BATCH_LOG_DIR}/summary.txt"

echo "Batch experiment summary - $(date)" > "$SUMMARY_FILE"
echo "Config file: $CONFIG_FILE" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
printf "%-4s | %-10s | %-6s | %8s | %s\n" "#" "Status" "SuccRate" "Duration" "Args" >> "$SUMMARY_FILE"
echo "---- | ---------- | ------ | -------- | ----" >> "$SUMMARY_FILE"

cd "$PROJECT_ROOT"

batch_start=$(date +%s)
succeeded=0
failed=0

for i in "${!experiments[@]}"; do
    exp_idx=$((i + 1))
    exp_args="${experiments[$i]}"
    exp_log="${BATCH_LOG_DIR}/exp_${exp_idx}.log"

    echo ""
    echo -e "${CYAN}======== Experiment $exp_idx / $total ========${NC}"

    if run_single_experiment "$exp_idx" "$total" "$exp_args" "$exp_log"; then
        succeeded=$((succeeded + 1))
    else
        failed=$((failed + 1))
        if [ "$CONTINUE_ON_FAILURE" = false ]; then
            log_error "Experiment failed, stopping subsequent experiments (--stop-on-failure)"
            break
        fi
        log_warn "Experiment failed, continuing to the next one..."
    fi

    if [ $exp_idx -lt $total ]; then
        log_info "Waiting 5s before starting the next experiment..."
        sleep 5
    fi
done

batch_end=$(date +%s)
batch_duration=$(( batch_end - batch_start ))
batch_min=$(( batch_duration / 60 ))
batch_sec=$(( batch_duration % 60 ))

# ======================== Final summary ========================

echo ""
echo "" >> "$SUMMARY_FILE"
echo -e "${CYAN}========== Batch experiments completed ==========${NC}"
echo ""
echo -e "  Succeeded: ${GREEN}$succeeded${NC}  Failed: ${RED}$failed${NC}  Total: $total"
echo -e "  Total duration: ${batch_min}m${batch_sec}s"
echo -e "  Summary file: ${BLUE}$SUMMARY_FILE${NC}"
echo -e "  Log directory: ${BLUE}$BATCH_LOG_DIR${NC}"
echo ""

echo "Succeeded: $succeeded  Failed: $failed  Total: $total" >> "$SUMMARY_FILE"
echo "Total duration: ${batch_min}m${batch_sec}s" >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

if [ $failed -gt 0 ]; then
    exit 1
fi
exit 0
