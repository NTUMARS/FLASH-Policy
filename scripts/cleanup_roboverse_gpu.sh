#!/usr/bin/env bash
# Manual cleanup: terminate current-user processes that occupy GPU and whose cmdline
# looks like this repo / Isaac / Omniverse leftovers.
# Use when GPU memory remains held after batch_run or training is interrupted.
#
# Usage:
#   bash scripts/cleanup_roboverse_gpu.sh          # Perform cleanup
#   bash scripts/cleanup_roboverse_gpu.sh --dry-run # Only print PIDs that would be killed
#
# Danger: SIGKILLs matched processes. Do not use while other important GPU tasks are running.

set -euo pipefail

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not available"
    exit 1
fi

killed=0
while IFS= read -r line; do
    pid=$(echo "$line" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/,"",$1); print $1}')
    [ -z "$pid" ] || ! [[ "$pid" =~ ^[0-9]+$ ]] && continue
    [ -r "/proc/$pid" ] || continue
    owner=$(stat -c %U "/proc/$pid" 2>/dev/null || true)
    [ "$owner" != "$USER" ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || echo "")
    case "$cmd" in
        *isaac*|*omni*|*Kit*|*carb*|*RoboVerse*|*A2A_Flow*|*POET*|*metasim*|*roboverse_learn*)
            if [ "$DRY_RUN" = true ]; then
                echo "would kill pid=$pid $cmd"
            else
                echo "SIGKILL pid=$pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
            killed=$((killed + 1))
            ;;
    esac
done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run: $killed process(es) matched."
else
    echo "Done. Killed or attempted: $killed process(es)."
fi
