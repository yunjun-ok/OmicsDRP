#!/usr/bin/env bash
# Stable, resumable, detached driver for the ablation sweep.
#
# Why: a full sweep is many GPU-hours; it must survive terminal/SSH close, and a
# crash or power-off must not lose progress. The Python runner is resumable at
# fold granularity (finished folds are skipped via their on-disk metrics), so
# this wrapper just needs to (a) run detached, (b) auto-restart on failure, and
# (c) prevent two sweeps clobbering the same output.
#
# Usage:
#   ./run_sweep.sh                 # all groups, background, auto-resume
#   ./run_sweep.sh --groups drug   # pass any run_ablations.py args through
#   GPU=0 ./run_sweep.sh           # pin a GPU
#   MAX_RETRIES=10 ./run_sweep.sh  # cap auto-restarts (default 20)
#
# Monitor:  tail -f Results/sweep_logs/sweep_*.log
# Stop:     kill "$(cat Results/sweep.pid)"     (progress is kept; rerun to resume)
set -uo pipefail

cd "$(dirname "$0")"                      # -> omicsdrp/scripts
ENV_NAME="${ENV_NAME:-omicsdrp}"
OUT_ROOT="${OUT_ROOT:-./Results}"
DATASET="${DATASET:-../../data}"
RUNNER="${RUNNER:-runner.py}"   # staged pipeline by default; set run_ablations.py for the OFAT grid
MAX_RETRIES="${MAX_RETRIES:-20}"
LOCK="$OUT_ROOT/sweep.lock"
PIDFILE="$OUT_ROOT/sweep.pid"
LOGDIR="$OUT_ROOT/sweep_logs"

mkdir -p "$OUT_ROOT" "$LOGDIR"
# export so the detached child inherits them
export ENV_NAME OUT_ROOT DATASET RUNNER MAX_RETRIES LOCK PIDFILE LOGDIR EMAIL_TO

# --- single-instance lock (atomic mkdir) ---
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "A sweep is already running (lock: $LOCK). If it crashed, remove the lock:"
    echo "    rmdir '$LOCK'"
    exit 1
fi
cleanup() { rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGDIR/sweep_$STAMP.log"
export STAMP GPU

run() {
    # shellcheck disable=SC1091
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "$ENV_NAME"
    [ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES="$GPU"
    export PYTHONUNBUFFERED=1

    echo "=== sweep start $STAMP (env=$ENV_NAME, out=$OUT_ROOT, gpu=${GPU:-all}) ==="
    local attempt=1
    # resumable: on any non-zero exit, restart; finished folds are skipped so it
    # picks up where it stopped. Stop on success (0) or after MAX_RETRIES.
    while true; do
        echo "--- attempt $attempt/$MAX_RETRIES @ $(date) ---"
        python "$RUNNER" --dataset_path "$DATASET" --out_root "$OUT_ROOT" "$@"
        local rc=$?
        if [ $rc -eq 0 ]; then
            echo "=== sweep COMPLETE @ $(date) ==="
            break
        fi
        echo "!!! runner exited rc=$rc @ $(date); resuming after 15s ..."
        attempt=$((attempt + 1))
        if [ $attempt -gt "$MAX_RETRIES" ]; then
            echo "=== gave up after $MAX_RETRIES attempts; rerun this script to continue ==="
            break
        fi
        sleep 15
    done
}

# Detach fully: survives terminal/SSH close (setsid + nohup), logs to file.
setsid bash -c "$(declare -f run cleanup); trap cleanup EXIT; run $*" \
    >"$LOG" 2>&1 < /dev/null &
CHILD=$!
echo "$CHILD" > "$PIDFILE"
# the detached child owns the lock now; don't remove it from this shell
trap - EXIT
echo "Sweep launched detached. PID $CHILD"
echo "  log:  $LOG"
echo "  tail: tail -f $LOG"
echo "  stop: kill $CHILD   (progress kept; rerun this script to resume)"
