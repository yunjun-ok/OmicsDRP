#!/usr/bin/env bash
# Runs GradientSHAP.py in the background with logging + a pidfile.
# Usage: ./run_gradientshap.sh [args passed through to GradientSHAP.py]
# Monitor: tail -f logs/gradientshap_*.log
# Stop:    kill "$(cat gradientshap.pid)"
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/gradientshap_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$HERE/gradientshap.pid"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate omicsdrp

nohup python -u GradientSHAP.py "$@" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Started GradientSHAP.py (pid $(cat "$PID_FILE")), logging to $LOG_FILE"
