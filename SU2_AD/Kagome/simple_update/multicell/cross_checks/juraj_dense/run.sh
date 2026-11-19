#!/bin/sh
set -eu
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [ "${1:-}" != "--foreground" ]; then
    command -v setsid >/dev/null 2>&1 || { echo 'Use sh run.sh --foreground on systems without setsid.' >&2; exit 1; }
    mkdir -p logs
    stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
    log="logs/juraj_dense_${stamp}.out"
    nohup setsid sh "$0" --foreground > "$log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "logs/juraj_dense_${stamp}.pid"
    printf 'Submitted PID/process-group=%s\nLog: %s/%s\nStop: kill -TERM -- -%s\n' "$pid" "$PWD" "$log" "$pid"
    exit 0
fi
exec "$PYTHON_BIN" -u run_server.py
