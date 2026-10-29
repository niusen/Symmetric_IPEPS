#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Physical/numerical settings are all in run_u1_D4_D10_server.jl.
JULIA_BIN="${JULIA_BIN:-julia}"
export JULIA_NUM_THREADS=10 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
if [ "${1:-}" = "--foreground" ]; then
    exec "$JULIA_BIN" --startup-file=no run_u1_D4_D10_server.jl
fi
command -v "$JULIA_BIN" >/dev/null 2>&1 || { echo 'Julia not found; set JULIA_BIN to its executable.' >&2; exit 1; }
command -v setsid >/dev/null 2>&1 || { echo 'setsid missing; use sh run_u1_D4_D10_server.sh --foreground.' >&2; exit 1; }
mkdir -p logs
stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
log="logs/u1_D4_D10_${stamp}.out"
# One Julia process group; no auto-restart loop.
nohup setsid "$JULIA_BIN" --startup-file=no run_u1_D4_D10_server.jl > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "logs/u1_D4_D10_${stamp}.pid"
printf 'Submitted PID/process-group=%s\nLog: %s/%s\nStop the whole job: kill -TERM -- -%s\nResume: rerun this sh with unchanged Julia settings.\n' "$pid" "$PWD" "$log" "$pid"
