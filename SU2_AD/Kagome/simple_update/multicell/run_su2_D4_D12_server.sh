#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Physical/numerical settings are all in run_su2_D4_D12_server.jl.
JULIA_BIN="${JULIA_BIN:-julia}"
export JULIA_NUM_THREADS=10 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
if [ "${1:-}" = "--foreground" ]; then
    exec "$JULIA_BIN" --startup-file=no server_bootstrap.jl
fi
command -v "$JULIA_BIN" >/dev/null 2>&1 || { echo 'Julia not found; set JULIA_BIN to its executable.' >&2; exit 1; }
command -v setsid >/dev/null 2>&1 || { echo 'setsid missing; use sh run_su2_D4_D12_server.sh --foreground.' >&2; exit 1; }
mkdir -p logs
stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
log="logs/su2_D4_D12_${stamp}.out"
# One process group includes Julia and its GPU measurement child. No auto restart.
nohup setsid "$JULIA_BIN" --startup-file=no server_bootstrap.jl > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "logs/su2_D4_D12_${stamp}.pid"
printf 'Submitted PID/process-group=%s\nLog: %s/%s\nStop the whole job: kill -TERM -- -%s\nResume: rerun this sh with unchanged Julia settings.\n' "$pid" "$PWD" "$log" "$pid"
