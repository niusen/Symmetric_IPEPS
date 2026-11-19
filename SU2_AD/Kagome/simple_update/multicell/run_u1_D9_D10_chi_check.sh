#!/bin/sh
set -eu
cd "$(dirname "$0")"
# All physical/numerical settings are in the Julia file, including threads.
JULIA_BIN="${JULIA_BIN:-julia}"
command -v "$JULIA_BIN" >/dev/null 2>&1 || { echo 'Julia not found; set JULIA_BIN to its executable.' >&2; exit 1; }
# Read only the settings tuple: this does not load TensorKit or run measurements.
threads=$("$JULIA_BIN" --startup-file=no -e 'const KAGOME_SETTINGS_ONLY=true; include("run_u1_D9_D10_chi_check.jl"); print(chi_check_settings.threads)')
case "$threads" in ''|*[!0-9]*) echo 'Invalid thread count in Julia settings.' >&2; exit 1;; esac
[ "$threads" -gt 0 ] || { echo 'Thread count must be positive.' >&2; exit 1; }
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
if [ "${1:-}" = "--foreground" ]; then
    exec "$JULIA_BIN" --startup-file=no --threads="$threads" run_u1_D9_D10_chi_check.jl
fi
command -v setsid >/dev/null 2>&1 || { echo 'setsid missing; use sh run_u1_D9_D10_chi_check.sh --foreground.' >&2; exit 1; }
mkdir -p logs
stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
log="logs/u1_D9_D10_chi_check_${stamp}.out"
# One Julia process group; no automatic restart loop.
nohup setsid "$JULIA_BIN" --startup-file=no --threads="$threads" run_u1_D9_D10_chi_check.jl > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "logs/u1_D9_D10_chi_check_${stamp}.pid"
printf 'Submitted PID/process-group=%s\nLog: %s/%s\nStop: kill -TERM -- -%s\nResume: rerun this sh with unchanged Julia settings.\n' "$pid" "$PWD" "$log" "$pid"
