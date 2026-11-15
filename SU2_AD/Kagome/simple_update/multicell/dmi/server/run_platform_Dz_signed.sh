#!/bin/sh
# Pure Julia CPU; physics settings are in run_platform_Dz_signed.jl.
set -eu
cd "$(dirname "$0")"
task_julia=${JULIA_BIN:-julia}
command -v "$task_julia" >/dev/null 2>&1 || { echo 'Julia not found; set JULIA_BIN.' >&2; exit 1; }
export JULIA_NUM_THREADS=10 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
mkdir -p ../data/logs
task_stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
task_log="../data/logs/platform_Dz_signed_${task_stamp}.out"
nohup "$task_julia" --startup-file=no run_platform_Dz_signed.jl > "$task_log" 2>&1 < /dev/null &
task_pid=$!
printf '%s\n' "$task_pid" > "../data/logs/platform_Dz_signed_${task_stamp}.pid"
printf 'PID=%s\nLog: %s/%s\nStop: kill -TERM %s\nResume: rerun this sh with unchanged settings.\n' "$task_pid" "$PWD" "$task_log" "$task_pid"
