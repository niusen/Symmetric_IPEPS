#!/bin/sh
# Same launch style as the existing server sh files: detach, log, and record PID.
set -eu
# Outside the project layout, point to the EXISTING shared src; never copy it:
# export SU4_SRC_DIR=/path/to/existing/src
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
julia_bin=${JULIA_BIN:-julia}
foreground=${SU4_STAR_FOREGROUND:-0}
if [ "${1:-}" = "--foreground" ]; then
    foreground=1
    shift
fi
# Help / config previews stay interactive. Allocated scheduler jobs stay alive.
for arg in "$@"; do
    case "$arg" in --help|--dry-run) foreground=1 ;; esac
done
if [ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}" ]; then
    foreground=1
fi
if ! command -v "$julia_bin" >/dev/null 2>&1; then
    printf 'Julia not found. Set JULIA_BIN or load your server Julia environment.\n' >&2
    exit 1
fi
# JULIA_PROJECT selects the user's existing compatible package environment.
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
if [ -n "${SU4_STAR_BLAS_THREADS:-}" ]; then
    export OMP_NUM_THREADS=$SU4_STAR_BLAS_THREADS
    export MKL_NUM_THREADS=$SU4_STAR_BLAS_THREADS
fi
entry=${SU4_STAR_ENTRY:-"$script_dir/server/run_star_simple_update.jl"}
if [ -z "${SU4_STAR_ENTRY:-}" ] && [ ! -f "$entry" ] && [ -f "$script_dir/run_star_simple_update.jl" ]; then
    entry="$script_dir/run_star_simple_update.jl"
fi
if [ ! -f "$entry" ]; then
    printf 'Julia entry not found: %s\nSet SU4_STAR_ENTRY to run_star_simple_update.jl.\n' "$entry" >&2
    exit 1
fi
if [ "$foreground" = 1 ]; then
    exec "$julia_bin" --startup-file=no "$entry" --auto-resume "$@"
fi
command -v setsid >/dev/null 2>&1 || {
    printf 'setsid missing; use sh run_star_su.sh --foreground.\n' >&2
    exit 1
}
command -v nohup >/dev/null 2>&1 || {
    printf 'nohup missing; use sh run_star_su.sh --foreground.\n' >&2
    exit 1
}
log_dir=${SU4_STAR_LOG_DIR:-"$script_dir/logs"}
mkdir -p "$log_dir"
log_dir=$(CDPATH= cd -- "$log_dir" && pwd)
stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
prefix="$log_dir/star_SU_${stamp}"
nohup setsid "$julia_bin" --startup-file=no "$entry" --auto-resume "$@" \
    >"$prefix.out" 2>"$prefix.err" < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$prefix.pid"
printf 'Started PID %s\nstdout: %s.out\nstderr: %s.err\nRepeat the same sh command to resume after stopping.\n' \
    "$pid" "$prefix" "$prefix"
