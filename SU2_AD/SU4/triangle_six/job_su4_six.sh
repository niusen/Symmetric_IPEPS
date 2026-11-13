#!/bin/sh
# Six-dimensional SU(4), two particles/site, triangular iPESS.
# Usage: sh job_su4_six.sh dense|su4
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
[ "$#" -le 1 ] || { printf 'Usage: sh %s [dense|su4]\n' "$0" >&2; exit 1; }
export SU4_SIX_BACKEND=${1:-${SU4_SIX_BACKEND:-dense}}
case "$SU4_SIX_BACKEND" in
    dense) default_D=4 ;;
    su4) default_D=7 ;;
    *) printf 'Backend must be dense or su4.\n' >&2; exit 1 ;;
esac
export SU4_DMAX=${SU4_DMAX:-$default_D}
export SU4_SEED=${SU4_SEED:-1234}
export SU4_BLAS_THREADS=${SU4_BLAS_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
export SU4_OUTPUT_DIR=${SU4_OUTPUT_DIR:-"$script_dir/results"}
julia_bin=${JULIA_BIN:-julia}
command -v "$julia_bin" >/dev/null 2>&1 || { printf 'Julia not found: %s\n' "$julia_bin" >&2; exit 1; }
for count in "$SU4_DMAX" "$SU4_BLAS_THREADS"; do
    case "$count" in ''|*[!0-9]*|0) printf 'Dmax and BLAS threads must be positive integers.\n' >&2; exit 1 ;; esac
done
case "${SU4_FOREGROUND:-0}" in 0|1) ;; *) printf 'SU4_FOREGROUND must be 0 or 1.\n' >&2; exit 1 ;; esac
mkdir -p "$SU4_OUTPUT_DIR"
SU4_OUTPUT_DIR=$(CDPATH= cd -- "$SU4_OUTPUT_DIR" && pwd)
export SU4_OUTPUT_DIR
mkdir -p "$SU4_OUTPUT_DIR/logs"
now=$(date +'%Y_%m_%d_%H_%M_%S')
log_prefix="$SU4_OUTPUT_DIR/logs/six_${SU4_SIX_BACKEND}_D${SU4_DMAX}_${now}_$$"
printf 'Backend: %s\nDmax: %s\nstdout: %s.out\nstderr: %s.err\n' "$SU4_SIX_BACKEND" "$SU4_DMAX" "$log_prefix" "$log_prefix"
cd "$repo_dir"
run_job() {
    status=0
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -v -o "$log_prefix.resources" \
            "$julia_bin" --startup-file=no "$script_dir/Run.jl" || status=$?
    else
        "$julia_bin" --startup-file=no "$script_dir/Run.jl" || status=$?
    fi
    printf '%s\n' "$status" >"$log_prefix.exit"
    return "$status"
}
if [ "${SU4_FOREGROUND:-0}" = 1 ] || [ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}" ]; then
    run_job </dev/null >>"$log_prefix.out" 2>>"$log_prefix.err"
else
    run_job </dev/null >>"$log_prefix.out" 2>>"$log_prefix.err" &
    run_pid=$!
    printf '%s\n' "$run_pid" >"$log_prefix.pid"
    printf 'Started background job PID %s (worker waiting for Julia).\n' "$run_pid"
fi
