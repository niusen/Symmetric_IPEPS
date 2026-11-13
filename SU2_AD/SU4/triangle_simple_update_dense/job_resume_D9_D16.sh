#!/bin/sh
# Ordinary terminal: background job. Scheduler allocation: foreground job.
# Usage: sh job_resume_D9_D16.sh [original/output/root]
# Set SU4_FOREGROUND=1 to force foreground execution in an ordinary terminal.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
if [ "$#" -gt 1 ]; then
    printf 'Usage: sh %s [original/output/root]\n' "$0" >&2
    exit 1
fi
if [ "$#" -eq 1 ]; then
    SU4_OUTPUT_DIR=$1
fi
checkpoint_relative=continuation/L2x2_J1_traceless_D09_seed1234/checkpoint.jls
# Explicit paths take precedence. Otherwise locate the actual checkpoint beside
# this script before trying the historical absolute server path.
if [ -z "${SU4_OUTPUT_DIR:-}" ]; then
    for output_candidate in "$script_dir/long_runs_tensormap" "$script_dir/long_runs" \
        "$repo_dir/SU4/simple_update_dense/long_runs_tensormap" \
        "$repo_dir/SU4/simple_update_dense/long_runs" \
        /scratch/home/sniu/iPEPS_SU4_spin/triangle_SU4/simple_update_dense/long_runs_tensormap; do
        if [ -f "$output_candidate/$checkpoint_relative" ]; then
            SU4_OUTPUT_DIR=$output_candidate
            break
        fi
    done
    SU4_OUTPUT_DIR=${SU4_OUTPUT_DIR:-"$script_dir/long_runs_tensormap"}
fi
checkpoint="$SU4_OUTPUT_DIR/$checkpoint_relative"
if [ ! -f "$checkpoint" ]; then
    printf 'D=9 seed=1234 checkpoint not found: %s\nCurrent directory: %s\nScript directory: %s\n' \
        "$checkpoint" "$(pwd -P)" "$script_dir" >&2
    ls -ld "$SU4_OUTPUT_DIR" "$SU4_OUTPUT_DIR/continuation/L2x2_J1_traceless_D09_seed1234" >&2 || :
    printf 'From the original run directory, try: sh job_resume_D9_D16.sh "$(pwd -P)/long_runs_tensormap"\n' >&2
    exit 1
fi
if [ ! -r "$checkpoint" ]; then
    printf 'Checkpoint exists but is not readable: %s\n' "$checkpoint" >&2
    exit 1
fi
SU4_OUTPUT_DIR=$(CDPATH= cd -- "$SU4_OUTPUT_DIR" && pwd)
export SU4_OUTPUT_DIR
export SU4_BLAS_THREADS=${SU4_BLAS_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
julia_bin=${JULIA_BIN:-julia}
command -v "$julia_bin" >/dev/null 2>&1 || { printf 'Julia not found: %s\n' "$julia_bin" >&2; exit 1; }
case "$SU4_BLAS_THREADS" in
    ''|*[!0-9]*|0) printf 'SU4_BLAS_THREADS must be a positive integer.\n' >&2; exit 1 ;;
esac
case "${SU4_FOREGROUND:-0}" in
    0|1) ;;
    *) printf 'SU4_FOREGROUND must be 0 or 1.\n' >&2; exit 1 ;;
esac
mkdir -p "$SU4_OUTPUT_DIR/logs"
now=$(date +'%Y_%m_%d_%H_%M_%S')
log_prefix="$SU4_OUTPUT_DIR/logs/resume_D9_D16_${now}_$$"
printf 'Output: %s\nBLAS threads: %s\nstdout: %s.out\nstderr: %s.err\nexit status: %s.exit\n' \
    "$SU4_OUTPUT_DIR" "$SU4_BLAS_THREADS" "$log_prefix" "$log_prefix" "$log_prefix"
cd "$repo_dir"
# This shell worker waits for Julia, records its exit status, then exits with
# the same status. The .pid file identifies this worker, not the Julia child.
run_job() {
    run_status=0
    # time -v adds peak RSS when GNU time is available.
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -v -o "$log_prefix.resources" \
            "$julia_bin" --startup-file=no "$script_dir/Resume_D9_D16.jl" || run_status=$?
    else
        "$julia_bin" --startup-file=no "$script_dir/Resume_D9_D16.jl" || run_status=$?
    fi
    printf '%s\n' "$run_status" >"$log_prefix.exit"
    return "$run_status"
}

if [ "${SU4_FOREGROUND:-0}" = 1 ] || [ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}" ]; then
    printf 'Running in foreground.\n'
    run_job </dev/null >>"$log_prefix.out" 2>>"$log_prefix.err"
else
    run_job </dev/null >>"$log_prefix.out" 2>>"$log_prefix.err" &
    run_pid=$!
    printf '%s\n' "$run_pid" >"$log_prefix.pid"
    printf 'Started background job PID %s (worker waiting for Julia).\nPID file: %s.pid\n' "$run_pid" "$log_prefix"
fi
