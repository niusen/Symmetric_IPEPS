#!/bin/sh
# Launch the 2x2 triangular SU(4) dense iPESS batch: J term, D=4,...,16.
# Keep this script beside LongRun_D4_D16.jl in the complete repository.
# Use the same Julia package environment as the existing CTMRG calculations.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
entry="$script_dir/LongRun_D4_D16.jl"

# Optional overrides before launching:
# export JULIA_BIN=/path/to/julia
# export JULIA_PROJECT=/path/to/existing/julia/environment
# export SU4_BLAS_THREADS=16
# export SU4_OUTPUT_DIR=/scratch/your_user/su4_results
julia_bin=${JULIA_BIN:-julia}
export SU4_BLAS_THREADS=${SU4_BLAS_THREADS:-${SLURM_CPUS_PER_TASK:-8}}
export SU4_OUTPUT_DIR=${SU4_OUTPUT_DIR:-"$script_dir/long_runs_tensormap"}

if ! command -v "$julia_bin" >/dev/null 2>&1; then
    printf 'Julia executable not found: %s\nSet JULIA_BIN or load the server Julia environment first.\n' "$julia_bin" >&2
    exit 1
fi
if [ ! -f "$entry" ] || [ ! -f "$repo_dir/src/bosonic/CTMRG_unitcell.jl" ]; then
    printf 'Keep this script in SU4/simple_update_dense/ within the complete repository.\n' >&2
    exit 1
fi
case "$SU4_BLAS_THREADS" in
    ''|*[!0-9]*|0) printf 'SU4_BLAS_THREADS must be a positive integer.\n' >&2; exit 1 ;;
esac

# Resolve a user-supplied relative output path before changing directory.
mkdir -p "$SU4_OUTPUT_DIR"
SU4_OUTPUT_DIR=$(CDPATH= cd -- "$SU4_OUTPUT_DIR" && pwd)
export SU4_OUTPUT_DIR
mkdir -p "$SU4_OUTPUT_DIR/logs"
now=$(date +'%Y_%m_%d_%H_%M_%S')
log_prefix="$SU4_OUTPUT_DIR/logs/su4_D4_D16_${now}_$$"

# Julia itself honors JULIA_PROJECT; no machine-specific environment is assumed.
cd "$repo_dir"

printf 'Output: %s\nBLAS threads: %s\nstdout: %s.out\nstderr: %s.err\n' \
    "$SU4_OUTPUT_DIR" "$SU4_BLAS_THREADS" "$log_prefix" "$log_prefix"

# Inside a scheduler allocation, keep Julia in the foreground so the job stays
# alive and receives its exit status. SU4_FOREGROUND=1 also enables this mode.
if [ "${SU4_FOREGROUND:-0}" = 1 ] || [ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${LSB_JOBID:-}" ]; then
    exec "$julia_bin" --startup-file=no "$entry" >>"$log_prefix.out" 2>>"$log_prefix.err"
fi

# Ordinary 'sh job_su4_D4_D16.sh': call Julia directly in the background,
# matching the original submission script.
"$julia_bin" --startup-file=no "$entry" >>"$log_prefix.out" 2>>"$log_prefix.err" &
run_pid=$!
printf '%s\n' "$run_pid" >"$log_prefix.pid"
printf 'Started PID %s\nRepeat this command with the same output directory to resume a stopped run.\n' "$run_pid"
