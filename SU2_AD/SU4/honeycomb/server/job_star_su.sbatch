#!/bin/bash
#SBATCH --job-name=su4_star
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=su4_star-%j.out
#SBATCH --error=su4_star-%j.err

# Resource settings above are an editable example, not a measured D=24 requirement.
# Add the partition/account required by your cluster before submission.
# One submission runs the SU schedule AND the configured CTMRG observables.
set -eu
run_dir=${SU4_STAR_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$run_dir/run_star_su.sh" ]; then
    if [ -f "$run_dir/../run_star_su.sh" ]; then
        run_dir="$run_dir/.."
    else
        run_dir="$run_dir/SU4/honeycomb"
    fi
fi
if [ ! -f "$run_dir/run_star_su.sh" ]; then
    printf 'Submit from SU2_AD or SU4/honeycomb, or set SU4_STAR_DIR.\n' >&2
    exit 1
fi
exec sh "$run_dir/run_star_su.sh" "$@"
