#!/bin/sh
# Delegate process handling to the existing background/log/PID launcher.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export SU4_STAR_ENTRY="$script_dir/server/run_full_update.jl"
export SU4_STAR_JOB_NAME=honeycomb_FU
exec sh "$script_dir/run_star_su.sh" "$@"
