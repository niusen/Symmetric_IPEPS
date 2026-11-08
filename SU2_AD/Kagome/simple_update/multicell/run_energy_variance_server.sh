#!/bin/sh
set -eu
cd "$(dirname "$0")"
JULIA_BIN="${JULIA_BIN:-julia}"
export JULIA_NUM_THREADS=10 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
[ "$#" -eq 0 ] || { echo 'Usage: sh run_energy_variance_server.sh' >&2; exit 2; }
command -v "$JULIA_BIN" >/dev/null 2>&1 || { echo 'Julia not found; set JULIA_BIN.' >&2; exit 1; }
command -v setsid >/dev/null 2>&1 || { echo 'setsid is required for background submission.' >&2; exit 1; }

# Preflight loads settings/stdlibs only. It does NOT run CTMRG or evolve states.
"$JULIA_BIN" --startup-file=no --project=. -e '
    const KAGOME_VARIANCE_SETTINGS_ONLY=true
    include("run_energy_variance_server.jl")
    isfile("Project.toml") || error("Missing Project.toml")
    for name in ("KagomeEnergyVariance.jl","variance_operators.jl","variance_contraction.jl",
                 "variance_array_sweeps.jl","variance_workflow.jl","variance_batch_plan.jl","variance_batch.jl")
        p=joinpath(dirname(VARIANCE_LIBRARY),name)
        isfile(p) || error("Missing source: "*p)
    end
    c=variance_settings
    jobs=KagomeVariancePlan.variance_jobs(c)
    println("Ready: ",length(jobs)," contractions / ",length(jobs)*length(c.sizes)," L estimates; D=",c.dims,"; chi=",c.chis)
    println("One contraction Lmax=",c.contraction_size," extracts L=",c.sizes,"; no independent R setting.")
    println("CPU, 10 threads; completed (D,chi) cases resume automatically.")
    println("Results: ",c.out)
'
mkdir -p logs
stamp=$(date +'%Y_%m_%d_%H_%M_%S')_$$
log="logs/energy_variance_batch_${stamp}.out"
pidfile="logs/energy_variance_batch_${stamp}.pid"
# No watchdog/restart. One Julia process group; log survives terminal logout.
set -- "$JULIA_BIN" --startup-file=no --project=. --threads=10 run_energy_variance_server.jl
# GNU time reports whole-job wall/CPU time and maximum resident memory in log.
# Julia's per-phase/per-D timing remains available even without this utility.
if [ -x /usr/bin/time ]; then
    set -- /usr/bin/time -v "$@"
fi
nohup setsid "$@" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pidfile"
printf 'Submitted PID/process-group=%s\nLog: %s/%s\nPID file: %s/%s\nWatch: tail -f "%s/%s"\nStop: kill -TERM -- -%s\n' \
    "$pid" "$PWD" "$log" "$PWD" "$pidfile" "$PWD" "$log" "$pid"
