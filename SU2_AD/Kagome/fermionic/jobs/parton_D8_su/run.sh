#!/usr/bin/env bash
# Also accept "sh run.sh ..." on systems where /bin/sh is dash.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
jl=("${JULIA_BIN:-julia}")
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
out="${KAGOME_RUN_ROOT:-$(cd "$here/../../iPESS" && pwd)/results/parton_D8_su}"

usage() {
    echo 'Usage: sh run.sh check | all | run FAMILY MODEL | submit [FAMILY MODEL] | report' >&2
    exit 2
}

check_case() {
    case "$1" in ipeps|ipess) ;; *) usage ;; esac
    case "$2" in orig|mild|zero) ;; *) usage ;; esac
}

log_path() {
    printf '%s/logs/%s_%s_%s.log\n' "$out" "$1" "$(date +%Y%m%d_%H%M%S)" "$$"
}

run_one() {
    local log
    check_case "$1" "$2"
    log=$(log_path "${1}_${2}")
    mkdir -p -- "$out/logs" || return 1
    "${jl[@]}" "$here/task.jl" run "$1" "$2" 2>&1 | tee "$log"
}

submit() {
    local name log
    local -a command
    case $# in
        0) name=all; command=(bash "$here/run.sh" all) ;;
        2) check_case "$1" "$2"
           name="${1}_${2}"; command=("${jl[@]}" "$here/task.jl" run "$1" "$2") ;;
        *) usage ;;
    esac
    command -v "${jl[0]}" >/dev/null || { echo "Julia not found: ${jl[0]}" >&2; return 1; }
    mkdir -p -- "$out/logs"
    log=$(log_path "$name")
    # Open the log before the fork so a redirection error fails the submission.
    exec 3>"$log"
    nohup "${command[@]}" </dev/null >&3 2>&1 3>&- &
    printf 'Submitted %s  PID=%s (launcher)  node=%s\nLog: %s\n' "$name" "$!" "$(hostname)" "$log"
    exec 3>&-
}

case "${1:-}" in
    setup|check) # Accept the older setup command as a check of the installed environment.
            [[ $# == 1 ]] || usage
            "${jl[@]}" "$here/task.jl" check ;;
    report)
            [[ $# == 1 ]] || usage
            "${jl[@]}" "$here/task.jl" "$1" ;;
    run)    [[ $# == 3 ]] || usage
            run_one "$2" "$3" ;;
    submit) shift; submit "$@" ;;
    all)    [[ $# == 1 ]] || usage
            "${jl[@]}" "$here/task.jl" check
            failed=0
            for family in ipeps ipess; do
                for model in orig mild zero; do
                    if ! run_one "$family" "$model"; then
                        echo "Failed: $family $model (see $out/logs)" >&2
                        failed=1
                    fi
                done
            done
            "${jl[@]}" "$here/task.jl" report
            exit "$failed" ;;
    *)      usage ;;
esac
