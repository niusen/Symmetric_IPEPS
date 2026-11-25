#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
if [ "${1:-}" = "--remote" ]; then
    if [ "$#" -ne 3 ]; then
        echo 'Usage: sh submit.sh --remote user@host /absolute/path/to/SU2_AD' >&2
        exit 2
    fi
    host=$2
    dest=$3
    case "$host" in ''|-*) echo 'Expected an SSH host.' >&2; exit 2 ;; esac
    case "$dest" in /*) ;; *) echo 'Expected an absolute server directory.' >&2; exit 2 ;; esac
    root=$(CDPATH= cd "$here/../../../.." && pwd)
    # Quote the directory for the remote login shell, including literal apostrophes.
    quoted=$(printf '%s' "$dest" | sed "s/'/'\\\\''/g")
    ssh "$host" "mkdir -p -- '$quoted'"
    rsync -rlts --files-from="$here/transfer.list" -- "$root/" "$host:$dest/"
    exec ssh "$host" "cd '$quoted' && sh Kagome/fermionic/jobs/parton_D8_su/submit.sh"
fi
exec sh "$here/run.sh" submit "$@"
