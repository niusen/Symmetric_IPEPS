# Runtime data

`logs/` preserves run logs and historical PID records; new launchers also write
here. A saved PID file does not imply that a job is still running.

`tests/before_layout_20261114.toml` is the preserved small-test baseline from
before directory reorganization. `tests/last_result.toml` is written by the
current small-test suite. The relocation manifest records original/current
paths and SHA256 hashes at the time of each move.

Numerical checkpoints and full experiment results remain in the existing
multicell `results/` directories. Exclude this `data/` folder when uploading code.
