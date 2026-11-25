#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
julia_bin="${JULIA_BIN:-julia}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mode="${1:-help}"
if (($#)); then shift; fi
case "$mode" in
  ipeps-su)
    exec "$julia_bin" "$app_dir/../iPEPS/SimpleUpdate.jl" "$@" ;;
  spin)
    exec "$julia_bin" "$app_dir/spin/run.jl" "$@" ;;
  setup)
    exec "$julia_bin" -e 'using TensorKit, MAT, JLD2, Zygote, ChainRulesCore, KrylovKit, HalfIntegers; println("Julia ", VERSION, "  TensorKit ", Base.pkgversion(TensorKit), "  project=", Base.active_project())' ;;
  gfpeps)
    exec "$julia_bin" "$app_dir/gfpeps/run.jl" "$@" ;;
  su)
    exec "$julia_bin" "$app_dir/SimpleUpdate_iPESS_Z2_ob.jl" "$@" ;;
  fu)
    exec "$julia_bin" "$app_dir/optimization/FullUpdate_iPESS_Z2_ob.jl" "$@" ;;
  ad)
    exec "$julia_bin" "$app_dir/optimization/Gradient_iPESS_Z2_ob.jl" "$@" ;;
  *)
    echo 'Usage: bash server/run.sh setup | ipeps-su MODEL INPUT OUTPUT | spin {su|fu|ad} CONFIG | {gfpeps|su|fu|ad} CONFIG [--check (gfpeps only)]'
    [[ "$mode" == help ]] || exit 2 ;;
esac
