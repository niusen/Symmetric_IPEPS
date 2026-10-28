# Measurement entry point; reusable functions live in src/.
if !isdefined(@__MODULE__,:DenseSU4LongRun)
    include(joinpath(@__DIR__,"../../src/bosonic/triangle/su4_dense/LongRun.jl"))
end
using .DenseSU4LongRun: evaluate_checkpoint

if abspath(PROGRAM_FILE)==@__FILE__
    isempty(ARGS) && error("Usage: julia Evaluate_observables.jl <checkpoint.jls> [output_prefix]")
    prefix=length(ARGS)>=2 ? ARGS[2] : splitext(ARGS[1])[1]*"_ctmrg_observables"
    evaluate_checkpoint(ARGS[1];output_prefix=prefix)
end
