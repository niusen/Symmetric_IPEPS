# Keep cfg.Ds=4:16: it defines previous-D initialization and checkpoint signatures.
include(joinpath(@__DIR__, "../../src/bosonic/triangle/su4_dense/Resume.jl"))
if abspath(PROGRAM_FILE) == @__FILE__
    haskey(ENV, "SU4_OUTPUT_DIR") || error("Set SU4_OUTPUT_DIR to the original output root")
    threads = parse(Int, get(ENV, "SU4_BLAS_THREADS", get(ENV, "SLURM_CPUS_PER_TASK", "8")))
    cfg = DenseSU4LongRun.LongRunConfig(output_dir=ENV["SU4_OUTPUT_DIR"], blas_threads=threads)
    DenseSU4Resume.resume_batch(cfg)
end
