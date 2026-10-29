# Server entry point; reusable functions live in src/.
if !isdefined(@__MODULE__,:DenseSU4LongRun)
    include(joinpath(@__DIR__,"../../src/bosonic/triangle/su4_dense/LongRun.jl"))
end
if abspath(PROGRAM_FILE)==@__FILE__
    threads=parse(Int,get(ENV,"SU4_BLAS_THREADS",get(ENV,"SLURM_CPUS_PER_TASK",string(min(8,Sys.CPU_THREADS)))))
    output=get(ENV,"SU4_OUTPUT_DIR",DenseSU4LongRun.DEFAULT_OUTPUT_DIR)
    rows=DenseSU4LongRun.run_batch(DenseSU4LongRun.LongRunConfig(blas_threads=threads,output_dir=output))
    any(r->!r.complete,rows) && exit(1)
end
