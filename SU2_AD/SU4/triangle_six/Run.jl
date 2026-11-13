include(joinpath(@__DIR__,"../../src/bosonic/triangle/su4_six/TriangleSU4Six.jl"))
using .TriangleSU4Six
if abspath(PROGRAM_FILE)==@__FILE__
    backend=Symbol(get(ENV,"SU4_SIX_BACKEND","dense"))
    D=parse(Int,get(ENV,"SU4_DMAX",backend==:su4 ? "7" : "4"))
    cfg=SixRunConfig(;backend,Dmax=D,seed=parse(Int,get(ENV,"SU4_SEED","1234")),
        blas_threads=parse(Int,get(ENV,"SU4_BLAS_THREADS",get(ENV,"SLURM_CPUS_PER_TASK","8"))),
        output_dir=get(ENV,"SU4_OUTPUT_DIR",joinpath(@__DIR__,"results")))
    run_simulation(cfg)
end
