include("config.jl")
include("../bootstrap.jl")
include("measurements.jl")

"""Run a frozen GFPEPS job serially and preserve its input/config provenance."""
function run_gfpeps_config(path)
    p=read_gfpeps_job(path)
    # A fresh output directory prevents accidental replacement of an earlier run.
    isdir(p.output) && !isempty(readdir(p.output)) && error("Output is not empty: $(p.output). Choose a new job.output directory.")
    data=matread(joinpath(p.input,"tensors.mat"))
    2^Int(data["m"])==p.D || error("Configured D differs from tensors.mat")
    expected=p.family=="ipess" ? "site_0_B" : "cell_0"
    haskey(data,expected) || error("Configured family differs from tensors.mat")
    BLAS.set_num_threads(get(p.job,"blas_threads",1))
    mkpath(p.output)
    provenance=Dict("config"=>p.config,"tensors_sha256"=>p.tensor_sha256,
        "input_directory"=>p.input,"julia_version"=>string(VERSION),
        "tensorkit_version"=>string(Base.pkgversion(TensorKit)),
        "started_at"=>string(now()),"source_config"=>p.path)
    open(io->TOML.print(io,provenance),joinpath(p.output,"job.toml"),"w")
    for chi in p.chis
        options=(;cutoff=get(p.ctm,"cutoff",1e-8),maxiter=get(p.ctm,"maxiter",100),
            tol=get(p.ctm,"tol",1e-7),gauge_sweeps=get(p.ctm,"gauge_sweeps",0),
            update_method=get(p.ctm,"update_method","continuous_update"),
            warmstart=p.warmstart,directory_override=p.input,output_directory=p.output)
        if p.mode=="heisenberg"
            run_spin_energy(p.family,p.D,chi;options...,
                target_mz_over_saturation=get(p.job,"target_mz_over_saturation",1/9),
                magnetization_tolerance=get(p.job,"magnetization_tolerance",1e-6))
        else
            validate_saved(p.family,p.D,chi;options...,save_environment=true)
        end
    end
    p.output
end
