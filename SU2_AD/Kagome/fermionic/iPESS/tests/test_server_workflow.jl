using Test, TOML
include(joinpath(@__DIR__,"../../../../src/fermionic/kagome/gfpeps/workflows/run_config.jl"))

@testset "Projected server workflow, provenance and restart" begin
    template=read_gfpeps_job(joinpath(@__DIR__,"../gfpeps/configs/heisenberg_ipess_D4.toml"))
    mktempdir() do directory
        config=deepcopy(template.config)
        config["job"]["input"]=template.input
        config["job"]["output"]="first"
        config["job"]["blas_threads"]=1
        config["ctm"]["chi"]=[4]
        config["ctm"]["maxiter"]=2
        path=joinpath(directory,"job.toml")
        open(io->TOML.print(io,config),path,"w")
        output=run_gfpeps_config(path)
        record=TOML.parsefile(joinpath(output,"heisenberg_chi4_cut1.0e-10.toml"))
        @test isfinite(record["energy_per_site"])
        @test record["D"]==4 && record["family"]=="ipess"
        @test TOML.parsefile(joinpath(output,"job.toml"))["tensors_sha256"]==template.tensor_sha256
        @test_throws ErrorException run_gfpeps_config(path)
        # Environment-only restart is supported, not just completed result files.
        config["job"]["warmstart"]="first/heisenberg_chi4_cut1.0e-10_environment.jld2"
        config["job"]["output"]="restart"
        open(io->TOML.print(io,config),path,"w")
        output=run_gfpeps_config(path)
        @test isfile(joinpath(output,"heisenberg_chi4_cut1.0e-10.jld2"))
        @test read_gfpeps_job(path).tensor_sha256==template.tensor_sha256
    end
end
