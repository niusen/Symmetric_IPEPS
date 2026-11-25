using Test, TOML
include(joinpath(@__DIR__,"../../../../src/fermionic/kagome/gfpeps/workflows/config.jl"))

@testset "Portable GFPEPS configurations" begin
    configs=joinpath(@__DIR__,"../gfpeps/configs")
    for filename in readdir(configs;join=true)
        p=read_gfpeps_job(filename)
        @test isabspath(p.input) && isabspath(p.output)
        @test p.input!=p.output
        @test length(p.tensor_sha256)==64
    end
    mktempdir() do dir
        path=joinpath(dir,"job.toml")
        template=read_gfpeps_job(joinpath(configs,"heisenberg_ipess_D4.toml"))
        config=deepcopy(template.config)
        config["job"]["input"]=template.input
        config["job"]["output"]="new_results"
        open(io->TOML.print(io,config),path,"w")
        cd(tempdir()) do
            @test read_gfpeps_job(path).output==joinpath(dir,"new_results")
        end
        for invalid in ([0],[32,16],[16,16],Int[])
            config["ctm"]["chi"]=invalid
            open(io->TOML.print(io,config),path,"w")
            @test_throws ErrorException read_gfpeps_job(path)
        end
        config["ctm"]["chi"]=[16]
        config["job"]["input"]=joinpath(dir,"missing")
        open(io->TOML.print(io,config),path,"w")
        @test_throws ErrorException read_gfpeps_job(path)
    end
end
