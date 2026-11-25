using Test
include(joinpath(@__DIR__,"..","optimization","run_optimization.jl"))

@testset "Optimization checkpoint and resume" begin
    mktempdir() do directory
        for method in (:full_update,:variational)
            example = method==:full_update ? "smoke_full_update.toml" : "smoke_variational.toml"
            config = TOML.parsefile(joinpath(@__DIR__,"..","optimization",example))
            config["run"]["output"] = joinpath(directory,String(method))
            config["ctm"]["measure"] = false
            config["optimization"]["check_gradient"] = false # covered by test_optimization.jl
            path = joinpath(directory,"$(method).toml")
            open(io -> TOML.print(io,config),path,"w")
            result = run_optimization(method,path)
            @test isfile(result.filename)
            restored,parameters,metadata = load_checkpoint(result.filename)
            @test all(isapprox.(restored.B,result.state.B))
            @test metadata["method"]==String(method)
            @test !isempty(metadata["history"])
            config["run"]["resume"] = result.filename
            config["run"]["output"] = joinpath(directory,"$(method)_resumed")
            config["optimization"]["maxiter"] = 0
            method==:full_update && (config["schedule"][1]["tau"] = 0.0)
            open(io -> TOML.print(io,config),path,"w")
            resumed = run_optimization(method,path)
            @test all(isapprox.(resumed.state.B,restored.B))
            @test isfile(resumed.filename)
            config["model"]["U"] = 3.0
            open(io -> TOML.print(io,config),path,"w")
            @test_throws ArgumentError run_optimization(method,path)
        end
    end
end
