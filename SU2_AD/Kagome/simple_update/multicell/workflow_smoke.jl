# Small end-to-end launcher/resume checks; no production evolution or CTMRG.
using Test
include(joinpath(@__DIR__,"run.jl"))
include(joinpath(@__DIR__,"compatibility_probe.jl"))

length(ARGS) == 1 || error("provide a NEW validation output directory")
validation_out = abspath(ARGS[1])
ispath(validation_out) && error("validation output directory already exists")
mkpath(validation_out)

@testset "Both launcher geometries, checkpoints and stage resume" begin
    @test dim(virtual_space("0:1,1/2:1")) == 3
    @test dim(virtual_space("0:1,1/2:1,1:1")) == 6
    @test_throws ErrorException virtual_space("0:1,0:2")
    @test_throws ErrorException virtual_space("1/3:1")
    for cell in ("sqrt3xsqrt3","3x3")
        out = joinpath(validation_out,cell)
        args = ["--cell",cell,"--Dmax","3","--seed","431","--smoke","--out",out]
        main(args)
        @test isfile(joinpath(out,"stage_01.jld2"))
        @test isfile(joinpath(out,"stage_02.jld2"))
        @test isfile(joinpath(out,"su_diagnostics.txt"))
        final,data = load_checkpoint(joinpath(out,"final.jld2"))
        @test data["metadata"]["completed_stages"] == 2
        @test data["metadata"]["seed"] == 431
        continued_out = joinpath(validation_out,cell*"_resumed")
        main(["--cell",cell,"--Dmax","3","--smoke","--out",continued_out,
              "--resume",joinpath(out,"stage_01.jld2")])
        resumed,resumed_data = load_checkpoint(joinpath(continued_out,"final.jld2"))
        @test resumed_data["metadata"]["seed"] == 431
        for r in eachindex(final.cells), name in fieldnames(KagomeSUTensors)
            a,b = getfield(final.cells[r],name),getfield(resumed.cells[r],name)
            @test norm(a-b) <= 3e-12*max(norm(a),norm(b))
        end
        @test_throws ErrorException main(args) # prevent accidental overwrite
    end
end
compatibility_report(joinpath(validation_out,"compatibility.toml"))
