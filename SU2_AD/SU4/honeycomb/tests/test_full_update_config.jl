using Test,TOML
include("../../../src/bosonic/honeycomb/star/FullRunConfig.jl")
include("../server/run_full_update.jl")

@testset "Full update configuration and CLI" begin
    template=joinpath(@__DIR__,"../config/FullUpdate.toml");
    c=read_full_config(template);
    @test (c.Lx,c.Ly,c.Dmax)==(4,4,9)
    @test c.space_policy==:fixed_space
    @test c.initial==:star
    @test c.chi==16
    @test sum(s.steps for s in c.stages)==10
    @test c.output_dir==normpath(joinpath(@__DIR__,"../server_runs/full_fixed_D9"))
    @test full_main(["--config",template,"--dry-run"])==:preview
    @test full_main(["--help"])==:help
    mktempdir() do dir
        raw=TOML.parsefile(template);path=joinpath(dir,"plan.toml");
        save()=open(io->TOML.print(io,raw),path,"w");
        raw["run"]["space_policy"]="tensorkit";raw["run"]["Dmax"]=24;save();
        @test read_full_config(path).space_policy==:tensorkit
        @test read_full_config(path).Dmax==24
        @test read_full_config(path;output_dir="scratch").output_dir==joinpath(dir,"scratch")
        for bad in ("lambda","optimal","")
            raw["run"]["space_policy"]=bad;save();
            @test_throws ErrorException read_full_config(path)
        end
        raw["run"]["space_policy"]="fixed_space";
        raw["run"]["Dmax"]=4;save();@test_throws ErrorException read_full_config(path)
        raw["run"]["initial"]="dense";save();@test read_full_config(path).Dmax==4
        raw["ctm"]["tol"]=0;save();@test_throws ErrorException read_full_config(path)
    end
end
