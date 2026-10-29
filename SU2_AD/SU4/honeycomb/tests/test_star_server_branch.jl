using Test,TOML,JLD2,LinearAlgebra
include("../../../src/bosonic/honeycomb/star/StarSimpleUpdateRun.jl")
using .HoneycombStarRun
const HC=HoneycombStarRun.HoneycombSU4

@testset "Start a new budget/schedule from a checkpoint" begin
    mktempdir() do dir
        # Synthetic progress metadata tests preservation of the parent clock;
        # no expensive imaginary-time evolution is performed in this test.
        T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=2,Ly=2);
        source=joinpath(dir,"source.jld2");
        progress=Dict{String,Any}("total_steps"=>7,"tau"=>.007);
        jldsave(source;schema_version=1,physics=(Lx=2,Ly=2,J=1.),T_set=T,
            lambdax_set=lx,lambday_set=ly,progress);
        raw=Dict("run"=>Dict("Lx"=>2,"Ly"=>2,"J"=>1.,"Dmax"=>24,
            "output_dir"=>"branch","identity_sweeps"=>0,"local_energy_every"=>0),
            "schedule"=>[Dict("steps"=>3,"dt"=>.002)],"ctm"=>Dict("enabled"=>false));
        path=joinpath(dir,"branch.toml");open(io->TOML.print(io,raw),path,"w");
        c=read_star_config(path);
        @test run_star_server(c;initial_checkpoint=source,stop_after_chunks=0)==:checkpointed
        s=load(joinpath(c.output_dir,"state_latest.jld2"));
        @test s["progress"]["tau"]==.007
        @test s["progress"]["total_steps"]==7
        @test s["progress"]["stage"]==1
        @test s["progress"]["stage_step"]==0
        @test s["physics"].stages[1].Dmax==24
        @test all(norm(T[i]-s["T_set"][i])<1e-14 for i in eachindex(T))
        @test run_star_server(c;resume=true,stop_after_chunks=0)==:checkpointed
        @test_throws ErrorException run_star_server(c;resume=true,initial_checkpoint=source)
        # A stale lock must not silently permit two jobs to write one checkpoint.
        lock=joinpath(c.output_dir,".star_su_lock");mkdir(lock);
        @test_throws ErrorException run_star_server(c;resume=true,stop_after_chunks=0)
        rm(lock);
    end
end
