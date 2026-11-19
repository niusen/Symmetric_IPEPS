using Test,TOML,LinearAlgebra,TensorKit,JLD2
if !isdefined(@__MODULE__,:HoneycombStarRun)
    include("../../../src/bosonic/honeycomb/star/StarSimpleUpdateRun.jl")
end
using .HoneycombStarRun
if !isdefined(@__MODULE__,:HC)
    const HC=HoneycombStarRun.HoneycombSU4
end
BLAS.set_num_threads(1);

@testset "Numerical tracing preserves the original update" begin
    mktempdir() do dir
        result=trace_star_simple_update(folder=joinpath(dir,"trace"),L=2,sweeps=1);
        T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=2,Ly=2);
        HC.itebd_honeycomb(Dict("J"=>1.),T,lx,ly,.01,.01,24;
            convention=:permutation,verbose=false,method=:two_site);
        @test result.status==:finished
        @test result.updates==10
        @test all(norm(T[i]-result.T_set[i])<1e-12 for i in eachindex(T))
        @test all(norm(lx[i]-result.lambdax_set[i])<1e-12 for i in eachindex(lx))
        @test all(norm(ly[i]-result.lambday_set[i])<1e-12 for i in eachindex(ly))
        @test !isfile(joinpath(dir,"trace","first_lost_support.jld2"))
        @test !isfile(joinpath(dir,"trace","first_zero_or_nonfinite.jld2"))
        for A in (T[1],result.T_set[1]),leg=1:5
            fast=HoneycombStarRun._trace_fusion_support(A)[leg];
            reference=HoneycombStarRun._trace_leg_support(A,leg);
            @test all(abs(get(fast,c,0.)-get(reference,c,0.))<1e-12
                for c in union(keys(fast),keys(reference)))
        end
        continued=trace_star_simple_update(folder=joinpath(dir,"continued"),L=2,sweeps=1,
            initial_checkpoint=joinpath(dir,"trace","state_step1.jld2"));
        HC.itebd_honeycomb(Dict("J"=>1.),T,lx,ly,.01,.01,24;
            convention=:permutation,verbose=false,step_offset=1,method=:two_site);
        @test continued.status==:finished && continued.updates==20
        @test all(norm(T[i]-continued.T_set[i])<1e-12 for i in eachindex(T))
        @test all(norm(lx[i]-continued.lambdax_set[i])<1e-12 for i in eachindex(lx))
        @test all(norm(ly[i]-continued.lambday_set[i])<1e-12 for i in eachindex(ly))
    end
end

@testset "Exact first-gate D9 -> D24 growth" begin
    T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=4,Ly=4);
    bond=first(HC.honeycomb_bonds(4,4));
    before=HC.honeycomb_bond_wavefunction(T,lx,ly,bond);
    p=space(T[1],5);I2=unitary(p*p,p*p);
    P=TensorKit.permute(I2,((2,1),(3,4)));dt=.001;
    gate=cosh(dt)*I2-sinh(dt)*P;
    @tensor expected[:]:=before[-1,-2,-3,-4,-5,-6,a,b]*gate[-7,-8,a,b];
    HC.tebd_xbond(1,T,lx,ly,gate,1.5,1,24);
    after=HC.honeycomb_bond_wavefunction(T,lx,ly,bond);
    scale=dot(after,expected)/dot(after,after);
    @test norm(scale*after-expected)/norm(expected)<1e-10
    @test dim(space(lx[2,1],1))==24
    @test sort([Int(dim(c)) for (c,b) in blocks(lx[2,1]) if norm(b)>1e-10])==[1,4,4,15]
end

@testset "Real SU checkpoint/resume and global Trotter order" begin
    mktempdir() do dir
        raw=Dict("run"=>Dict("Lx"=>2,"Ly"=>2,"J"=>0.,"Dmax"=>9,
            "output_dir"=>"results","checkpoint_every"=>1,"keep_every"=>0,
            "local_energy_every"=>0,"identity_sweeps"=>0,"blas_threads"=>1,"method"=>"two_site"),
            "schedule"=>[Dict("steps"=>2,"dt"=>.001)],"ctm"=>Dict("enabled"=>false));
        config=joinpath(dir,"config.toml");open(io->TOML.print(io,raw),config,"w");
        c=read_star_config(config);
        T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=2,Ly=2);
        # Independent two-sweep call: a reset of the order at each checkpoint
        # would not match this sequence of direction permutations.
        HC.itebd_honeycomb(Dict("J"=>0.),T,lx,ly,.002,.001,9;convention=:permutation,verbose=false,method=:two_site);
        @test run_star_server(c;stop_after_chunks=1)==:checkpointed
        latest=joinpath(c.output_dir,"state_latest.jld2");s=load(latest);
        @test s["progress"]["total_steps"]==1
        @test !s["progress"]["finished"]
        @test !isdir(joinpath(c.output_dir,".star_su_lock"))
        write(joinpath(c.output_dir,"orphan.partial"),"unfinished write");
        @test run_star_server(c;resume=true)==:finished
        s=load(latest);
        @test s["progress"]["finished"]
        @test s["progress"]["tau"]≈.002
        @test length(s["history"])==3
        @test all(norm(T[i]-s["T_set"][i])<1e-10 for i in eachindex(T))
        @test all(norm(lx[i]-s["lambdax_set"][i])<1e-10 for i in eachindex(lx))
        @test all(norm(ly[i]-s["lambday_set"][i])<1e-10 for i in eachindex(ly))
        @test_throws ErrorException run_star_server(c)
        raw["schedule"][1]["dt"]=.002;open(io->TOML.print(io,raw),config,"w");
        @test_throws ErrorException run_star_server(read_star_config(config);resume=true)
        @test load(latest,"progress")["tau"]≈.002
    end
end
