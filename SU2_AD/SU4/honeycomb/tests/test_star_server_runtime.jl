using Test,TOML,LinearAlgebra,TensorKit,JLD2
include("../../../src/bosonic/honeycomb/star/StarSimpleUpdateRun.jl")
using .HoneycombStarRun
const HC=HoneycombStarRun.HoneycombSU4
BLAS.set_num_threads(1);

@testset "Exact first-gate D9 -> D24 growth" begin
    T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=4,Ly=4);
    bond=first(HC.honeycomb_bonds(4,4));
    before=HC.honeycomb_bond_wavefunction(T,lx,ly,bond);
    p=space(T[1],5);I2=unitary(p*p,p*p);
    P=TensorKit.permute(I2,((2,1),(3,4)));dt=.001;
    gate=cosh(dt)*I2-sinh(dt)*P;
    @tensor expected[:]:=before[-1,-2,-3,-4,-5,-6,a,b]*gate[-7,-8,a,b];
    HC.tebd_xbond(1,T,lx,ly,gate,1.5,1,HC.WholeMultipletBudget(24));
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
            "local_energy_every"=>0,"identity_sweeps"=>0,"blas_threads"=>1),
            "schedule"=>[Dict("steps"=>2,"dt"=>.001)],"ctm"=>Dict("enabled"=>false));
        config=joinpath(dir,"config.toml");open(io->TOML.print(io,raw),config,"w");
        c=read_star_config(config);
        T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=2,Ly=2);
        # Independent two-sweep call: a reset of the order at each checkpoint
        # would not match this sequence of direction permutations.
        HC.itebd_honeycomb(Dict("J"=>0.),T,lx,ly,.002,.001,9;convention=:permutation,verbose=false);
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
