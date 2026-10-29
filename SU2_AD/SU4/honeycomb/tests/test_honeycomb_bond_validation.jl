using Test,LinearAlgebra,TensorKit,SUNRepresentations
if !isdefined(@__MODULE__,:HoneycombSU4)
    include("../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
end
if !isdefined(@__MODULE__,:HC)
    const HC=HoneycombSU4
end

@testset "Blockwise SU bond inverse" begin
    # Independent expected values for regularized inverse, including exact zeros.
    for scale in (1.,1e-200,1e200)
        S=DiagonalTensorMap(scale.*[1.,.5,0.],ComplexSpace(3));
        R=HC.my_pinv(S);
        expected=[1/(1+1e-12),.5/(.25+1e-12),0.]./scale;
        @test all(isfinite,R.data)
        @test R.data≈expected rtol=1e-12
        @test S.data==scale.*[1.,.5,0.]
    end
    V=Rep[SU₄]("1"=>1,"4"=>1,"4⁺"=>1);
    S=DiagonalTensorMap(unitary(V,V));R=HC.my_pinv(S);
    @test norm(R-S/(1+1e-12))<1e-12
    empty=DiagonalTensorMap(Float64[],ComplexSpace(0));
    @test_throws ArgumentError HC.my_pinv(empty)
    @test_throws ArgumentError HC.my_pinv(0*S)
end

@testset "Reject empty lambda on an unused brick-wall leg" begin
    T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=2,Ly=2);
    @test isnothing(HC._check_honeycomb_state(T,lx,ly;maxdim=9))
    # (1,2) has a trivial DOWN leg, omitted by the old real-bond-only check.
    ly[1,2]=DiagonalTensorMap(Float64[],zero(space(ly[1,2],1)));
    @test_throws ArgumentError HC._check_honeycomb_state(T,lx,ly;maxdim=9)
    @test_throws ArgumentError HC.itebd_honeycomb(Dict("J"=>1.),T,lx,ly,.001,.001,24;verbose=false)
end

@testset "All five Trotter layers on the server's 4x4 D24 state" begin
    T,lx,ly,_=HC.initial_honeycomb_star_iPEPS(Lx=4,Ly=4);
    HC.itebd_honeycomb(Dict("J"=>1.),T,lx,ly,.01,.01,24;convention=:permutation,verbose=false);
    @test isnothing(HC._check_honeycomb_state(T,lx,ly;maxdim=24))
    @test all(isfinite(norm(t)) && norm(t)>0 for t in T)
end

@testset "Update errors preserve bond and backend context" begin
    bond=first(HC.honeycomb_bonds(4,4));
    error=HC.HoneycombUpdateError(7,bond,.01,24,[],CapturedException(ArgumentError("empty lambda"),backtrace()));
    text=sprint(showerror,error);
    @test occursin("sweep=7",text) && occursin("Dmax=24",text)
    @test occursin("TensorKit=",text) && occursin("empty lambda",text)
end
