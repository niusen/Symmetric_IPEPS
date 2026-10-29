using Test, LinearAlgebra, Random, TensorKit, SUNRepresentations
BLAS.set_num_threads(1);
if !isdefined(@__MODULE__,:HoneycombSU4)
    include("../../../src/bosonic/honeycomb/Honeycomb_SU4.jl")
end
using .HoneycombSU4
const HC=HoneycombSU4;

function diagonal_spectrum(reps,values)
    V=Rep[SU₄](reps...);
    S=DiagonalTensorMap(unitary(V,V));
    for (c,b) in blocks(S)
        diag(b).=values[Int(dim(c))];
        # diag returns a copy for some versions; assign the diagonal explicitly.
        for k=1:size(b,1)
            b[k,k]=values[Int(dim(c))][k];
        end
    end
    return S
end

@testset "Optimal whole-multiplet dimension budget" begin
    S=diagonal_spectrum(["1"=>1,"4"=>1,"6"=>1],Dict(1=>[0.8],4=>[1.0],6=>[0.9]));
    _,s,_=HC.tsvd(S;trunc=WholeMultipletBudget(5));
    @test dim(space(s,1))==5;
    @test norm(s)^2≈4+0.8^2;

    Sbig=diagonal_spectrum(["1"=>1,"4"=>1,"10"=>1],Dict(1=>[0.8],4=>[1.0],10=>[1.2]));
    _,s,_=HC.tsvd(Sbig;trunc=WholeMultipletBudget(9));
    @test dim(space(s,1))==5;
    @test norm(s)^2≈4+0.8^2;

    # Skipping an oversized irrep is not enough: undoing earlier selections can
    # retain more physical norm. Here the whole 10 wins over the leading 1+6.
    Sweight=diagonal_spectrum(["1"=>1,"6"=>1,"10"=>1],Dict(1=>[1.0],6=>[0.95],10=>[0.94]));
    _,s,_=HC.tsvd(Sweight;trunc=WholeMultipletBudget(10));
    @test dim(space(s,1))==10;
    @test norm(s)^2≈10*0.94^2;

    # Exhaustive independent enumeration of every per-irrep multiplicity.
    rng=MersenneTwister(901);
    values=Dict(d=>sort(rand(rng,n);rev=true) for (d,n) in ((1,3),(4,2),(6,2),(10,1)));
    Srandom=diagonal_spectrum(["1"=>3,"4"=>2,"6"=>2,"10"=>1],values);
    for budget=1:24
        expected=maximum(sum(d*sum(abs2,values[d][1:k]) for (d,k) in zip((1,4,6,10),ks))
            for ks in Iterators.product(0:3,0:2,0:2,0:1)
            if sum(d*k for (d,k) in zip((1,4,6,10),ks))<=budget);
        u,s,v=HC.tsvd(Srandom;trunc=WholeMultipletBudget(budget));
        @test dim(space(s,1))<=budget;
        @test norm(s)^2≈expected atol=1e-12;
        @test norm(Srandom-u*s*v)^2≈norm(Srandom)^2-expected atol=1e-11;
    end

    dense=DiagonalTensorMap([0.9,0.8,0.4,0.1],ComplexSpace(4));
    for budget=1:4
        _,new,_=HC.tsvd(dense;trunc=WholeMultipletBudget(budget));
        _,old,_=HC.tsvd(dense;trunc=HC.truncdim(budget));
        @test norm(new)^2≈norm(old)^2;
        @test space(new,1)==space(old,1);
    end

    @test_throws ArgumentError WholeMultipletBudget(0);
    only4=diagonal_spectrum(["4"=>1],Dict(4=>[1.0]));
    @test_throws ArgumentError HC.tsvd(only4;trunc=WholeMultipletBudget(3));
    zero=0*only4;
    @test_throws ArgumentError HC.tsvd(zero;trunc=WholeMultipletBudget(4));
end
