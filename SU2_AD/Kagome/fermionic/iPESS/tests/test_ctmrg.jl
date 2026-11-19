# Can also run directly: julia --startup-file=no test_ctmrg.jl
if !isdefined(@__MODULE__, :KF)
    using Test, TensorKit, LinearAlgebra, Random
    include(joinpath(@__DIR__,"..","..","..","..","src","fermionic","kagome","KagomeFermionicZ2.jl"))
    using .KagomeFermionicZ2
    const KF = KagomeFermionicZ2
    BLAS.set_num_threads(1)
    include("fixtures.jl")
end
include(joinpath(@__DIR__,"..","..","..","..","src","fermionic","kagome","ctmrg.jl"))
using .KagomeFermionicCTM

# Exact product of disconnected spinful Hubbard dimers. For :dn, the dimers
# cross the blocked PEPS cells, so this tests the x/y/diagonal fermionic strings
# against an independent many-body eigenstate with NONZERO hopping energy.
function dimer_state(orientation,bond)
    hoppings = ntuple(k -> k == bond ? cis(0.37) : 0.0im,3)
    zero_hop = (0.0,0.0,0.0)
    params = Dict{String,Any}("t_up"=>orientation == :up ? hoppings : zero_hop,
                              "t_dn"=>orientation == :dn ? hoppings : zero_hop,
                              "U"=>0.0,"μ"=>0.0)
    spectator = (3,1,2)[bond]
    occupations = (0,2,1,1)
    states = [(a,b,c) for c in 1:4 for b in 1:4 for a in 1:4]
    inds = findall(s -> s[spectator] == 1 && sum(occupations[i] for i in s) == 2,states)
    M = exact_triangle(params,orientation)
    E = eigen(Hermitian(M[inds,inds]))
    ψ = zeros(ComplexF64,64)
    ψ[inds] = E.vectors[:,1]
    state = atomic_state(1)
    template = triangle_wavefunction(state,1,1,orientation)
    W = TensorMap(reshape(ψ,1,4,1,4,1,4),codomain(template),domain(template))
    core,Us,Ss,_ = KF._hosvd(W,4,1e-12)
    (orientation == :up ? state.T_u : state.T_d)[1,1] = core
    for b in 1:3
        order = orientation == :up ? (1,3,2) : (3,1,2)
        state.B[1,1,b] = KF.fpermute(Us[b],order)
        weights = orientation == :up ? state.lambda_u : state.lambda_d
        weights[1,1,b] = KF._normalized(TensorKit.permute(Ss[b],((2,),(1,))))
    end
    return check_state(state),params,E.values[1]/3
end

@testset "Legacy fermionic CTMRG atomic limits" begin
    params = Dict("t"=>1.0,"U"=>4.0,"μ"=>0.3)
    for (occ,density) in ((1,0.0),(2,2.0))
        s = atomic_state(occ)
        result = measure_ctmrg(s,params;chi=2,maxiter=5,verbose=false)
        @test result.observables.energy_per_site ≈ 1.0-0.3density atol=1e-10
        @test result.observables.density ≈ density atol=1e-10
        @test maximum(abs,result.observables.down_hopping) < 1e-10
    end
end

@testset "Exact decoupled dimers: intra-cell and three inter-cell bonds" begin
    for (orientation,bond) in ((:up,1),(:dn,1),(:dn,2),(:dn,3))
        state,params,energy = dimer_state(orientation,bond)
        @test su_observables(state,params).energy_per_site ≈ energy atol=1e-11
        result = measure_ctmrg(state,params;chi=16,maxiter=12,verbose=false)
        @test result.observables.energy_per_site ≈ energy atol=1e-8
        @test result.observables.density ≈ 2/3 atol=1e-8
    end
end

include("test_cover.jl")

@testset "CTMRG blocked operators preserve internal JW strings" begin
    s = initial_iPESS(1,1;rng=MersenneTwister(17))
    _,_,U = build_A_kagome(s,1,1)
    op = KagomeFermionicCTM._blocked_operators(U,Dict("U"=>2.0))
    fusion = reshape(convert(Array,U),64,64)
    dense = KF._triangle_operators()
    for site in 1:3, spin in 1:2
        c = convert(Array,op.Cs[site])[spin,:,:]
        @test fusion'*c*fusion ≈ dense.c[site][spin] atol=1e-12
    end
    result = measure_ctmrg(s,Dict("U"=>2.0);chi=4,maxiter=3,verbose=false)
    @test isfinite(result.observables.energy_per_site)
    @test all(isfinite,result.observables.n)
    @test result.iterations <= 3
end
