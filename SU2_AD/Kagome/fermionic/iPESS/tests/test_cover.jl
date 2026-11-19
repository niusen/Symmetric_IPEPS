@testset "Nonuniform 2x2 exact dimer covering" begin
    state = initial_iPESS(2,2,Rep[ℤ₂](0=>1);rng=MersenneTwister(21))
    empty = atomic_state(1)
    for x in 1:2,y in 1:2,b in 1:3
        state.B[x,y,b] = deepcopy(empty.B[1,1,b])
    end
    used = Set{NTuple{3,Int}}()
    for (x,y,orientation,bond) in ((1,1,:up,1),(1,2,:dn,1),(2,1,:dn,2),(2,2,:dn,3))
        prototype,_,_ = dimer_state(orientation,bond)
        tensors = orientation == :up ? state.T_u : state.T_d
        tensors[x,y] = deepcopy((orientation == :up ? prototype.T_u : prototype.T_d)[1,1])
        sites = triangle_sites(state,x,y,orientation)
        for b in ((1,2),(2,3),(3,1))[bond]
            site = sites[b]
            @test !(site in used)
            push!(used,site)
            state.B[site...] = deepcopy(prototype.B[1,1,b])
            weights = orientation == :up ? state.lambda_u : state.lambda_d
            weights[site...] = deepcopy((orientation == :up ? prototype.lambda_u : prototype.lambda_d)[1,1,b])
        end
    end
    check_state(state)
    params = Dict("t"=>cis(0.37),"U"=>2.0,"μ"=>0.2)
    result = measure_ctmrg(state,params;chi=16,maxiter=20,verbose=false)
    @test result.observables.energy_per_site ≈ -2/3 + 2/12 - 0.2*2/3 atol=1e-8
    @test result.observables.density ≈ 2/3 atol=1e-8
    @test result.observables.double_occupancy ≈ 1/6 atol=1e-8
end
