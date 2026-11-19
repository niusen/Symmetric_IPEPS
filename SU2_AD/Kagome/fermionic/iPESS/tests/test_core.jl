@testset "Hubbard signs, onsite counting, and imaginary time" begin
    p = Dict("t_up"=>(1.0+0.3im,-0.7,0.8im),"t_dn"=>(0.2,-0.4im,1.1),"U"=>3.4,"μ"=>0.37)
    for centered in (true,false), orientation in (:up,:dn)
        params = merge(p,Dict("centered"=>centered))
        H = Hamiltonian_triangle(params;orientation)
        M = reshape(convert(Array,H),64,64)
        @test M ≈ exact_triangle(params,orientation) atol=1e-12
        @test M ≈ M'
        G,Ghalf = trotter_gate(H,0.03)
        @test Ghalf*Ghalf ≈ G atol=1e-12
        @test trotter_gate(H,0.0)[1] ≈ unitary(codomain(H),domain(H)) atol=1e-12
    end
    z = Hamiltonian_triangle(Dict("t"=>0.0))
    @test norm(z) == 0
    @test trotter_gate(z,0.1)[1] ≈ unitary(codomain(z),domain(z))
end

@testset "Explicit fermionic exchanges" begin
    V = Rep[ℤ₂](0=>1,1=>1)
    A = KF._allout(randn(MersenneTwister(3),ComplexF64,V*V,V'))
    exchanged = KF.fpermute(A,(2,1,3))
    expected = permutedims(convert(Array,A),(2,1,3))
    expected[2,2,:] .*= -1
    @test convert(Array,exchanged) ≈ expected
    @test KF.fpermute(exchanged,(2,1,3)) ≈ A
    @test KF.fpermute(KF.fpermute(A,(2,3,1)),(3,1,2)) ≈ A
end

@testset "Periodic geometry, HOSVD, checkpoint, evolution" begin
    s = initial_iPESS(2,3; rng=MersenneTwister(5))
    expected = Set((x,y,b) for x in 1:2 for y in 1:3 for b in 1:3)
    for orientation in (:up,:dn)
        layer = [site for y in 1:3 for x in 1:2 for site in triangle_sites(s,x,y,orientation)]
        @test length(layer) == 18
        @test Set(layer) == expected
    end
    @test triangle_sites(s,2,3,:dn) == ((1,3,1),(2,3,2),(2,1,3))
    H = Hamiltonian_triangle(Dict("U"=>2.0))
    identity_gate = unitary(codomain(H),domain(H))
    for orientation in (:up,:dn)
        localstate = deepcopy(s)
        W = KF._normalized(triangle_wavefunction(localstate,2,3,orientation))
        # Identity gate with no truncation must preserve the entire boundary
        # wavefunction, including all odd physical and virtual sectors.
        error = triangle_update!(localstate,2,3,orientation,identity_gate;Dmax=8,trun_tol=0.0)
        Wnew = KF._normalized(triangle_wavefunction(localstate,2,3,orientation))
        @test abs(dot(W,Wnew)) ≈ 1 atol=1e-11
        @test error < 1e-20
        @test check_state(localstate;Dmax=8) === localstate
        # Compare the NONZERO-time untruncated update to direct gate action.
        gate,_ = trotter_gate(H,0.07)
        target = KF._normalized(KF._apply_gate(W,gate))
        trial = deepcopy(s)
        triangle_update!(trial,2,3,orientation,gate;Dmax=8,trun_tol=0.0)
        actual = KF._normalized(triangle_wavefunction(trial,2,3,orientation))
        @test abs(dot(target,actual)) ≈ 1 atol=1e-11
    end
    params = Dict("t"=>1.0,"U"=>2.0,"μ"=>0.2)
    small = initial_iPESS(1,1;rng=MersenneTwister(8))
    history = itebd!(small,params,0.02,0.01,2)
    @test length(history) == 2
    @test history[end].tau == 0.02
    @test check_state(small;Dmax=2) === small
    @test all(r -> isfinite(r.discarded_weight) && r.discarded_weight>=0,history)
    @test isfinite(su_observables(small,params).energy_per_site)
    @test norm(convert_iPESS_to_iPEPS(small)[1][1]) ≈ 1
    mktempdir() do dir
        path = joinpath(dir,"state.jld2")
        save_checkpoint(path,small;parameters=params,metadata=Dict("steps"=>2))
        restored,p,m = load_checkpoint(path)
        @test p == params
        @test m["steps"] == 2
        @test restored.B == small.B
        @test restored.T_u == small.T_u
        @test restored.T_d == small.T_d
        @test restored.lambda_u == small.lambda_u
        @test restored.lambda_d == small.lambda_d
        itebd!(restored,p,0.01,0.01,2)
        itebd!(small,p,0.01,0.01,2)
        @test norm(convert_iPESS_to_iPEPS(restored)[1][1]-convert_iPESS_to_iPEPS(small)[1][1]) < 1e-11
    end
    @test_throws ArgumentError itebd!(small,params,0.025,0.01,2)
    @test_throws ArgumentError itebd!(small,params,0.1,0.0,2)
    @test_throws ArgumentError initial_iPESS(0,1)
    @test_throws ArgumentError triangle_sites(small,1,1,:sideways)
end

@testset "Atomic limit and normalization per physical site" begin
    params = Dict("t"=>1.0,"U"=>4.0,"μ"=>0.3)
    for (occ,density,doublon) in ((1,0.0,0.0),(2,2.0,1.0))
        s = atomic_state(occ)
        ob = su_observables(s,params)
        @test ob.energy_per_site ≈ 1.0-0.3density
        @test ob.density ≈ density atol=1e-12
        @test ob.double_occupancy ≈ doublon atol=1e-12
    end
end
