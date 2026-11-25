using Test, TensorKit, LinearAlgebra, Random
if !isdefined(@__MODULE__,:KO)
    include(joinpath(@__DIR__,"..","..","..","..","src","fermionic","kagome","KagomeFermionicZ2.jl"))
    using .KagomeFermionicZ2
    include(joinpath(@__DIR__,"..","..","..","..","src","fermionic","kagome","ctmrg.jl"))
    using .KagomeFermionicCTM
    include(joinpath(@__DIR__,"..","..","..","..","src","fermionic","kagome","optimization","KagomeFermionicOptimization.jl"))
    using .KagomeFermionicOptimization
    const KO = KagomeFermionicOptimization
end
BLAS.set_num_threads(1)

include("test_reblocking.jl")

@testset "Simplex orientation and differentiable blocking" begin
    state = initial_iPESS(2,3;rng=MersenneTwister(7))
    rotated = reverse_triangles(state)
    restored = reverse_triangles(rotated)
    @test restored.T_u == state.T_u
    @test restored.T_d == state.T_d
    @test all(isapprox.(restored.B,state.B))
    @test restored.lambda_u == state.lambda_u
    @test restored.lambda_d == state.lambda_d
    for y in 1:3,x in 1:2
        wr = triangle_wavefunction(rotated,mod1(2-x,2),mod1(2-y,3),:up;weighted=false)
        wd = triangle_wavefunction(state,x,y,:dn;weighted=false)
        @test norm(wr-wd)<1e-12
        w = triangle_wavefunction(state,x,y,:up;weighted=false)
        a = KO._block_triangle(w,state.T_d[x,y],KO._fusion())
        @test a ≈ build_A_kagome(state,x,y)[2] atol=1e-12
    end
    v,layout = pack_state(state)
    @test all(isapprox.(unpack_state(v,layout).B,state.B))
end

@testset "CTMRG triangle metric and four-factor ALS" begin
    state = initial_iPESS(1,1;rng=MersenneTwister(2))
    parameters = Dict("t"=>1.0,"U"=>2.0,"μ"=>0.1)
    environment = triangle_environment(state,1,1;chi=2,ctm_iterations=2)
    @test environment.diagnostics.extraction_error<1e-8
    @test environment.metric ≈ environment.root'*environment.root atol=1e-10
    @test all(minimum(eigvals(Hermitian(b)))>=-1e-10 for (_,b) in blocks(environment.metric))
    gate,_ = trotter_gate(Hamiltonian_triangle(parameters),0.04)
    target = KO.KF._apply_gate(environment.W0,gate)
    fitted = fit_triangle(target,environment;Dmax=2,n_sweep=3)
    @test length(fitted.factors)==4
    @test all(KO.KF._rank(t)==3 for t in fitted.factors)
    @test all(c.final_loss <= c.initial_loss+1e-10 for c in fitted.candidates)
    @test all(diff(fitted.losses).<=1e-10)
    @test 0<=fitted.fidelity<=1+1e-10
    @test maximum(dim(space(fitted.factors[1],k)) for k in 1:3)<=2
    exact = fit_triangle(target,environment;Dmax=8,n_sweep=1)
    @test exact.fidelity ≈ 1 atol=1e-9
    @test exact.losses[end]<1e-16
    for orientation in (:up,:dn)
        trial = deepcopy(state)
        old = triangle_wavefunction(trial,1,1,orientation;weighted=false)
        identity_gate = unitary(codomain(gate),domain(gate))
        result = full_update_triangle!(trial,1,1,orientation,identity_gate;Dmax=2,chi=2,ctm_iterations=2,n_sweep=1)
        new = triangle_wavefunction(trial,1,1,orientation;weighted=false)
        @test abs(dot(old,new))/(norm(old)*norm(new)) ≈ 1 atol=1e-8
        @test check_state(trial;Dmax=2) === trial
        @test result.fit.fidelity ≈ 1 atol=1e-8
    end
    history = full_update!(state,parameters,0.01,0.01,2;chi=2,ctm_iterations=2,n_sweep=1)
    @test length(history)==3
    @test [h.orientation for h in history]==[:dn,:up,:dn]
    @test all(h -> isfinite(h.loss) && h.fidelity>0,history)
    @test check_state(state;Dmax=2) === state
end

@testset "AD through CTMRG and variational descent" begin
    state = initial_iPESS(1,1;rng=MersenneTwister(2))
    parameters = Dict("t"=>1.0,"U"=>2.0,"μ"=>0.1)
    v,_ = pack_state(state)
    problem = VariationalProblem(state,parameters;chi=2,ctm_iterations=1)
    result = check_gradient(v,problem;epsilon=1e-5)
    @test result.relative_error<2e-3
    result2 = check_gradient(v,problem;epsilon=3e-6,rng=MersenneTwister(91))
    @test result2.relative_error<2e-3
    E,g = value_and_gradient(v,problem)
    @test isfinite(E) && all(isfinite,g)
    @test norm(g)>1e-8
    fixed_environment = KO._fixed_environment(KO._ipeps(v,problem),problem.chi,problem.settings)
    function frozen_energy(x)
        A = KO._ipeps(x,problem)
        AA = map(column -> map(a -> KO.CTM.build_double_layer_swap(a',a)[1],column),A)
        KO._energy(A,AA,fixed_environment.CTM,problem.settings,problem.operators,problem.hoppings)
    end
    frozen_gradient = KO.Zygote.gradient(frozen_energy,v)[1]
    @test norm(g-frozen_gradient)>1e-7 # environment response is actually included
    deeper = VariationalProblem(state,parameters;chi=4,ctm_iterations=2)
    deeper_check = check_gradient(v,deeper;epsilon=1e-5)
    @test deeper_check.relative_error<2e-3
    optimized = optimize_variational(state,parameters;chi=2,ctm_iterations=1,maxiter=2,step_size=0.2)
    @test length(optimized.history)>=2
    @test all(diff([r.energy for r in optimized.history]).<=1e-10)
    @test optimized.energy<E
    @test check_state(optimized.state;Dmax=2) === optimized.state
    @test all(space(a)==space(b) for (a,b) in zip(state.B,optimized.state.B))
    @test all(space(a)==space(b) for (a,b) in zip(state.T_u,optimized.state.T_u))
end

@testset "Nonuniform two-cell CTMRG gradient" begin
    state = initial_iPESS(2,1;rng=MersenneTwister(17))
    parameters = Dict("t_up"=>(cis(0.2),0.7,1.1),"t_dn"=>(0.9,cis(-0.3),0.8),"U"=>2.0,"μ"=>0.1)
    v,_ = pack_state(state)
    problem = VariationalProblem(state,parameters;chi=2,ctm_iterations=1)
    report = check_gradient(v,problem;epsilon=1e-5)
    @test report.relative_error<2e-3
    measurement = measure_ctmrg(state,parameters;chi=2,maxiter=1,tol=1e-30,verbose=false)
    @test report.energy ≈ measurement.observables.energy_per_site atol=1e-10
    environment = triangle_environment(state,2,1;chi=2,ctm_iterations=2)
    @test environment.diagnostics.extraction_error<1e-8
end
