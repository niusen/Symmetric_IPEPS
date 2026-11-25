using Test, LinearAlgebra, Random, TensorKit
include("../../../../src/fermionic/kagome/spin/bootstrap.jl")
using .KagomeSpinOptimization, .KagomeFermionicZ2, .KagomeGFPEPSValidation
using .KagomeFermionicOptimization
BLAS.set_num_threads(1)

@testset "Spin triangle gate and longitudinal field" begin
    model = HeisenbergField(1.0,0.6)
    H = Hamiltonian_triangle(model)
    matrix = reshape(convert(Array,H),8,8)
    # |↑↑↑> and |↓↓↓> carry three aligned bonds and half of the field.
    @test matrix[1,1] ≈ 3/4-3model.h/4
    @test matrix[end,end] ≈ 3/4+3model.h/4
    @test matrix[2,3] ≈ 1/2
    full,half = trotter_gate(H,0.07)
    @test full ≈ half*half
    @test convert(Array,Hamiltonian_triangle(model;orientation=:dn)) ≈ convert(Array,H)
end

@testset "Projected spin energy and CTM gradient" begin
    seed = gutzwiller_project(initial_iPESS(1,1;rng=MersenneTwister(307)))
    model = HeisenbergField(1.0,0.53)
    p = VariationalProblem(seed,model;chi=2,ctm_iterations=1)
    v,_ = pack_state(seed)
    r = measure_heisenberg(convert_iPESS_to_iPEPS(seed),p.U_phy;
        J=model.J,h=model.h,chi=2,maxiter=1,projector_tol=1e-8,verbose=false)
    @test r.energy_per_site ≈ r.exchange_per_site-model.h*r.magnetization_per_site[3]
    @test variational_energy(v,p) ≈ r.energy_per_site atol=1e-9
    gradient = check_gradient(v,p;epsilon=1e-5)
    @show gradient
    @test gradient.relative_error < 2e-3
end
