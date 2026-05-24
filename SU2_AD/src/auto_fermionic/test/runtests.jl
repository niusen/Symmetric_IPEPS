using LinearAlgebra
using Test
using AutoFermionicPESS
using TensorKit

@testset "one-site spin trace" begin
    psi = kron([1.0, 0.0], [1.0, 0.0])
    rho = psi * psi'
    @test isapprox(trace_spin(rho; keep=:up), [1.0 0.0; 0.0 0.0])

    bell = zeros(ComplexF64, 4)
    bell[1] = 1 / sqrt(2)
    bell[4] = 1 / sqrt(2)
    rho_bell = bell * bell'
    @test isapprox(entropy_vn(trace_spin(rho_bell; keep=:up)), log(2))
end

@testset "spin partition" begin
    nsites = 2
    psi_up = [1.0, 0.0, 0.0, 0.0]
    psi_dn = [0.0, 1.0, 0.0, 0.0]
    psi = kron(psi_up, psi_dn)
    rho_up = spin_partition_rdm(psi, nsites; keep=:up)
    @test isapprox(entropy_vn(rho_up), 0.0; atol=1e-12)
end

@testset "dense spinful operators" begin
    ops = dense_spinful_operators()
    z = zeros(ComplexF64, 4, 4)

    @test isapprox(ops.c_up * ops.c_up + ops.c_up * ops.c_up, z)
    @test isapprox(ops.c_dn * ops.c_dn + ops.c_dn * ops.c_dn, z)
    @test isapprox(ops.c_up * ops.c_dn + ops.c_dn * ops.c_up, z)
    @test isapprox(ops.c_up * ops.cdag_up + ops.cdag_up * ops.c_up, ops.id)
    @test isapprox(ops.c_dn * ops.cdag_dn + ops.cdag_dn * ops.c_dn, ops.id)
    @test isapprox(ops.c_up * ops.cdag_dn + ops.cdag_dn * ops.c_up, z)

    @test isapprox(ops.n_total, ops.n_up + ops.n_dn)
    @test isapprox(ops.n_double, ops.n_up * ops.n_dn)
    @test isapprox(ops.sx, ops.sx')
    @test isapprox(ops.sy, ops.sy')
    @test isapprox(ops.sz, ops.sz')

    M = reshape(ComplexF64.(1:16), 4, 4)
    @test sector_to_dense_basis(dense_to_sector_basis(M)) == M
    M2 = reshape(ComplexF64.(1:256), 16, 16)
    @test sector_multisite_to_dense_basis(dense_multisite_to_sector_basis(M2, 2), 2) == M2
end

@testset "spin Hall coefficients and triangle Hamiltonian" begin
    setting = SpinHallEnergySetting(Lx=2, Ly=2, magnetic_cell=2)
    pars = SpinHallParameters(t1=1.0, t2=1.0, mu=0.0, U=0.0, mx=0.0, B=0.0)
    coeffs = spin_hall_coefficients(setting, pars)

    @test isapprox(coeffs.tx_up[1, 1], 1.0im; atol=1e-14)
    @test isapprox(coeffs.tx_dn[1, 1], -1.0im; atol=1e-14)
    @test isapprox(coeffs.ty_up[1, 1], 1.0; atol=1e-14)
    @test isapprox(coeffs.ty_dn[1, 1], -1.0; atol=1e-14)
    @test isapprox(coeffs.t2_up[1, 1], 1.0; atol=1e-14)
    @test isapprox(coeffs.t2_dn[1, 1], -1.0; atol=1e-14)

    h = triangle_hamiltonian_dense(coeffs, 1, 1)
    @test size(h) == (64, 64)
    @test isapprox(h, h'; atol=1e-12)

    g = triangle_gate_dense(coeffs, 1, 1, 0.01)
    @test size(g) == (64, 64)
    @test isapprox(g, exp(-0.01 * h); atol=1e-12)

    hmap = triangle_hamiltonian_tensormap(coeffs, 1, 1)
    gmap = triangle_gate_tensormap(coeffs, 1, 1, 0.01)
    @test isapprox(convert(Array, hmap), reshape(dense_multisite_to_sector_basis(h, 3), 4, 4, 4, 4, 4, 4); atol=1e-12)
    @test isapprox(convert(Array, gmap), reshape(dense_multisite_to_sector_basis(g, 3), 4, 4, 4, 4, 4, 4); atol=1e-12)
end

@testset "TensorKit fermionic constructors" begin
    V = physical_spinful_space()
    @test dim(V) == 4

    ops = tensorkit_spinful_operators()
    @test dim(ops.V) == 4

    setting = SpinHallEnergySetting(Lx=2, Ly=2, magnetic_cell=2)
    pars = SpinHallParameters(t1=1.0, t2=1.0)
    coeffs = spin_hall_coefficients(setting, pars)
    gate = triangle_gate_tensormap(coeffs, 1, 1, 0.01)
    @test dim(domain(gate)) == 64
    @test dim(codomain(gate)) == 64

    pess = random_triangle_pess(1, 1)
    A = pess_to_ipeps_tensor(pess)
    @test dim(space(A, 5)) == 4

    AA, U_L, U_D, U_R, U_U = graded_build_double_layer_direct(A)
    @test numind(AA) == 4
    @test dim(space(AA, 1)) == 4
    @test dim(space(AA, 2)) == 4
    @test dim(space(AA, 3)) == 4
    @test dim(space(AA, 4)) == 4
    @test dim(domain(U_L)) == 4
end

@testset "minimal graded CTM skeleton" begin
    A_cell = Matrix{Any}(undef, 1, 1)
    A_cell[1, 1] = pess_to_ipeps_tensor(random_triangle_pess(1, 1))

    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)
    AA = setup.double_layer.AA[1, 1]
    CTM = setup.CTM[1, 1]

    @test numind(AA) == 4
    @test norm(AA) > 0
    @test norm(CTM.Cset.C1) > 0
    @test norm(CTM.Cset.C2) > 0
    @test norm(CTM.Cset.C3) > 0
    @test norm(CTM.Cset.C4) > 0
    @test norm(CTM.Tset.T1) > 0
    @test norm(CTM.Tset.T2) > 0
    @test norm(CTM.Tset.T3) > 0
    @test norm(CTM.Tset.T4) > 0
end

@testset "graded CTM left update" begin
    A_cell = Matrix{Any}(undef, 1, 1)
    A_cell[1, 1] = pess_to_ipeps_tensor(random_triangle_pess(1, 1))

    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)
    AA = setup.double_layer.AA[1, 1]
    CTM = setup.CTM[1, 1]

    result = graded_ctm_left_update(CTM, AA)
    CTM2 = result.CTM

    @test norm(result.projectors.P) > 0
    @test norm(result.projectors.Pinv) > 0
    @test norm(CTM2.Cset.C1) > 0
    @test norm(CTM2.Cset.C4) > 0
    @test norm(CTM2.Tset.T4) > 0
    @test numind(CTM2.Cset.C1) == 2
    @test numind(CTM2.Cset.C4) == 2
    @test numind(CTM2.Tset.T4) == 3
end

@testset "graded CTM directional updates" begin
    A_cell = Matrix{Any}(undef, 1, 1)
    A_cell[1, 1] = pess_to_ipeps_tensor(random_triangle_pess(1, 1))

    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)
    AA = setup.double_layer.AA[1, 1]
    CTM = setup.CTM[1, 1]

    for direction in 1:4
        result = graded_ctm_directional_update(CTM, AA; direction)
        @test norm(result.projectors.P) > 0
        @test norm(result.projectors.Pinv) > 0
        @test norm(result.CTM.Cset.C1) > 0
        @test norm(result.CTM.Cset.C2) > 0
        @test norm(result.CTM.Cset.C3) > 0
        @test norm(result.CTM.Cset.C4) > 0
        @test norm(result.CTM.Tset.T1) > 0
        @test norm(result.CTM.Tset.T2) > 0
        @test norm(result.CTM.Tset.T3) > 0
        @test norm(result.CTM.Tset.T4) > 0
    end
end

@testset "graded CTM cell updates" begin
    A_cell = Matrix{Any}(undef, 1, 1)
    A_cell[1, 1] = pess_to_ipeps_tensor(random_triangle_pess(1, 1))

    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)
    result = graded_ctm_cell_directional_update(setup.CTM, setup.double_layer.AA; direction=1)
    CTM2 = result.CTM[1, 1]

    @test size(result.CTM) == (1, 1)
    @test norm(result.projectors[1, 1].P) > 0
    @test norm(result.projectors[1, 1].Pinv) > 0
    @test norm(CTM2.Cset.C1) > 0
    @test norm(CTM2.Cset.C4) > 0
    @test norm(CTM2.Tset.T4) > 0

    sweep = graded_ctm_cell_sweep(setup.CTM, setup.double_layer.AA)
    @test size(sweep.CTM) == (1, 1)
    @test norm(sweep.CTM[1, 1].Cset.C1) > 0
    @test norm(sweep.CTM[1, 1].Cset.C2) > 0
    @test norm(sweep.CTM[1, 1].Cset.C3) > 0
    @test norm(sweep.CTM[1, 1].Cset.C4) > 0

    spec = graded_ctm_spectrum_signature(setup.CTM)
    @test !isempty(spec)
    @test all(isfinite, spec)
    @test maximum(spec) <= 1 + 1e-12

    iter = graded_ctm_cell_iterate(setup.CTM, setup.double_layer.AA; maxiter=1)
    @test size(iter.CTM) == (1, 1)
    @test length(iter.signatures) == 2
    @test length(iter.errors) == 1
    @test isfinite(iter.errors[1])

    iter_norm = graded_ctm_cell_iterate(setup.CTM, setup.double_layer.AA; maxiter=1, conv_check=:norm)
    @test length(iter_norm.signatures) == 2
    @test isfinite(iter_norm.errors[1])
end

@testset "graded 2x2 norm contraction" begin
    A_cell = Matrix{Any}(undef, 1, 1)
    A_cell[1, 1] = pess_to_ipeps_tensor(random_triangle_pess(1, 1))

    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)
    sweep = graded_ctm_cell_sweep(setup.CTM, setup.double_layer.AA)
    rho = graded_ob_2x2_norm(sweep.CTM, setup.double_layer.AA, 1, 1)

    @test isfinite(abs(rho))
    @test abs(rho) > 0
end

@testset "graded onsite observables" begin
    A_cell = Matrix{Any}(undef, 1, 1)
    A_cell[1, 1] = pess_to_ipeps_tensor(random_triangle_pess(1, 1))

    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)
    sweep = graded_ctm_cell_sweep(setup.CTM, setup.double_layer.AA)
    ops = tensorkit_spinful_operators()

    id_ob = graded_ob_onsite(sweep.CTM, ops.id, A_cell, setup.double_layer.AA, 1, 1)
    n_ob = graded_ob_onsite(sweep.CTM, ops.n_total, A_cell, setup.double_layer.AA, 1, 1)
    onsite = graded_spinful_onsite_observables(sweep.CTM, A_cell, setup.double_layer.AA)

    @test isapprox(id_ob, 1; atol=1e-10)
    @test isfinite(abs(n_ob))
    for site in (:LU, :RU, :LD, :RD)
        @test isapprox(graded_ob_onsite_at(sweep.CTM, ops.id, A_cell, setup.double_layer.AA, 1, 1; site), 1; atol=1e-10)
        @test isfinite(abs(graded_ob_onsite_at(sweep.CTM, ops.n_total, A_cell, setup.double_layer.AA, 1, 1; site)))
    end
    @test_throws ArgumentError graded_ob_onsite_at(sweep.CTM, ops.id, A_cell, setup.double_layer.AA, 1, 1; site=:bad)
    @test isapprox(
        graded_ob_product_2x2(sweep.CTM, (LU=ops.id, RU=ops.id, LD=ops.id, RD=ops.id), A_cell, setup.double_layer.AA, 1, 1),
        1;
        atol=1e-10,
    )
    @test isapprox(
        graded_ob_product_2x2(sweep.CTM, (LU=ops.n_total,), A_cell, setup.double_layer.AA, 1, 1),
        graded_ob_onsite_at(sweep.CTM, ops.n_total, A_cell, setup.double_layer.AA, 1, 1; site=:LU);
        atol=1e-10,
    )
    @test isfinite(abs(graded_ob_product_2x2(sweep.CTM, (LU=ops.n_total, RD=ops.n_double), A_cell, setup.double_layer.AA, 1, 1)))
    @test_throws ArgumentError graded_ob_product_2x2(sweep.CTM, (bad=ops.id,), A_cell, setup.double_layer.AA, 1, 1)
    @test size(onsite.n_total) == (1, 1)
    @test isapprox(onsite.id[1, 1], 1; atol=1e-10)
    @test isapprox(onsite.n_total, onsite.n_up + onsite.n_dn; atol=1e-10)
end

@testset "graded triangle observables" begin
    A_cell = Matrix{Any}(undef, 1, 1)
    A_cell[1, 1] = pess_to_ipeps_tensor(random_triangle_pess(1, 1))

    setup = graded_init_ctm_cell(A_cell; init_type="PBC", builder=:direct)
    sweep = graded_ctm_cell_sweep(setup.CTM, setup.double_layer.AA)

    V = physical_spinful_space()
    id3 = TensorKit.id(V * V * V)

    @test isapprox(graded_ob_triangle_2x2(sweep.CTM, id3, A_cell, setup.double_layer.AA, 1, 1; orientation=:up), 1; atol=1e-10)
    @test isapprox(graded_ob_triangle_2x2(sweep.CTM, id3, A_cell, setup.double_layer.AA, 1, 1; orientation=:down), 1; atol=1e-10)
    @test_throws ArgumentError graded_ob_triangle_2x2(sweep.CTM, id3, A_cell, setup.double_layer.AA, 1, 1; orientation=:bad)

    setting = SpinHallEnergySetting(Lx=1, Ly=1, magnetic_cell=1)
    pars = SpinHallParameters(t1=1.0, t2=1.0, U=0.1, mu=0.2, mx=0.0)
    coeffs = spin_hall_coefficients(setting, pars)
    h3 = triangle_hamiltonian_tensormap(coeffs, 1, 1)
    @test isfinite(abs(graded_ob_triangle_2x2(sweep.CTM, h3, A_cell, setup.double_layer.AA, 1, 1; orientation=:up)))
    @test isfinite(abs(graded_ob_triangle_2x2(sweep.CTM, h3, A_cell, setup.double_layer.AA, 1, 1; orientation=:down)))
end
