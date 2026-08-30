using Test

include("FullUpdate_iPESS_J1_Jchi_up.jl")

global Lx = 1
global Ly = 1
global multiplet_tol = 1e-5

V = Rep[SU₂](0 => 1, 1 / 2 => 1)
Vp = Rep[SU₂](1 / 2 => 1)
B_set, T_set, _, _, _ = initial_iPESS_uniform(1, 1, V, Vp)
physical_space = space(T_set[1, 1], 2)

@testset "bosonic triangular iPESS local full update" begin
    gate_identity = bosonic_triangle_J1_gate(0.0, physical_space)
    identity_map = unitary(codomain(gate_identity), domain(gate_identity))
    @test norm(gate_identity - identity_map) < 1e-12

    gate_J1 = bosonic_triangle_J1_gate(0.01, physical_space; J1=1, Jchi_up=0)
    gate_chiral = bosonic_triangle_J1_gate(
        0.01,
        physical_space;
        J1=1,
        Jchi_up=0.2,
    )
    @test norm(gate_chiral - gate_J1) > 1e-8

    parts = bosonic_split_3tensors(
        T_set[1, 1],
        T_set[1, 1],
        T_set[1, 1],
        B_set[1, 1],
        gate_J1,
    )
    B1_res, _, _, _, _, _, big_triangle, evolved_triangle = parts
    @test norm(evolved_triangle) > 0

    double_simplex, _, _, _ = bosonic_build_double_layer_noswap_Tm(
        B_set[1, 1]',
        B_set[1, 1],
        false,
    )
    double_site, _, _, _ = bosonic_build_double_layer_noswap_Bm(
        T_set[1, 1]',
        T_set[1, 1],
        true,
    )
    double_residual, _, _, _ = bosonic_build_double_layer_noswap_Bm(
        B1_res',
        B1_res,
        false,
    )
    @test norm(double_simplex) > 0
    @test norm(double_site) > 0
    @test norm(double_residual) > 0

    truncated = bosonic_truncation_direct(
        evolved_triangle,
        4,
        "simultaneous",
        1e-8,
    )
    B_new, T1_new, T2_new, T3_new, compressed = truncated
    @test norm(B_new) > 0
    @test norm(T1_new) > 0
    @test norm(T2_new) > 0
    @test norm(T3_new) > 0
    @test norm(compressed) > 0
    @test maximum(dim(space(B_new, leg)) for leg in 1:3) <= 4
    @test space(big_triangle) == space(evolved_triangle)
end
