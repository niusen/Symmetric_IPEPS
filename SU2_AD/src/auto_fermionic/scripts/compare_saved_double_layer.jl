using LinearAlgebra
using TensorKit
using JLD2
using AutoFermionicPESS

const AUTO_DIR = normpath(joinpath(@__DIR__, ".."))
const SU2_AD_SRC = normpath(joinpath(AUTO_DIR, ".."))
const SU2_AD_ROOT = normpath(joinpath(SU2_AD_SRC, ".."))
const SPINHALL_VAR_DIR = joinpath(SU2_AD_ROOT, "Triangle", "iPESS", "spinHall", "variational")

const DEFAULT_STATE_FILE = joinpath(
    SPINHALL_VAR_DIR,
    "var_iPESS_Z2_Triangle_Hofstadter_Hubbard_spinHall_D4_chi_40_-2.3854.jld2",
)

const TIMES_SYMBOL = Symbol(Char(0x00d7))
if !isdefined(Main, TIMES_SYMBOL)
    @eval Main const $(TIMES_SYMBOL) = getproperty(TensorKit, $(QuoteNode(TIMES_SYMBOL)))
end

function TensorKit.permute(
    t::TensorKit.AbstractTensorMap,
    codomain_order::Tuple,
    domain_order::Tuple;
    copy::Bool=false,
)
    return TensorKit.permute(t, (codomain_order, domain_order); copy=copy)
end

function include_legacy_double_layer_code()
    include(joinpath(SU2_AD_SRC, "bosonic", "iPEPS_ansatz.jl"))
    include(joinpath(SU2_AD_SRC, "fermionic", "swap_funs.jl"))
    include(joinpath(SU2_AD_SRC, "fermionic", "Fermionic_CTMRG.jl"))
    return nothing
end

function legacy_pess_to_ipeps_tensor(pess)
    return permute(pess.Tm * pess.Bm, (1, 5, 4, 2, 3))
end

function fermion_parity_labels(V)
    even_dim = TensorKit.dim(V, fsector(0))
    odd_dim = TensorKit.dim(V, fsector(1))
    return vcat(fill(0, even_dim), fill(1, odd_dim))
end

function legacy_initial_ap_signs(Ap)
    data = convert(Array, Ap)
    labels = [fermion_parity_labels(space(Ap, n)) for n in 1:5]
    signed = similar(data)
    for I in CartesianIndices(data)
        p1 = labels[1][I[1]]
        p2 = labels[2][I[2]]
        p3 = labels[3][I[3]]
        p4 = labels[4][I[4]]
        exponent = p1 * p4 + p2 * p3 + p4 + p2
        signed[I] = isodd(exponent) ? -data[I] : data[I]
    end
    return TensorMap(signed, codomain(Ap), domain(Ap))
end

function apply_external_parity_mask(T, mask::Integer)
    data = convert(Array, T)
    labels = [fermion_parity_labels(space(T, n)) for n in 1:ndims(data)]
    signed = similar(data)
    for I in CartesianIndices(data)
        exponent = 0
        for n in 1:ndims(data)
            if !iszero(mask & (1 << (n - 1)))
                exponent += labels[n][I[n]]
            end
        end
        signed[I] = isodd(exponent) ? -data[I] : data[I]
    end
    return TensorMap(signed, codomain(T), domain(T))
end

function best_external_parity_mask(T, target)
    best_mask = 0
    best_diff = Inf
    for mask in 0:(2^numind(T) - 1)
        diff = norm(apply_external_parity_mask(T, mask) - target)
        if diff < best_diff
            best_mask = mask
            best_diff = diff
        end
    end
    return best_mask, best_diff, best_diff / max(norm(target), eps(real(scalartype(target))))
end

function legacy_build_double_layer_noswap_nofinal(Ap, A)
    Ap = permute(Ap, (1, 2), (3, 4, 5))
    A = permute(A, (1, 2), (3, 4, 5))

    U_L = unitary(fuse(space(Ap, 1) * space(A, 1)), space(Ap, 1) * space(A, 1))
    U_D = unitary(fuse(space(Ap, 2) * space(A, 2)), space(Ap, 2) * space(A, 2))
    U_R = unitary(space(Ap, 3)' * space(A, 3)', fuse(space(Ap, 3)' * space(A, 3)'))
    U_U = unitary(space(Ap, 4)' * space(A, 4)', fuse(space(Ap, 4)' * space(A, 4)'))

    U_tem = unitary(fuse(space(A, 1) * space(A, 2)), space(A, 1) * space(A, 2)) * (1 + 0im)
    vM = U_tem * A
    uM = U_tem'
    U_temp = unitary(fuse(space(Ap, 1) * space(Ap, 2)), space(Ap, 1) * space(Ap, 2)) * (1 + 0im)
    vMp = U_temp * Ap
    uMp = U_temp'

    uMp = permute(uMp, (1, 2, 3), ())
    uM = permute(uM, (1, 2, 3), ())
    Vp = space(uMp, 3)
    V = space(vM, 1)
    U = unitary(fuse(Vp' * V), Vp' * V)

    @tensor double_LD[:] := uMp[-1, -2, 1] * U'[1, -3, -4]
    @tensor double_LD[:] := double_LD[-1, -3, 1, -5] * uM[-2, -4, 1]

    vMp = permute(vMp, (1, 2, 3, 4), ())
    vM = permute(vM, (1, 2, 3, 4), ())

    @tensor double_RU[:] := U[-1, -2, 1] * vM[1, -3, -4, -5]
    @tensor double_RU[:] := vMp[1, -2, -4, 2] * double_RU[-1, 1, -3, -5, 2]

    double_LD = permute(double_LD, (1, 2), (3, 4, 5))
    double_LD = U_L * double_LD
    double_LD = permute(double_LD, (2, 3), (1, 4))
    double_LD = U_D * double_LD
    double_LD = permute(double_LD, (2, 1), (3,))

    double_RU = permute(double_RU, (1, 4, 5), (2, 3))
    double_RU = double_RU * U_R
    double_RU = permute(double_RU, (1, 4), (2, 3))
    double_RU = double_RU * U_U

    double_LD = permute(double_LD, (1, 2), (3,))
    double_RU = permute(double_RU, (1,), (2, 3))
    AA_fused = double_LD * double_RU
    return permute(AA_fused, (1, 2, 3, 4))
end

function compare_saved_double_layer(;
    state_file::AbstractString=DEFAULT_STATE_FILE,
    site::Tuple{Int,Int}=(1, 1),
)
    include_legacy_double_layer_code()

    data = load(state_file)
    haskey(data, "x") || error("Expected key `x` in saved variational state.")
    x = data["x"]
    fx = z2_to_fermion_state(x)

    A_legacy = legacy_pess_to_ipeps_tensor(x[site...])
    A_graded = pess_to_ipeps_tensor(fx[site...])
    A_legacy_fermion = legacy_z2_pess_to_fermion_ipeps_tensor(x[site...])
    A_legacy_direct = z2_to_fermion_tensor(A_legacy)
    Ap_legacy_fermion_from_z2_adjoint = z2_to_fermion_tensor(A_legacy')

    AA_legacy, = Base.invokelatest(build_double_layer_swap, A_legacy', A_legacy)
    AA_legacy_noswap_nofinal = legacy_build_double_layer_noswap_nofinal(A_legacy', A_legacy)
    AA_graded, = graded_build_double_layer(A_graded)
    AA_graded_from_legacy_A, = graded_build_double_layer(A_legacy_fermion)
    AA_graded_direct_from_legacy_A, = AutoFermionicPESS._graded_build_double_layer_direct_from_pair(
        A_legacy_fermion',
        A_legacy_fermion,
    )
    AA_graded_from_z2_adjoint_Ap, = AutoFermionicPESS._graded_build_double_layer_from_pair(
        Ap_legacy_fermion_from_z2_adjoint,
        A_legacy_fermion,
    )
    AA_graded_planar_from_legacy_A, = AutoFermionicPESS._graded_build_double_layer_planar_from_pair(
        A_legacy_fermion',
        A_legacy_fermion,
    )
    AA_graded_legacy_fused_parity, = AutoFermionicPESS._graded_build_double_layer_with_legacy_fused_parity_from_pair(
        A_legacy_fermion',
        A_legacy_fermion,
    )
    AA_graded_initial_signed, = AutoFermionicPESS._graded_build_double_layer_from_pair(
        legacy_initial_ap_signs(A_legacy_fermion'),
        A_legacy_fermion,
    )
    AA_legacy_fermion = z2_to_fermion_tensor(AA_legacy)
    AA_legacy_noswap_nofinal_fermion = z2_to_fermion_tensor(AA_legacy_noswap_nofinal)

    legacy_dense = convert(Array, AA_legacy)
    graded_dense = convert(Array, AA_graded)
    diff_norm = norm(graded_dense - legacy_dense)
    rel_diff = diff_norm / max(norm(legacy_dense), eps(real(eltype(legacy_dense))))
    same_category_diff = norm(AA_graded - AA_legacy_fermion)
    same_category_rel_diff = same_category_diff / max(norm(AA_legacy_fermion), eps(real(eltype(legacy_dense))))
    A_diff = norm(A_graded - A_legacy_fermion)
    A_rel_diff = A_diff / max(norm(A_legacy_fermion), eps(real(eltype(legacy_dense))))
    A_bridge_diff = norm(A_legacy_direct - A_legacy_fermion)
    Ap_adjoint_path_diff = norm(A_legacy_fermion' - Ap_legacy_fermion_from_z2_adjoint)
    Ap_adjoint_path_rel_diff = Ap_adjoint_path_diff / max(norm(Ap_legacy_fermion_from_z2_adjoint), eps(real(eltype(legacy_dense))))
    builder_diff = norm(AA_graded_from_legacy_A - AA_legacy_fermion)
    builder_rel_diff = builder_diff / max(norm(AA_legacy_fermion), eps(real(eltype(legacy_dense))))
    direct_diff = norm(AA_graded_direct_from_legacy_A - AA_legacy_fermion)
    direct_rel_diff = direct_diff / max(norm(AA_legacy_fermion), eps(real(eltype(legacy_dense))))
    direct_vs_legacy_noswap_diff = norm(AA_graded_direct_from_legacy_A - AA_legacy_noswap_nofinal_fermion)
    direct_vs_legacy_noswap_rel_diff = direct_vs_legacy_noswap_diff / max(norm(AA_legacy_noswap_nofinal_fermion), eps(real(eltype(legacy_dense))))
    builder_vs_legacy_noswap_diff = norm(AA_graded_from_legacy_A - AA_legacy_noswap_nofinal_fermion)
    builder_vs_legacy_noswap_rel_diff = builder_vs_legacy_noswap_diff / max(norm(AA_legacy_noswap_nofinal_fermion), eps(real(eltype(legacy_dense))))
    planar_vs_legacy_noswap_diff = norm(AA_graded_planar_from_legacy_A - AA_legacy_noswap_nofinal_fermion)
    planar_vs_legacy_noswap_rel_diff = planar_vs_legacy_noswap_diff / max(norm(AA_legacy_noswap_nofinal_fermion), eps(real(eltype(legacy_dense))))
    z2_adjoint_Ap_diff = norm(AA_graded_from_z2_adjoint_Ap - AA_legacy_fermion)
    z2_adjoint_Ap_rel_diff = z2_adjoint_Ap_diff / max(norm(AA_legacy_fermion), eps(real(eltype(legacy_dense))))
    planar_diff = norm(AA_graded_planar_from_legacy_A - AA_legacy_fermion)
    planar_rel_diff = planar_diff / max(norm(AA_legacy_fermion), eps(real(eltype(legacy_dense))))
    legacy_fused_parity_diff = norm(AA_graded_legacy_fused_parity - AA_legacy_fermion)
    legacy_fused_parity_rel_diff = legacy_fused_parity_diff / max(norm(AA_legacy_fermion), eps(real(eltype(legacy_dense))))
    initial_signed_diff = norm(AA_graded_initial_signed - AA_legacy_fermion)
    initial_signed_rel_diff = initial_signed_diff / max(norm(AA_legacy_fermion), eps(real(eltype(legacy_dense))))
    best_A_mask, best_A_diff, best_A_rel_diff = best_external_parity_mask(A_graded, A_legacy_fermion)
    best_AA_mask, best_AA_diff, best_AA_rel_diff = best_external_parity_mask(AA_graded_from_legacy_A, AA_legacy_fermion)

    println("state_file = ", state_file)
    println("site = ", site)
    println("legacy AA space = ", space(AA_legacy))
    println("graded AA space = ", space(AA_graded))
    println("legacy AA dense size = ", size(legacy_dense))
    println("graded AA dense size = ", size(graded_dense))
    println("norm(graded A - converted legacy A) = ", A_diff)
    println("relative A difference = ", A_rel_diff)
    println("norm(direct converted legacy A - bridge converted legacy A) = ", A_bridge_diff)
    println("norm((converted legacy A)' - converted legacy A') = ", Ap_adjoint_path_diff)
    println("relative adjoint-path difference = ", Ap_adjoint_path_rel_diff)
    println("norm(graded AA - legacy AA) = ", diff_norm)
    println("relative double-layer difference = ", rel_diff)
    println("norm(graded AA - converted legacy AA) = ", same_category_diff)
    println("relative same-category difference = ", same_category_rel_diff)
    println("norm(graded AA from converted legacy A - converted legacy AA) = ", builder_diff)
    println("relative builder-only difference = ", builder_rel_diff)
    println("norm(direct graded AA from converted legacy A - converted legacy AA) = ", direct_diff)
    println("relative direct-builder difference = ", direct_rel_diff)
    println("relative graded builder vs legacy noswap/nofinal = ", builder_vs_legacy_noswap_rel_diff)
    println("relative direct builder vs legacy noswap/nofinal = ", direct_vs_legacy_noswap_rel_diff)
    println("relative no-braid builder vs legacy noswap/nofinal = ", planar_vs_legacy_noswap_rel_diff)
    println("norm(graded AA using converted legacy A' - converted legacy AA) = ", z2_adjoint_Ap_diff)
    println("relative converted-legacy-adjoint-Ap difference = ", z2_adjoint_Ap_rel_diff)
    println("norm(planar graded AA from converted legacy A - converted legacy AA) = ", planar_diff)
    println("relative planar-builder difference = ", planar_rel_diff)
    println("norm(legacy-fused-parity graded AA - converted legacy AA) = ", legacy_fused_parity_diff)
    println("relative legacy-fused-parity difference = ", legacy_fused_parity_rel_diff)
    println("norm(initial-signed graded AA - converted legacy AA) = ", initial_signed_diff)
    println("relative initial-signed difference = ", initial_signed_rel_diff)
    println("best external parity mask for A = ", best_A_mask)
    println("best external parity-gauge A relative difference = ", best_A_rel_diff)
    println("best external parity mask for AA = ", best_AA_mask)
    println("best external parity-gauge AA relative difference = ", best_AA_rel_diff)

    return nothing
end

compare_saved_double_layer()
