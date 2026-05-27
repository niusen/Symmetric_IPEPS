using LinearAlgebra: I, norm
using Zygote: @ignore_derivatives

function chiral_pair_build_double_layer_extra_leg(A, operator)
    # operator has two physical legs plus one extra SU2 leg, e.g. from tsvd(S_i.S_j).
    A = permute(A, (1, 2,), (3, 4, 5,))
    U_L = @ignore_derivatives unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1)) * (1 + 0im)
    U_D = @ignore_derivatives unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2)) * (1 + 0im)
    U_R = @ignore_derivatives unitary(space(A, 3) ⊗ space(A, 3)', fuse(space(A, 3)' ⊗ space(A, 3))) * (1 + 0im)
    U_U = @ignore_derivatives unitary(space(A, 4) ⊗ space(A, 4)', fuse(space(A, 4)' ⊗ space(A, 4))) * (1 + 0im)

    uM, sM, vM = tsvd(A)
    uM = uM * sM

    uM = permute(uM, (1, 2, 3,), ())
    V = space(vM, 1)
    U = @ignore_derivatives unitary(fuse(V' ⊗ V), V' ⊗ V) * (1 + 0im)
    @tensor double_LD[:] := uM'[-1, -2, 1] * U'[1, -3, -4]
    @tensor double_LD[:] := double_LD[-1, -3, 1, -5] * uM[-2, -4, 1]

    vM = permute(vM, (1, 2, 3, 4,), ())
    @tensor double_RU[:] := U[-1, -2, 1] * vM[1, -3, -4, -5]
    @tensor double_RU[:] := vM'[3, -2, -4, 1] * operator[1, 2, -6] *
        double_RU[-1, 3, -3, -5, 2]

    double_LD = permute(double_LD, (1, 2,), (3, 4, 5,))
    double_LD = U_L * double_LD
    double_LD = permute(double_LD, (2, 3,), (1, 4,))
    double_LD = U_D * double_LD
    double_LD = permute(double_LD, (2, 1,), (3,))

    double_RU = permute(double_RU, (1, 4, 5, 6,), (2, 3,))
    double_RU = double_RU * U_R
    double_RU = permute(double_RU, (1, 5, 4,), (2, 3,))
    double_RU = double_RU * U_U

    double_LD = permute(double_LD, (1, 2,), (3,))
    double_RU = permute(double_RU, (1,), (2, 4, 3,))
    AA_op = double_LD * double_RU
    return permute(AA_op, (1, 2, 3, 4, 5,), ())
end

function _chiral_pair_single_copy_spin_svd()
    U_pair = @ignore_derivatives chiral_pair_fuse_unitary()
    Vcopy = space(U_pair, 2)
    sx = ComplexF64[0 1; 1 0] / 2
    sy = ComplexF64[0 -im; im 0] / 2
    sz = ComplexF64[1 0; 0 -1] / 2
    @tensor HSS[:] := sx[-1, -3] * sx[-2, -4] +
        sy[-1, -3] * sy[-2, -4] +
        sz[-1, -3] * sz[-2, -4]
    HSS = TensorMap(HSS, Vcopy ⊗ Vcopy ← Vcopy ⊗ Vcopy)
    HSS = permute(HSS, (1, 3,), (2, 4,))
    u, s, v = tsvd(HSS)
    return u * s, permute(v, (2, 3,), (1,))
end

function _chiral_pair_lift_copy_operator(op, copy::Symbol)
    @assert copy in (:A, :B)
    U_pair = @ignore_derivatives chiral_pair_fuse_unitary()
    Vcopy_A = space(U_pair, 2)
    Vcopy_B = space(U_pair, 3)
    IdA = TensorMap(Matrix{Float64}(I, dim(Vcopy_A), dim(Vcopy_A)), Vcopy_A, Vcopy_A)
    IdB = TensorMap(Matrix{Float64}(I, dim(Vcopy_B), dim(Vcopy_B)), Vcopy_B, Vcopy_B)

    if copy == :A
        @tensor op_pair[:] := U_pair'[1, 2, -1] * op[1, 3, -3] * IdB[2, 4] *
            U_pair[-2, 3, 4]
    else
        @tensor op_pair[:] := U_pair'[1, 2, -1] * IdA[1, 3] * op[2, 4, -3] *
            U_pair[-2, 3, 4]
    end

    return op_pair
end

function chiral_pair_single_spin_operator(copy::Symbol)
    op_left, op_right = _chiral_pair_single_copy_spin_svd()
    return _chiral_pair_lift_copy_operator(op_left, copy),
        _chiral_pair_lift_copy_operator(op_right, copy)
end

function _chiral_pair_scalar(x)
    if x isa Number
        return x
    end
    try
        return blocks(x)[Irrep[SU₂](0)][1]
    catch
    end
    try
        return only(x.data)
    catch
    end
    return x
end

function _chiral_pair_correl_norm_x(AA_fused, CTM, distance::Int)
    vals = Vector{Any}(undef, distance)
    C1 = CTM.Cset.C1
    C2 = CTM.Cset.C2
    C3 = CTM.Cset.C3
    C4 = CTM.Cset.C4
    T1 = CTM.Tset.T1
    T2 = CTM.Tset.T2
    T3 = CTM.Tset.T3
    T4 = CTM.Tset.T4

    @tensor va[:] := C1[1, 3] * T4[2, 5, 1] * C4[7, 2] *
        T1[3, 4, -1] * AA_fused[5, 6, -2, 4] * T3[-3, 6, 7]
    @tensor vb[:] := T1[-1, 4, 3] * AA_fused[-2, 6, 5, 4] *
        T3[7, 6, -3] * C2[3, 1] * T2[1, 5, 2] * C3[2, 7]
    vals[1] = _chiral_pair_scalar(@tensor va[1, 2, 3] * vb[1, 2, 3])

    for dis in 2:distance
        @tensor va[:] := va[1, 3, 5] * T1[1, 2, -1] *
            AA_fused[3, 4, -2, 2] * T3[-3, 4, 5]
        vals[dis] = _chiral_pair_scalar(@tensor va[1, 2, 3] * vb[1, 2, 3])
    end
    return vals
end

function _chiral_pair_correl_spinspin_x(AA_fused, AA_op1, AA_op2, CTM, distance::Int)
    vals = Vector{Any}(undef, distance)
    C1 = CTM.Cset.C1
    C2 = CTM.Cset.C2
    C3 = CTM.Cset.C3
    C4 = CTM.Cset.C4
    T1 = CTM.Tset.T1
    T2 = CTM.Tset.T2
    T3 = CTM.Tset.T3
    T4 = CTM.Tset.T4

    @tensor va[:] := C1[1, 3] * T4[2, 5, 1] * C4[7, 2] *
        T1[3, 4, -1] * AA_op1[5, 6, -2, 4, -4] * T3[-3, 6, 7]
    @tensor vb[:] := T1[-1, 4, 3] * AA_op2[-2, 6, 5, 4, -4] *
        T3[7, 6, -3] * C2[3, 1] * T2[1, 5, 2] * C3[2, 7]
    vals[1] = _chiral_pair_scalar(@tensor va[1, 2, 3, 4] * vb[1, 2, 3, 4])

    for dis in 2:distance
        @tensor va[:] := va[1, 3, 5, -4] * T1[1, 2, -1] *
            AA_fused[3, 4, -2, 2] * T3[-3, 4, 5]
        vals[dis] = _chiral_pair_scalar(@tensor va[1, 2, 3, 4] * vb[1, 2, 3, 4])
    end
    return vals
end

function chiral_pair_spinspin_correl_x(A, AA_fused, CTM; distance::Int=20)
    A = A / norm(A)
    SA_L, SA_R = chiral_pair_single_spin_operator(:A)
    SB_L, SB_R = chiral_pair_single_spin_operator(:B)

    AA_SA_L = chiral_pair_build_double_layer_extra_leg(A, SA_L)
    AA_SA_R = chiral_pair_build_double_layer_extra_leg(A, SA_R)
    AA_SB_L = chiral_pair_build_double_layer_extra_leg(A, SB_L)
    AA_SB_R = chiral_pair_build_double_layer_extra_leg(A, SB_R)

    norms = _chiral_pair_correl_norm_x(AA_fused, CTM, distance)
    corr_AA = _chiral_pair_correl_spinspin_x(AA_fused, AA_SA_L, AA_SA_R, CTM, distance) ./ norms
    corr_BB = _chiral_pair_correl_spinspin_x(AA_fused, AA_SB_L, AA_SB_R, CTM, distance) ./ norms
    corr_AB = _chiral_pair_correl_spinspin_x(AA_fused, AA_SA_L, AA_SB_R, CTM, distance) ./ norms
    corr_BA = _chiral_pair_correl_spinspin_x(AA_fused, AA_SB_L, AA_SA_R, CTM, distance) ./ norms

    return (
        distance=collect(1:distance),
        norm=norms,
        spin_AA=corr_AA,
        spin_BB=corr_BB,
        spin_AB=corr_AB,
        spin_BA=corr_BA,
    )
end

function chiral_pair_spinspin_correl_x(
    state::Square_iPEPS,
    chi::Int,
    ctm_setting;
    distance::Int=20,
    parameters=nothing,
    init=initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true),
    init_CTM=[],
)
    A = state.T / norm(state.T)
    CTM, AA_fused, U_L, U_D, U_R, U_U, ite_num, ite_err =
        chiral_pair_ctmrg(A, chi, init, init_CTM, ctm_setting)
    correl = chiral_pair_spinspin_correl_x(A, AA_fused, CTM; distance=distance)

    if parameters === nothing
        return merge(correl, (ctm_ite_num=ite_num, ctm_ite_err=ite_err, CTM=CTM))
    end

    E, E_triangles, triangles = evaluate_chiral_pair_triangle(
        A,
        AA_fused,
        U_L,
        U_D,
        U_R,
        U_U,
        CTM,
        ctm_setting,
        parameters,
    )
    obs = chiral_pair_observables(triangles, parameters)

    return merge(
        correl,
        (
            energy=real(E),
            E_triangles=real.(collect(E_triangles)),
            observables=obs,
            ctm_ite_num=ite_num,
            ctm_ite_err=ite_err,
            CTM=CTM,
        ),
    )
end
