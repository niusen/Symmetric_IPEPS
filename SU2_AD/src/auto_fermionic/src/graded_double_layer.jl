"""
    graded_build_double_layer(A)

Build the fused CTMRG double-layer tensor from a rank-5 fermionic iPEPS tensor
`A` ordered as `(left, down, right, up, physical)`.

This is the graded counterpart of the legacy `build_double_layer_swap(A', A)`:
the explicit swap/parity gates are deliberately absent.  The tensor must live
in a fermionic graded TensorKit space, so that permutations and fusions carry
the fermionic braiding signs.
"""
function _graded_build_double_layer_from_pair(Ap::TensorKit.AbstractTensorMap, A::TensorKit.AbstractTensorMap)
    Ap = permute(Ap, ((1, 2), (3, 4, 5)))
    A = permute(A, ((1, 2), (3, 4, 5)))

    U_L = unitary(fuse(space(Ap, 1) * space(A, 1)), space(Ap, 1) * space(A, 1))
    U_D = unitary(fuse(space(Ap, 2) * space(A, 2)), space(Ap, 2) * space(A, 2))
    U_R = unitary(space(Ap, 3)' * space(A, 3)', fuse(space(Ap, 3)' * space(A, 3)'))
    U_U = unitary(space(Ap, 4)' * space(A, 4)', fuse(space(Ap, 4)' * space(A, 4)'))

    U_tem = unitary(fuse(space(A, 1) * space(A, 2)), space(A, 1) * space(A, 2))
    vM = U_tem * A
    uM = U_tem'

    U_temp = unitary(fuse(space(Ap, 1) * space(Ap, 2)), space(Ap, 1) * space(Ap, 2))
    vMp = U_temp * Ap
    uMp = U_temp'

    uMp = permute(uMp, ((1, 2, 3), ()))
    uM = permute(uM, ((1, 2, 3), ()))

    Vp = space(uMp, 3)
    V = space(vM, 1)
    U = unitary(fuse(Vp' * V), Vp' * V)

    @tensor double_LD[:] := uMp[-1, -2, 1] * U'[1, -3, -4]
    @tensor double_LD[:] := double_LD[-1, -3, 1, -5] * uM[-2, -4, 1]

    vMp = permute(vMp, ((1, 2, 3, 4), ()))
    vM = permute(vM, ((1, 2, 3, 4), ()))

    @tensor double_RU[:] := U[-1, -2, 1] * vM[1, -3, -4, -5]
    @tensor double_RU[:] := vMp[1, -2, -4, 2] * double_RU[-1, 1, -3, -5, 2]

    double_LD = permute(double_LD, ((1, 2), (3, 4, 5)))
    double_LD = U_L * double_LD
    double_LD = permute(double_LD, ((2, 3), (1, 4)))
    double_LD = U_D * double_LD
    double_LD = permute(double_LD, ((2, 1), (3,)))

    double_RU = permute(double_RU, ((1, 4, 5), (2, 3)))
    double_RU = double_RU * U_R
    double_RU = permute(double_RU, ((1, 4), (2, 3)))
    double_RU = double_RU * U_U

    double_LD = permute(double_LD, ((1, 2), (3,)))
    double_RU = permute(double_RU, ((1,), (2, 3)))
    AA_fused = double_LD * double_RU
    AA_fused = permute(AA_fused, (1, 2, 3, 4))

    return AA_fused, U_L, U_D, U_R, U_U
end

function _dense_nobraid_permute(t::TensorKit.AbstractTensorMap, p)
    target = permute(t, p)
    flat_order = p isa Tuple{<:Tuple,<:Tuple} ? (p[1]..., p[2]...) : p
    data = permutedims(convert(Array, t), flat_order)
    return TensorMap(data, codomain(target), domain(target))
end

function _graded_build_double_layer_planar_from_pair(Ap::TensorKit.AbstractTensorMap, A::TensorKit.AbstractTensorMap)
    Ap = _dense_nobraid_permute(Ap, ((1, 2), (3, 4, 5)))
    A = _dense_nobraid_permute(A, ((1, 2), (3, 4, 5)))

    U_L = unitary(fuse(space(Ap, 1) * space(A, 1)), space(Ap, 1) * space(A, 1))
    U_D = unitary(fuse(space(Ap, 2) * space(A, 2)), space(Ap, 2) * space(A, 2))
    U_R = unitary(space(Ap, 3)' * space(A, 3)', fuse(space(Ap, 3)' * space(A, 3)'))
    U_U = unitary(space(Ap, 4)' * space(A, 4)', fuse(space(Ap, 4)' * space(A, 4)'))

    U_tem = unitary(fuse(space(A, 1) * space(A, 2)), space(A, 1) * space(A, 2))
    vM = U_tem * A
    uM = U_tem'

    U_temp = unitary(fuse(space(Ap, 1) * space(Ap, 2)), space(Ap, 1) * space(Ap, 2))
    vMp = U_temp * Ap
    uMp = U_temp'

    uMp = _dense_nobraid_permute(uMp, ((1, 2, 3), ()))
    uM = _dense_nobraid_permute(uM, ((1, 2, 3), ()))

    Vp = space(uMp, 3)
    V = space(vM, 1)
    U = unitary(fuse(Vp' * V), Vp' * V)

    @tensor double_LD[:] := uMp[-1, -2, 1] * U'[1, -3, -4]
    @tensor double_LD[:] := double_LD[-1, -3, 1, -5] * uM[-2, -4, 1]

    vMp = _dense_nobraid_permute(vMp, ((1, 2, 3, 4), ()))
    vM = _dense_nobraid_permute(vM, ((1, 2, 3, 4), ()))

    @tensor double_RU[:] := U[-1, -2, 1] * vM[1, -3, -4, -5]
    @tensor double_RU[:] := vMp[1, -2, -4, 2] * double_RU[-1, 1, -3, -5, 2]

    double_LD = _dense_nobraid_permute(double_LD, ((1, 2), (3, 4, 5)))
    double_LD = U_L * double_LD
    double_LD = _dense_nobraid_permute(double_LD, ((2, 3), (1, 4)))
    double_LD = U_D * double_LD
    double_LD = _dense_nobraid_permute(double_LD, ((2, 1), (3,)))

    double_RU = _dense_nobraid_permute(double_RU, ((1, 4, 5), (2, 3)))
    double_RU = double_RU * U_R
    double_RU = _dense_nobraid_permute(double_RU, ((1, 4), (2, 3)))
    double_RU = double_RU * U_U

    double_LD = _dense_nobraid_permute(double_LD, ((1, 2), (3,)))
    double_RU = _dense_nobraid_permute(double_RU, ((1,), (2, 3)))
    AA_fused = double_LD * double_RU
    AA_fused = _dense_nobraid_permute(AA_fused, (1, 2, 3, 4))

    return AA_fused, U_L, U_D, U_R, U_U
end

function _graded_build_double_layer_direct_from_pair(Ap::TensorKit.AbstractTensorMap, A::TensorKit.AbstractTensorMap)
    U_L = unitary(fuse(space(Ap, 1) * space(A, 1)), space(Ap, 1) * space(A, 1))
    U_D = unitary(fuse(space(Ap, 2) * space(A, 2)), space(Ap, 2) * space(A, 2))
    U_R = unitary(space(Ap, 3)' * space(A, 3)', fuse(space(Ap, 3)' * space(A, 3)'))
    U_U = unitary(space(Ap, 4)' * space(A, 4)', fuse(space(Ap, 4)' * space(A, 4)'))

    @tensor AA8[-1, -2, -3, -4, -5, -6, -7, -8] := Ap[-1, -2, -3, -4, 1] * A[-5, -6, -7, -8, 1]
    @tensor AA_L[-1, -2, -3, -4, -5, -6, -7] := U_L[-1, 1, 2] * AA8[1, -2, -3, -4, 2, -5, -6, -7]
    @tensor AA_LD[-1, -2, -3, -4, -5, -6] := U_D[-2, 1, 2] * AA_L[-1, 1, -3, -4, 2, -5, -6]
    @tensor AA_LDR[-1, -2, -3, -4, -5] := AA_LD[-1, -2, 1, -4, 2, -5] * U_R[1, 2, -3]
    @tensor AA_fused[-1, -2, -3, -4] := AA_LDR[-1, -2, -3, 1, 2] * U_U[1, 2, -4]

    return AA_fused, U_L, U_D, U_R, U_U
end

function _graded_build_open_double_layer_direct_from_pair(Ap::TensorKit.AbstractTensorMap, A::TensorKit.AbstractTensorMap)
    U_L = unitary(fuse(space(Ap, 1) * space(A, 1)), space(Ap, 1) * space(A, 1))
    U_D = unitary(fuse(space(Ap, 2) * space(A, 2)), space(Ap, 2) * space(A, 2))
    U_R = unitary(space(Ap, 3)' * space(A, 3)', fuse(space(Ap, 3)' * space(A, 3)'))
    U_U = unitary(space(Ap, 4)' * space(A, 4)', fuse(space(Ap, 4)' * space(A, 4)'))

    @tensor AA10[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10] := Ap[-1, -2, -3, -4, -9] * A[-5, -6, -7, -8, -10]
    @tensor AA_L[-1, -2, -3, -4, -5, -6, -7, -8, -9] := U_L[-1, 1, 2] * AA10[1, -2, -3, -4, 2, -5, -6, -7, -8, -9]
    @tensor AA_LD[-1, -2, -3, -4, -5, -6, -7, -8] := U_D[-2, 1, 2] * AA_L[-1, 1, -3, -4, 2, -5, -6, -7, -8]
    @tensor AA_LDR[-1, -2, -3, -4, -5, -6, -7] := AA_LD[-1, -2, 1, -4, 2, -5, -6, -7] * U_R[1, 2, -3]
    @tensor AA_fused[-1, -2, -3, -4, -5, -6] := AA_LDR[-1, -2, -3, 1, 2, -5, -6] * U_U[1, 2, -4]

    return AA_fused, U_L, U_D, U_R, U_U
end

graded_build_double_layer_direct(A::TensorKit.AbstractTensorMap) =
    _graded_build_double_layer_direct_from_pair(A', A)

graded_build_open_double_layer_direct(A::TensorKit.AbstractTensorMap) =
    _graded_build_open_double_layer_direct_from_pair(A', A)

function graded_build_double_layer_direct(A::TensorKit.AbstractTensorMap, O::TensorKit.AbstractTensorMap)
    @tensor Aop[-1, -2, -3, -4, -5] := A[-1, -2, -3, -4, 1] * O[-5, 1]
    return _graded_build_double_layer_direct_from_pair(A', Aop)
end

function graded_build_double_layer(A::TensorKit.AbstractTensorMap)
    return _graded_build_double_layer_from_pair(A', A)
end

function _fermion_parity_projector(V, parity::Integer)
    even_dim = dim(V, fsector(0))
    odd_dim = dim(V, fsector(1))
    subspace = isodd(parity) ? fZ2space(0, odd_dim; dual=isdual(V)) :
               fZ2space(even_dim, 0; dual=isdual(V))
    return isometry(V, subspace)'
end

function _apply_legacy_fused_parity_corrections(AA_fused, U_L, U_D, U_R, U_U)
    P_odd_Lp = _fermion_parity_projector(space(U_L', 1), 1)
    P_odd_Up = _fermion_parity_projector(space(U_U', 2), 1)
    P_odd_U = _fermion_parity_projector(space(U_U', 3), 1)

    @tensor isom_Lp[:] := U_L[-1, 4, 3] * P_odd_Lp'[4, 1] * P_odd_Lp[1, 2] * U_L'[2, 3, -2]
    @tensor isom_U[:] := U_U[3, 4, -1] * P_odd_U'[4, 1] * P_odd_U[1, 2] * U_U'[-2, 3, 2]
    @tensor isom_Up_U[:] := U_U[3, 4, -1] * P_odd_Up'[3, 1] * P_odd_Up[1, 5] * P_odd_U'[4, 2] * P_odd_U[2, 6] * U_U'[-2, 5, 6]
    @tensor AA_Lp_U[:] := AA_fused[1, -2, -3, 4] * isom_Lp[-1, 1] * isom_U[-4, 4]
    AA_fused = AA_fused - 2 * AA_Lp_U
    @tensor AA_Up_U[:] := AA_fused[-1, -2, -3, 4] * isom_Up_U[-4, 4]
    AA_fused = AA_fused - 2 * AA_Up_U

    P_odd_Dp = _fermion_parity_projector(space(U_D', 1), 1)
    P_odd_D = _fermion_parity_projector(space(U_D', 2), 1)
    P_odd_R = _fermion_parity_projector(space(U_R', 3), 1)

    @tensor isom_Dp[:] := U_D[-1, 4, 3] * P_odd_Dp'[4, 1] * P_odd_Dp[1, 2] * U_D'[2, 3, -2]
    @tensor isom_R[:] := U_R[3, 4, -1] * P_odd_R'[4, 1] * P_odd_R[1, 2] * U_R'[-2, 3, 2]
    @tensor isom_Dp_D[:] := U_D[-1, 3, 4] * P_odd_Dp'[3, 1] * P_odd_Dp[1, 5] * P_odd_D'[4, 2] * P_odd_D[2, 6] * U_D'[5, 6, -2]
    @tensor AA_Dp_D[:] := AA_fused[-1, 2, -3, -4] * isom_Dp_D[-2, 2]
    AA_fused = AA_fused - 2 * AA_Dp_D
    @tensor AA_Dp_R[:] := AA_fused[-1, 2, 3, -4] * isom_Dp[-2, 2] * isom_R[-3, 3]
    AA_fused = AA_fused - 2 * AA_Dp_R

    return AA_fused
end

function _graded_build_double_layer_with_legacy_fused_parity_from_pair(Ap, A)
    AA_fused, U_L, U_D, U_R, U_U = _graded_build_double_layer_from_pair(Ap, A)
    return _apply_legacy_fused_parity_corrections(AA_fused, U_L, U_D, U_R, U_U),
           U_L, U_D, U_R, U_U
end
