"""
GPU-safe tensor builders for the square-J1 Full Update.

`unitary(codomain, domain)` uses CPU `Vector` storage by default.  The original
CPU routines therefore cannot multiply those temporary maps by a CuArray-backed
iPEPS tensor.  These helpers keep the original contraction order, but match all
space-derived maps to both the scalar type and storage of the input tensor.

Only CuArray-specialized public methods are installed below.  The original CPU
CTMRG and Full Update methods remain unchanged.
"""

function _square_J1_unitary_like(reference::TensorKit.AbstractTensorMap, codomain, domain)
    tensor = unitary(codomain, domain)
    return square_J1_to_storage_like(tensor, reference)
end

function _square_J1_operator_like(reference::TensorKit.AbstractTensorMap, operator)
    operator isa AbstractVector && isempty(operator) && return operator
    operator isa TensorKit.AbstractTensorMap || return operator
    storage = square_J1_storage_family(TensorKit.storagetype(reference))
    if storage === Array && TensorKit.storagetype(operator) <: Array
        return operator
    elseif storage !== Array && TensorKit.storagetype(operator) <: storage
        return operator
    end
    return square_J1_to_storage(storage, operator)
end

function _square_J1_build_double_layer_storage(A, operator)
    A = permute(A, (1, 2), (3, 4, 5))
    U_L = @ignore_derivatives _square_J1_unitary_like(
        A, fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1),
    )
    U_D = @ignore_derivatives _square_J1_unitary_like(
        A, fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2),
    )
    U_R = @ignore_derivatives _square_J1_unitary_like(
        A, space(A, 3) ⊗ space(A, 3)', fuse(space(A, 3)' ⊗ space(A, 3)),
    )
    U_U = @ignore_derivatives _square_J1_unitary_like(
        A, space(A, 4) ⊗ space(A, 4)', fuse(space(A, 4)' ⊗ space(A, 4)),
    )

    U_tem = @ignore_derivatives _square_J1_unitary_like(
        A, fuse(space(A, 1) * space(A, 2)), space(A, 1) * space(A, 2),
    )
    vM = U_tem * A
    uM = U_tem'
    @assert norm(uM * vM - A) / norm(A) < 1.0e-12

    uM = permute(uM, (1, 2, 3), ())
    V = space(vM, 1)
    U = @ignore_derivatives _square_J1_unitary_like(A, fuse(V' ⊗ V), V' ⊗ V)
    @tensor double_LD[:] := uM'[-1, -2, 1] * U'[1, -3, -4]
    @tensor double_LD[:] := double_LD[-1, -3, 1, -5] * uM[-2, -4, 1]

    vM = permute(vM, (1, 2, 3, 4), ())
    operator_run = _square_J1_operator_like(A, operator)
    if operator_run isa AbstractVector && isempty(operator_run)
        @tensor double_RU[:] := U[-1, -2, 1] * vM[1, -3, -4, -5]
        @tensor double_RU[:] := vM'[1, -2, -4, 2] * double_RU[-1, 1, -3, -5, 2]
    else
        @tensor double_RU[:] := U[-1, -2, 1] * vM[1, -3, -4, -5]
        @tensor double_RU[:] := vM'[3, -2, -4, 1] *
                                operator_run[2, 1] *
                                double_RU[-1, 3, -3, -5, 2]
    end

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
    return AA_fused, U_L, U_D, U_R, U_U
end

function _square_J1_init_CTM_storage(chi, A, type, CTM_ite_info)
    @ignore_derivatives if CTM_ite_info
        display("initialize CTM")
    end
    Cset = Cset_struc(A, A, A, A)
    Tset = Tset_struc(A, A, A, A)

    if type == "PBC"
        U_L = @ignore_derivatives _square_J1_unitary_like(
            A, fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1),
        )
        U_D = @ignore_derivatives _square_J1_unitary_like(
            A, fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2),
        )
        U_R = @ignore_derivatives _square_J1_unitary_like(
            A, space(A, 3) ⊗ space(A, 3)', fuse(space(A, 3)' ⊗ space(A, 3)),
        )
        U_U = @ignore_derivatives _square_J1_unitary_like(
            A, space(A, 4) ⊗ space(A, 4)', fuse(space(A, 4)' ⊗ space(A, 4)),
        )

        @tensor C1[:] := A'[2, 4, 6, 3, 1] * A[2, 5, 7, 3, 1] *
                          U_D[-1, 4, 5] * U_R[6, 7, -2]
        @tensor C2[:] := A'[4, 6, 3, 2, 1] * A[5, 7, 3, 2, 1] *
                          U_L[-1, 4, 5] * U_D[-2, 6, 7]
        @tensor C3[:] := A'[6, 3, 2, 4, 1] * A[7, 3, 2, 5, 1] *
                          U_U[4, 5, -1] * U_L[-2, 6, 7]
        @tensor C4[:] := A'[2, 3, 6, 4, 1] * A[2, 3, 7, 5, 1] *
                          U_R[6, 7, -1] * U_U[4, 5, -2]

        @tensor T4[:] := A'[2, 3, 5, 7, 1] * A[2, 4, 6, 8, 1] *
                          U_D[-1, 3, 4] * U_R[5, 6, -2] * U_U[7, 8, -3]
        @tensor T1[:] := A'[3, 5, 7, 2, 1] * A[4, 6, 8, 2, 1] *
                          U_L[-1, 3, 4] * U_D[-2, 5, 6] * U_R[7, 8, -3]
        @tensor T2[:] := A'[5, 7, 2, 3, 1] * A[6, 8, 2, 4, 1] *
                          U_U[3, 4, -1] * U_L[-2, 5, 6] * U_D[-3, 7, 8]
        @tensor T3[:] := A'[7, 2, 3, 5, 1] * A[8, 2, 4, 6, 1] *
                          U_R[3, 4, -1] * U_U[5, 6, -2] * U_L[-3, 7, 8]

        Cset = Cset_struc(C1, C2, C3, C4)
        Tset = Tset_struc(T1, T2, T3, T4)
    elseif type != "random"
        throw(ArgumentError("unknown CTM initialization type: $type"))
    end
    return CTM_struc(Cset, Tset)
end

function _square_J1_build_cross_double_layer_open_storage(A_bra, A_ket)
    numind(A_bra) == 5 || throw(ArgumentError("square iPEPS bra tensor must have five legs"))
    numind(A_ket) == 5 || throw(ArgumentError("square iPEPS ket tensor must have five legs"))
    space(A_bra, 5) == space(A_ket, 5) ||
        throw(SpaceMismatch("bra and ket physical spaces differ"))

    U_L = @ignore_derivatives _square_J1_unitary_like(
        A_bra, fuse(space(A_bra, 1)' ⊗ space(A_ket, 1)),
        space(A_bra, 1)' ⊗ space(A_ket, 1),
    )
    U_D = @ignore_derivatives _square_J1_unitary_like(
        A_bra, fuse(space(A_bra, 2)' ⊗ space(A_ket, 2)),
        space(A_bra, 2)' ⊗ space(A_ket, 2),
    )
    U_R = @ignore_derivatives _square_J1_unitary_like(
        A_bra, space(A_bra, 3) ⊗ space(A_ket, 3)',
        fuse(space(A_bra, 3)' ⊗ space(A_ket, 3)),
    )
    U_U = @ignore_derivatives _square_J1_unitary_like(
        A_bra, space(A_bra, 4) ⊗ space(A_ket, 4)',
        fuse(space(A_bra, 4)' ⊗ space(A_ket, 4)),
    )

    A_bra_adj = permute(A_bra', (1, 2, 5), (3, 4))
    U_bra = @ignore_derivatives _square_J1_unitary_like(
        A_bra,
        fuse(space(A_bra_adj, 1) ⊗ space(A_bra_adj, 2) ⊗ space(A_bra_adj, 3)),
        space(A_bra_adj, 1) ⊗ space(A_bra_adj, 2) ⊗ space(A_bra_adj, 3),
    )
    v_bra = U_bra * A_bra_adj
    u_bra = U_bra'

    U_ket = @ignore_derivatives _square_J1_unitary_like(
        A_ket, fuse(space(A_ket, 1) ⊗ space(A_ket, 2)),
        space(A_ket, 1) ⊗ space(A_ket, 2),
    )
    v_ket = U_ket * permute(A_ket, (1, 2), (3, 4, 5))
    u_ket = U_ket'

    u_bra = permute(u_bra, (1, 2, 3, 4), ())
    u_ket = permute(u_ket, (1, 2, 3), ())
    V_bra = space(v_bra, 1)
    V_ket = space(v_ket, 1)
    U_mid = @ignore_derivatives _square_J1_unitary_like(
        A_bra, fuse(V_bra ⊗ V_ket), V_bra ⊗ V_ket,
    )

    @tensor double_LD[:] := u_bra[-1, -2, -3, 1] * U_mid'[1, -4, -5]
    @tensor double_LD[:] := double_LD[-1, -3, -5, 1, -6] * u_ket[-2, -4, 1]
    v_bra = permute(v_bra, (1, 2, 3), ())
    v_ket = permute(v_ket, (1, 2, 3, 4))
    @tensor double_RU[:] := U_mid[-1, -2, 1] * v_ket[1, -3, -4, -5]
    @tensor double_RU[:] := v_bra[1, -2, -4] * double_RU[-1, 1, -3, -5, -6]

    double_LD = permute(double_LD, (1, 2), (3, 4, 5, 6))
    double_LD = U_L * double_LD
    double_LD = permute(double_LD, (2, 3), (1, 4, 5))
    double_LD = U_D * double_LD
    double_LD = permute(double_LD, (2, 1, 3, 4), ())
    double_RU = permute(double_RU, (1, 2, 3, 6), (4, 5))
    double_RU = double_RU * U_U
    @tensor double_RU[:] := double_RU[-1, 1, 2, -4, -3] * U_R[1, 2, -2]

    V_physical_bra = space(A_bra, 5)
    V_physical_ket = space(A_ket, 5)
    V_physical_pair = @ignore_derivatives fuse(V_physical_bra' ⊗ V_physical_ket)
    U_physical = @ignore_derivatives _square_J1_unitary_like(
        A_bra, V_physical_pair, V_physical_bra' ⊗ V_physical_ket,
    )

    # None of the SVD/fusion workspaces below is needed once double_LD,
    # double_RU and U_physical have been formed.  At D=16 retaining them until
    # function return can fill an entire 96-GiB GPU before AA_open is allocated.
    A_bra_adj = nothing
    U_bra = nothing
    U_ket = nothing
    U_mid = nothing
    u_bra = nothing
    u_ket = nothing
    v_bra = nothing
    v_ket = nothing
    U_L = nothing
    U_D = nothing
    U_R = nothing
    U_U = nothing
    square_J1_reclaim_device_memory!()

    # Force a memory-bounded binary contraction order.  The three-tensor
    # expression previously let TensorOperations retain a large transform
    # workspace together with both double-layer factors.
    @tensor LD_physical[:] := double_LD[-1, -2, 1, -3] *
                              U_physical[-4, 1, -5]
    double_LD = nothing
    square_J1_reclaim_device_memory!()
    @tensor AA_open[:] := LD_physical[-1, -2, 1, -5, 2] *
                          double_RU[1, -3, -4, 2]
    LD_physical = nothing
    double_RU = nothing
    square_J1_reclaim_device_memory!()
    return AA_open, U_physical'
end

if isdefined(@__MODULE__, :CUDA)
    function build_double_layer(
        A::TensorKit.TensorMap{T,S,N1,N2,Storage}, operator,
    ) where {T,S,N1,N2,Storage<:CUDA.CuArray}
        return _square_J1_build_double_layer_storage(A, operator)
    end

    function init_CTM(
        chi, A::TensorKit.TensorMap{T,S,N1,N2,Storage}, type, CTM_ite_info,
    ) where {T,S,N1,N2,Storage<:CUDA.CuArray}
        return _square_J1_init_CTM_storage(chi, A, type, CTM_ite_info)
    end

    function build_square_cross_double_layer_open(
        A_bra::TensorKit.TensorMap{TB,SB,NB1,NB2,StorageB},
        A_ket::TensorKit.TensorMap{TK,SK,NK1,NK2,StorageK},
    ) where {
        TB,SB<:TensorKit.ElementarySpace,NB1,NB2,
        StorageB<:CUDA.CuArray{TB,1},
        TK,SK<:TensorKit.ElementarySpace,NK1,NK2,
        StorageK<:CUDA.CuArray{TK,1},
    }
        return _square_J1_build_cross_double_layer_open_storage(A_bra, A_ket)
    end
end
