using TensorKit
using LinearAlgebra: I
using Zygote: @ignore_derivatives

if !isdefined(@__MODULE__, :chiral_pair_fuse_unitary)
    include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "bosonic", "square", "square_chiral_pair_model.jl"))
end

"""
    chiral_pair_Ty1_mpo()

Return the local physical MPO tensor for translating one copy of the chiral-pair
physical leg by one site along y.

Leg convention:

    W[p_bra, p_ket, m_in, m_out]

where `p = fuse(a, b)` is the fused d=4 physical leg, `a` is the copy being
translated, `b` is the onsite copy, `m_in = a_{j-1}`, and `m_out = a_j`.

The tensor implements

    delta(a_bra, m_in) * delta(b_bra, b_ket) * delta(m_out, a_ket)

in the unfused copy basis, with `chiral_pair_fuse_unitary()` converting between
fused and copy bases.
"""
function chiral_pair_Ty1_mpo()
    U_pair = @ignore_derivatives chiral_pair_fuse_unitary()
    Va = @ignore_derivatives space(U_pair, 2)
    Vb = @ignore_derivatives space(U_pair, 3)
    @assert Va == Vb "The two chiral-pair copy spaces should be identical."

    Id_a = unitary(Va, Va) * (1 + 0im)

    @tensor W[-1, -2, -3, -4] :=
        U_pair'[1, 2, -1] *
        Id_a[1, -3] *
        Id_a[-4, 3] *
        U_pair[-2, 3, 2]

    return W
end

"""
    chiral_pair_Tym1_mpo()

Return the local physical MPO tensor for translating the A copy by one site in
the opposite y direction.

Leg convention:

    W[p_bra, p_ket, m_in, m_out]

Here `m_in = A_{j+1}` and `m_out = A_j`.  The B copy is still contracted
onsite directly between bra and ket.
"""
function chiral_pair_Tym1_mpo()
    Wp = chiral_pair_Ty1_mpo()
    return permute(Wp, (1, 2, 4, 3))
end

function compose_chiral_pair_shift_mpo(Wa, Wb)
    @assert space(Wa, 2) == space(Wb, 1)' "The intermediate physical spaces of the two MPO tensors are not dual."

    # Wa[p_bra, p_mid, a_in, a_out] * Wb[p_mid, p_ket, b_in, b_out].
    # The composed MPO has two incoming and two outgoing auxiliary legs before
    # fusion: W_raw[p_bra, p_ket, a_in, b_in, a_out, b_out].
    @tensor W_raw[-1, -2, -3, -4, -5, -6] :=
        Wa[-1, 1, -3, -5] *
        Wb[1, -2, -4, -6]

    U_in = @ignore_derivatives unitary(
        fuse(space(W_raw, 3) * space(W_raw, 4)),
        space(W_raw, 3) * space(W_raw, 4),
    ) * (1 + 0im)


    @tensor W[-1, -2, -3, -4] :=
        W_raw[-1, -2, 1, 2, 3, 4] *
        U_in[-3, 1, 2] *
        U_in'[3, 4,-4]

    return W
end

"""
    chiral_pair_Tym2_mpo()

Return the local physical MPO tensor for translating the A copy by two sites in
the negative y direction.  It is built as `Tym1 * Tym1`, with the two incoming
auxiliary legs fused into one leg and the two outgoing auxiliary legs fused into
one leg.

Leg convention:

    W[p_bra, p_ket, m_in, m_out]

where `m_in` and `m_out` are fused two-copy auxiliary spaces.
"""
function chiral_pair_Tym2_mpo()
    Wm1 = chiral_pair_Tym1_mpo()
    return compose_chiral_pair_shift_mpo(Wm1, Wm1)
end

"""
    apply_chiral_pair_shift_mpo_to_peps(A, W; return_unitary=false)

Apply a local chiral-pair physical shift MPO to the physical leg of a PEPS
tensor and fuse the MPO auxiliary legs into the vertical virtual legs.

Input conventions:

    A[L, D, R, U, p_ket]
    W[p_bra, p_ket, m_in, m_out]

Output convention:

    A_shift[L, Dm, R, Um, p_bra]

where

    Dm = fuse(D, m_in)
    Um is fused with the adjoint of the same vertical unitary.

The same `U_vertical` is used on the down leg and `U_vertical'` on the up leg,
matching the convention used in the older `build_double_layer_Ty1` code.
"""
function apply_mpo_to_peps(A, W; return_unitary::Bool=false)
    @assert space(A, 5) == space(W, 2)' "The ket physical space of W does not match the PEPS physical leg."

    U_vertical = @ignore_derivatives unitary(
        fuse(space(A, 2) * space(W, 3)),
        space(A, 2) * space(W, 3),
    ) * (1 + 0im)

    @tensor A_shift[-1, -2, -3, -4, -5] :=
        A[-1, 1, -3, 3, 5] *
        W[-5, 5, 2, 4] *
        U_vertical[-2, 1, 2] *
        U_vertical'[3, 4, -4]

    return return_unitary ? (A_shift, U_vertical) : A_shift
end

function print_chiral_pair_Ty1_mpo_spaces()
    W = chiral_pair_Ty1_mpo()
    println("chiral_pair_Ty1_mpo space:")
    println(space(W))
    println("leg convention: W[p_bra, p_ket, m_in, m_out]")
    println("  p_bra = " * string(space(W, 1)))
    println("  p_ket = " * string(space(W, 2)))
    println("  m_in  = " * string(space(W, 3)))
    println("  m_out = " * string(space(W, 4)))
    flush(stdout)
    return W
end

function print_chiral_pair_Tym1_mpo_spaces()
    W = chiral_pair_Tym1_mpo()
    println("chiral_pair_Tym1_mpo space:")
    println(space(W))
    println("leg convention: W[p_bra, p_ket, m_in, m_out]")
    println("  p_bra = " * string(space(W, 1)))
    println("  p_ket = " * string(space(W, 2)))
    println("  m_in  = " * string(space(W, 3)))
    println("  m_out = " * string(space(W, 4)))
    flush(stdout)
    return W
end

function print_chiral_pair_Tym2_mpo_spaces()
    W = chiral_pair_Tym2_mpo()
    println("chiral_pair_Tym2_mpo space:")
    println(space(W))
    println("leg convention: W[p_bra, p_ket, m_in, m_out]")
    println("  p_bra = " * string(space(W, 1)))
    println("  p_ket = " * string(space(W, 2)))
    println("  m_in  = " * string(space(W, 3)))
    println("  m_out = " * string(space(W, 4)))
    flush(stdout)
    return W
end



function build_double_layer_NoSwap(Ap,A)
    #display(space(A))

    Ap=permute(Ap,((1,2,),(3,4,5)))
    A=permute(A,((1,2,),(3,4,5)));
    
    # U_L=unitary(fuse(space(A, 1)' ⊗ space(A, 1)), space(A, 1)' ⊗ space(A, 1));
    # U_D=unitary(fuse(space(A, 2)' ⊗ space(A, 2)), space(A, 2)' ⊗ space(A, 2));
    # U_R=inv(U_L);
    # U_U=inv(U_D);

    # U_Lp=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    # U_Dp=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    # U_Rp=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    # U_Up=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # println(norm(U_R-U_Rp)/norm(U_R))
    # println(norm(U_L-U_Lp)/norm(U_L))
    # println(norm(U_D-U_Dp)/norm(U_D))
    # println(norm(U_U-U_Up)/norm(U_U))

    U_L=unitary(fuse(space(Ap, 1) ⊗ space(A, 1)), space(Ap, 1) ⊗ space(A, 1));
    U_D=unitary(fuse(space(Ap, 2) ⊗ space(A, 2)), space(Ap, 2) ⊗ space(A, 2));
    U_R=unitary(space(Ap, 3)' ⊗ space(A, 3)', fuse(space(Ap, 3)' ⊗ space(A, 3)'));
    U_U=unitary(space(Ap, 4)' ⊗ space(A, 4)', fuse(space(Ap, 4)' ⊗ space(A, 4)'));

    # display(space(U_L))
    # display(space(U_D))
    # display(space(U_R))
    # display(space(U_D))

    uMp,sMp,vMp=tsvd(Ap);
    uMp=uMp*sMp;
    uM,sM,vM=tsvd(A);
    uM=uM*sM;

    uMp=permute(uMp,(1,2,3,),())
    uM=permute(uM,(1,2,3,),())
    Vp=space(uMp,3);
    V=space(vM,1);
    U=unitary(fuse(Vp' ⊗ V), Vp' ⊗ V);

    @tensor double_LD[:]:=uMp[-1,-2,1]*U'[1,-3,-4];
    @tensor double_LD[:]:=double_LD[-1,-3,1,-5]*uM[-2,-4,1];

    vMp=permute(vMp,(1,2,3,4,),());
    vM=permute(vM,(1,2,3,4,),());

    @tensor double_RU[:]:=U[-1,-2,1]*vM[1,-3,-4,-5];
    @tensor double_RU[:]:=vMp[1,-2,-4,2]*double_RU[-1,1,-3,-5,2];

    #display(space(double_RU))

    double_LD=permute(double_LD,(1,2,),(3,4,5,));
    double_LD=U_L*double_LD;
    double_LD=permute(double_LD,(2,3,),(1,4,));
    double_LD=U_D*double_LD;
    double_LD=permute(double_LD,(2,1,),(3,));
    #display(space(double_LD))
    double_RU=permute(double_RU,(1,4,5,),(2,3,));
    double_RU=double_RU*U_R;
    double_RU=permute(double_RU,(1,4,),(2,3,));
    double_RU=double_RU*U_U;
    double_LD=permute(double_LD,(1,2,),(3,));
    double_RU=permute(double_RU,(1,),(2,3,));
    AA_fused=double_LD*double_RU;


    ##########################

    AA_fused=permute(AA_fused,(1,2,3,4,));


    return AA_fused, U_L,U_D,U_R,U_U
end
