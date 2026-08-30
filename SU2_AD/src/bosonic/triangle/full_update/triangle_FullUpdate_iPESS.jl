# Environment-aware bosonic full update for the simplex-centred (up) triangles
# of the triangular-lattice iPESS ansatz. This file intentionally contains no
# fermionic parity convention.

if !isdefined(@__MODULE__, :bosonic_sweep_optimizations)
    include("bosonic_Full_Update_lib.jl")
end

# The repository historically used `truncdim`; upstream TensorKit renamed it
# to `truncrank` and does not provide the fork-specific `truncmultiplet`.
# Keep this compatibility local to the bosonic full-update entry point.
if !isdefined(@__MODULE__, :truncdim) && isdefined(TensorKit, :truncrank)
    function truncdim(howmany::Integer; multiplet_tol=nothing, by=abs, rev::Bool=true)
        return TensorKit.truncrank(howmany; by=by, rev=rev)
    end
end

function bosonic_parity_gate(A, p::Integer)
    V = space(A, p)
    return unitary(V, V)
end
function bosonic_get_overlap_env(env_top,env_bot,triangle_top,triangle_bot)
    #env_bot:   (env_ind),  (2,3,1)
    #env_top:   (2,3,1), (env_ind)
    #triangle_bot: (2, 1), (d123, 3)
    #triangle_top: (d123, 3), (2, 1)
    @tensor Bot[:]:=env_bot[-1,2,3,1]*triangle_bot[2,1,-2,3];
    @tensor Top[:]:=env_top[2,3,1,-1]*triangle_top[-2,3,2,1];
    ov=@tensor Bot[1,2]*Top[1,2];
    return ov
end

function bosonic_build_double_layer_Tm(Ap,A, with_physical)
    if ~with_physical #no physical leg
        @assert (length(codomain(A))==2)&(length(domain(A))==1)
        @assert (length(codomain(Ap))==1)&(length(domain(Ap))==2)
        #Treat (LU,M) as (LU,D)
        #Treat (M',L'U') as (D',L'U')
        # println(space(Ap))
        # println(space(A))

        gate=@ignore_derivatives bosonic_sign_gate(Ap,2,3); #gate L'U'
        @tensor Ap[:]:=Ap[-1,1,2]*gate[-2,-3,1,2];

        gate=@ignore_derivatives bosonic_parity_gate(Ap,1); #gate D'
        @tensor Ap[:]:=Ap[1,-2,-3]*gate[-1,1];
        gate=@ignore_derivatives bosonic_parity_gate(Ap,3); #gate U'
        @tensor Ap[:]:=Ap[-1,-2,1]*gate[-3,1];


        A=permute(A,(1,2,),(3,));
        Ap=permute(Ap,(1,),(2,3,));


        U_L=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 1)), space(Ap, 2) ⊗ space(A, 1));
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 1) ⊗ space(A, 3)), space(Ap, 1) ⊗ space(A, 3));
        U_U=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 2)', fuse(space(Ap, 3)' ⊗ space(A, 2)'));

        @tensor AA_fused[:]:=Ap[5,1,3]*A[2,4,6]*U_L[-1,1,2]*U_D[-2,5,6]*U_U[3,4,-3];

    else #with 3 physical legs grouped as one leg
        @assert (length(codomain(A))==2)&(length(domain(A))==2)
        @assert (length(codomain(Ap))==2)&(length(domain(Ap))==2)
        #Treat (LU,dM) as (LU,dD)
        #Treat (d'M',L'U') as (d'D',L'U')
        # println(space(Ap))
        # println(space(A))

        gate=@ignore_derivatives bosonic_sign_gate(Ap,3,4); #gate L'U'
        @tensor Ap[:]:=Ap[-1,-2,1,2]*gate[-3,-4,1,2];

        gate=@ignore_derivatives bosonic_parity_gate(Ap,2); #gate D'
        @tensor Ap[:]:=Ap[-1,1,-3,-4]*gate[-2,1];
        gate=@ignore_derivatives bosonic_parity_gate(Ap,4); #gate U'
        @tensor Ap[:]:=Ap[-1,-2,-3,1]*gate[-4,1];


        A=permute(A,(1,2,),(3,4,));
        Ap=permute(Ap,(1,2,),(3,4,));


        U_L=@ignore_derivatives unitary(fuse(space(Ap, 3) ⊗ space(A, 1)), space(Ap, 3) ⊗ space(A, 1));
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 4)), space(Ap, 2) ⊗ space(A, 4));
        U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 2)', fuse(space(Ap, 4)' ⊗ space(A, 2)'));

        @tensor AA_fused[:]:=Ap[3,6,1,4]*A[2,5,3,7]*U_L[-1,1,2]*U_D[-2,6,7]*U_U[4,5,-3];
    end


    P_odd_Lp,_=@ignore_derivatives projector_parity(space(U_L',1));
    P_odd_Up,_=@ignore_derivatives projector_parity(space(U_U',2));
    P_odd_U,_=@ignore_derivatives projector_parity(space(U_U',3));

    @tensor isom_Lp[:]:=U_L[-1,4,3]*P_odd_Lp'[4,1]*P_odd_Lp[1,2]*U_L'[2,3,-2];
    @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2];
    @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6];
    @tensor AA_Lp_U[:]:=AA_fused[1,-2,4]*isom_Lp[-1,1]*isom_U[-3,4];
    AA_fused=AA_fused-2*AA_Lp_U;
    @tensor AA_Up_U[:]:=AA_fused[-1,-2,4]*isom_Up_U[-3,4];
    AA_fused=AA_fused-2*AA_Up_U;



    P_odd_Dp,_=@ignore_derivatives projector_parity(space(U_D',1));
    P_odd_D,_=@ignore_derivatives projector_parity(space(U_D',2));
    @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2];
    @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2];
    @tensor AA_Dp_D[:]:=AA_fused[-1,2,-3]*isom_Dp_D[-2,2];
    AA_fused=AA_fused-2*AA_Dp_D;


    #double layer order: L M U = L D U
    return AA_fused, U_L,U_D,U_U
end

function bosonic_build_double_layer_noswap_Tm(Ap,A, with_physical)
    if ~with_physical #no physical leg
        @assert (length(codomain(A))==2)&(length(domain(A))==1)
        @assert (length(codomain(Ap))==1)&(length(domain(Ap))==2)
        #Treat (LU,M) as (LU,D)
        #Treat (M',L'U') as (D',L'U')
        # println(space(Ap))
        # println(space(A))

        # gate=@ignore_derivatives bosonic_sign_gate(Ap,2,3); #gate L'U'
        # @tensor Ap[:]:=Ap[-1,1,2]*gate[-2,-3,1,2];

        # gate=@ignore_derivatives bosonic_parity_gate(Ap,1); #gate D'
        # @tensor Ap[:]:=Ap[1,-2,-3]*gate[-1,1];
        # gate=@ignore_derivatives bosonic_parity_gate(Ap,3); #gate U'
        # @tensor Ap[:]:=Ap[-1,-2,1]*gate[-3,1];


        A=permute(A,(1,2,),(3,));
        Ap=permute(Ap,(1,),(2,3,));


        U_L=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 1)), space(Ap, 2) ⊗ space(A, 1));
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 1) ⊗ space(A, 3)), space(Ap, 1) ⊗ space(A, 3));
        U_U=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 2)', fuse(space(Ap, 3)' ⊗ space(A, 2)'));

        @tensor AA_fused[:]:=Ap[5,1,3]*A[2,4,6]*U_L[-1,1,2]*U_D[-2,5,6]*U_U[3,4,-3];

    else #with 3 physical legs grouped as one leg
        @assert (length(codomain(A))==2)&(length(domain(A))==2)
        @assert (length(codomain(Ap))==2)&(length(domain(Ap))==2)
        #Treat (LU,dM) as (LU,dD)
        #Treat (d'M',L'U') as (d'D',L'U')
        # println(space(Ap))
        # println(space(A))

        # gate=@ignore_derivatives bosonic_sign_gate(Ap,3,4); #gate L'U'
        # @tensor Ap[:]:=Ap[-1,-2,1,2]*gate[-3,-4,1,2];

        # gate=@ignore_derivatives bosonic_parity_gate(Ap,2); #gate D'
        # @tensor Ap[:]:=Ap[-1,1,-3,-4]*gate[-2,1];
        # gate=@ignore_derivatives bosonic_parity_gate(Ap,4); #gate U'
        # @tensor Ap[:]:=Ap[-1,-2,-3,1]*gate[-4,1];


        A=permute(A,(1,2,),(3,4,));
        Ap=permute(Ap,(1,2,),(3,4,));


        U_L=@ignore_derivatives unitary(fuse(space(Ap, 3) ⊗ space(A, 1)), space(Ap, 3) ⊗ space(A, 1));
        U_D=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 4)), space(Ap, 2) ⊗ space(A, 4));
        U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 2)', fuse(space(Ap, 4)' ⊗ space(A, 2)'));

        @tensor AA_fused[:]:=Ap[3,6,1,4]*A[2,5,3,7]*U_L[-1,1,2]*U_D[-2,6,7]*U_U[4,5,-3];
    end


    # P_odd_Lp,_=@ignore_derivatives projector_parity(space(U_L',1));
    # P_odd_Up,_=@ignore_derivatives projector_parity(space(U_U',2));
    # P_odd_U,_=@ignore_derivatives projector_parity(space(U_U',3));

    # @tensor isom_Lp[:]:=U_L[-1,4,3]*P_odd_Lp'[4,1]*P_odd_Lp[1,2]*U_L'[2,3,-2];
    # @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2];
    # @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6];
    # @tensor AA_Lp_U[:]:=AA_fused[1,-2,4]*isom_Lp[-1,1]*isom_U[-3,4];
    # AA_fused=AA_fused-2*AA_Lp_U;
    # @tensor AA_Up_U[:]:=AA_fused[-1,-2,4]*isom_Up_U[-3,4];
    # AA_fused=AA_fused-2*AA_Up_U;



    # P_odd_Dp,_=@ignore_derivatives projector_parity(space(U_D',1));
    # P_odd_D,_=@ignore_derivatives projector_parity(space(U_D',2));
    # @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2];
    # @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2];
    # @tensor AA_Dp_D[:]:=AA_fused[-1,2,-3]*isom_Dp_D[-2,2];
    # AA_fused=AA_fused-2*AA_Dp_D;


    #double layer order: L M U = L D U
    return AA_fused, U_L,U_D,U_U
end

function bosonic_build_double_layer_noswap_Bm(Ap, A, with_physical)
    if with_physical
        @assert (length(codomain(A)) == 1) & (length(domain(A)) == 3)
        @assert (length(codomain(Ap)) == 3) & (length(domain(Ap)) == 1)

        A = permute(A, (1,), (2, 3, 4))
        Ap = permute(Ap, (1, 2, 3), (4,))

        U_D = @ignore_derivatives unitary(
            fuse(space(Ap, 3) ⊗ space(A, 4)),
            space(Ap, 3) ⊗ space(A, 4),
        )
        U_R = @ignore_derivatives unitary(
            space(Ap, 2)' ⊗ space(A, 3)',
            fuse(space(Ap, 2)' ⊗ space(A, 3)'),
        )
        U_U = @ignore_derivatives unitary(
            space(Ap, 4)' ⊗ space(A, 1)',
            fuse(space(Ap, 4)' ⊗ space(A, 1)'),
        )

        @tensor AA_fused[:] := Ap[3, 1, 6, 4] * A[5, 3, 2, 7] *
                                U_U[4, 5, -3] * U_R[1, 2, -2] * U_D[-1, 6, 7]
    else
        @assert (length(codomain(A)) == 1) & (length(domain(A)) == 2)
        @assert (length(codomain(Ap)) == 2) & (length(domain(Ap)) == 1)

        A = permute(A, (1,), (2, 3))
        Ap = permute(Ap, (1, 2), (3,))

        U_D = @ignore_derivatives unitary(
            fuse(space(Ap, 2) ⊗ space(A, 3)),
            space(Ap, 2) ⊗ space(A, 3),
        )
        U_R = @ignore_derivatives unitary(
            space(Ap, 1)' ⊗ space(A, 2)',
            fuse(space(Ap, 1)' ⊗ space(A, 2)'),
        )
        U_U = @ignore_derivatives unitary(
            space(Ap, 3)' ⊗ space(A, 1)',
            fuse(space(Ap, 3)' ⊗ space(A, 1)'),
        )

        @tensor AA_fused[:] := Ap[1, 6, 4] * A[5, 2, 7] *
                                U_U[4, 5, -3] * U_R[1, 2, -2] * U_D[-1, 6, 7]
    end

    # Double-layer order: D, R, M (the last leg is the simplex/up leg).
    return AA_fused, U_D, U_R, U_U
end

function bosonic_build_double_layer_Bm(Ap,A, with_physical)
    if with_physical #with one physical leg
        @assert (length(codomain(A))==1)&(length(domain(A))==3)
        @assert (length(codomain(Ap))==3)&(length(domain(Ap))==1)
        #treat (M,dRD) as (U,dRD)
        #treat (d'R'D',M') as (d'R'D',U')
        # println(space(Ap))
        # println(space(A))


        gate=@ignore_derivatives bosonic_sign_gate(Ap,2,3); #gate D'R'
        @tensor Ap[:]:=Ap[-1,1,2,-4]*gate[-2,-3,1,2];
        gate=@ignore_derivatives bosonic_parity_gate(Ap,4); #gate U'
        @tensor Ap[:]:=Ap[-1,-2,-3,1]*gate[-4,1];
        gate=@ignore_derivatives bosonic_parity_gate(Ap,3);  #gate D'
        @tensor Ap[:]:=Ap[-1,-2,1,-4]*gate[-3,1];

        A=permute(A,(1,),(2,3,4,));
        Ap=permute(Ap,(1,2,3,),(4,));


        U_D=@ignore_derivatives unitary(fuse(space(Ap, 3) ⊗ space(A, 4)), space(Ap, 3) ⊗ space(A, 4));
        U_R=@ignore_derivatives unitary(space(Ap, 2)' ⊗ space(A, 3)', fuse(space(Ap, 2)' ⊗ space(A, 3)'));
        U_U=@ignore_derivatives unitary(space(Ap, 4)' ⊗ space(A, 1)', fuse(space(Ap, 4)' ⊗ space(A, 1)'));


        @tensor AA_fused[:]:=Ap[3,1,6,4]*A[5,3,2,7]*U_U[4,5,-3]*U_R[1,2,-2]*U_D[-1,6,7];
    else
        @assert (length(codomain(A))==1)&(length(domain(A))==2)
        @assert (length(codomain(Ap))==2)&(length(domain(Ap))==1)
        #treat (M,RD) as (U,RD)
        #treat (R'D',M') as (R'D',U')

        gate=@ignore_derivatives bosonic_sign_gate(Ap,1,2); #gate D'R'
        @tensor Ap[:]:=Ap[1,2,-3]*gate[-1,-2,1,2];
        gate=@ignore_derivatives bosonic_parity_gate(Ap,3); #gate U'
        @tensor Ap[:]:=Ap[-1,-2,1]*gate[-3,1];
        gate=@ignore_derivatives bosonic_parity_gate(Ap,2);  #gate D'
        @tensor Ap[:]:=Ap[-1,1,-3]*gate[-2,1];

        A=permute(A,(1,),(2,3,));
        Ap=permute(Ap,(1,2,),(3,));


        U_D=@ignore_derivatives unitary(fuse(space(Ap, 2) ⊗ space(A, 3)), space(Ap, 2) ⊗ space(A, 3));
        U_R=@ignore_derivatives unitary(space(Ap, 1)' ⊗ space(A, 2)', fuse(space(Ap, 1)' ⊗ space(A, 2)'));
        U_U=@ignore_derivatives unitary(space(Ap, 3)' ⊗ space(A, 1)', fuse(space(Ap, 3)' ⊗ space(A, 1)'));


        @tensor AA_fused[:]:=Ap[1,6,4]*A[5,2,7]*U_U[4,5,-3]*U_R[1,2,-2]*U_D[-1,6,7];
    end


    ##########################



    P_odd_Up,_=@ignore_derivatives projector_parity(space(U_U',2));
    P_odd_U,_=@ignore_derivatives projector_parity(space(U_U',3));


    @tensor isom_U[:]:=U_U[3,4,-1]*P_odd_U'[4,1]*P_odd_U[1,2]*U_U'[-2,3,2];
    @tensor isom_Up_U[:]:=U_U[3,4,-1]*P_odd_Up'[3,1]*P_odd_Up[1,5]*P_odd_U'[4,2]*P_odd_U[2,6]*U_U'[-2,5,6];
    @tensor AA_Up_U[:]:=AA_fused[-1,-2,4]*isom_Up_U[-3,4];
    AA_fused=AA_fused-2*AA_Up_U;



    P_odd_Dp,_=@ignore_derivatives projector_parity(space(U_D',1));
    P_odd_D,_=@ignore_derivatives projector_parity(space(U_D',2));
    P_odd_R,_=@ignore_derivatives projector_parity(space(U_R',3));
    @tensor isom_Dp[:]:=U_D[-1,4,3]*P_odd_Dp'[4,1]*P_odd_Dp[1,2]*U_D'[2,3,-2];
    @tensor isom_R[:]:=U_R[3,4,-1]*P_odd_R'[4,1]*P_odd_R[1,2]*U_R'[-2,3,2];
    @tensor isom_Dp_D[:]:=U_D[-1,3,4]*P_odd_Dp'[3,1]*P_odd_Dp[1,5]*P_odd_D'[4,2]*P_odd_D[2,6]*U_D'[5,6,-2];
    @tensor AA_Dp_D[:]:=AA_fused[2,-2,-3]*isom_Dp_D[-1,2];
    AA_fused=AA_fused-2*AA_Dp_D;
    @tensor AA_Dp_R[:]:=AA_fused[2,3,-3]*isom_Dp[-1,2]*isom_R[-2,3];
    AA_fused=AA_fused-2*AA_Dp_R;


    #double layer order: D R M = D R U
    return AA_fused, U_D,U_R,U_U
end



function bosonic_contract_triangle_env(
    CTM,
    T_double_LU,
    T_double_RU,
    T_double_LD,
    B_double_LU,
    B_double_RU,
    B_double_LD,
    B_double_RD,
    cx,
    cy,
    Lx,
    Ly,
)
    #leading memory cost:
    #chi^2*D^4*d^4
    #D^6*d^6
    Cset=CTM.Cset;
    Tset=CTM.Tset;

    @tensor MM_LU[:]:=Cset[mod1(cx-1,Lx)][mod1(cy-1,Ly)].C1[1,2]*Tset[mod1(cx,Lx)][mod1(cy-1,Ly)].T1[2,3,-3]*Tset[mod1(cx-1,Lx)][mod1(cy,Ly)].T4[-1,4,1]*T_double_LU[4,5,3]*B_double_LU[-2,-4,5];
    @tensor MM_RU[:]:=Tset[mod1(cx+1,Lx)][mod1(cy-1,Ly)].T1[-1,3,1]* Cset[mod1(cx+2,Lx)][mod1(cy-1,Ly)].C2[1,2]* T_double_RU[-2,5,3]*B_double_RU[-4,4,5]*Tset[mod1(cx+2,Lx)][mod1(cy,Ly)].T2[2,4,-3];

    @tensor MM_LD[:]:=Tset[mod1(cx-1,Lx)][mod1(cy+1,Ly)].T4[1,3,-1]*T_double_LD[3,5,-2]*B_double_LD[4,-4,5]*Cset[mod1(cx-1,Lx)][mod1(cy+2,Ly)].C4[2,1]*Tset[mod1(cx,Lx)][mod1(cy+2,Ly)].T3[-3,4,2];
    @tensor MM_RD[:]:=Tset[mod1(cx+2,Lx)][mod1(cy+1,Ly)].T2[-4,-3,2]*Tset[mod1(cx+1,Lx)][mod1(cy+2,Ly)].T3[1,-2,-1]*Cset[mod1(cx+2,Lx)][mod1(cy+2,Ly)].C3[2,1];
    @tensor MM_RD[:]:=MM_RD[-1,1,2,-3]*B_double_RD[1,2,-2];


    @tensor LD_LU_RU[:]:=MM_LD[1,2,-1,-2]*MM_LU[1,2,3,4]*MM_RU[3,4,-3,-4];
    @tensor BigTriangle[:]:= LD_LU_RU[1,-1,2,-3]*MM_RD[1,-2,2]; # L M U = L D U
    return BigTriangle
end

function bosonic_build_triangle_from_4tensors(T,B1_keep,B2_keep,B3_keep)
    @tensor B2_T[:]:=B2_keep[-1,-2,1]*T[1,-3,-4];     #(new2, d2, R2),  (R2, D1, M3) => (new2, d2, D1, M3)
    B2_T=bosonic_permute_neighbour_ind(B2_T,2,3,4);#(new2, D1, d2, M3)
    B2_T=bosonic_permute_neighbour_ind(B2_T,1,2,4);#(D1, new2, d2, M3)
    @tensor B1_B2_T[:]:=B1_keep[-1,-2,1]*B2_T[1,-3,-4,-5];#(new1, d1,  D1), (D1, new2, d2, M3) => (new1, d1, new2, d2, M3)

    @tensor B1_B2_T_B3[:]:=B1_B2_T[-1,-2,-3,-4,1]*B3_keep[1,-5,-6];#(new1, d1, new2, d2, M3), (M3, d3, new3) => (new1, d1, new2, d2, d3, new3)
    B1_B2_T_B3=bosonic_permute_neighbour_ind(B1_B2_T_B3,2,3,6);# new1, new2, d1, d2, d3, new3

    bosonic_fu_Up=unitary(fuse(space(B1_B2_T_B3,3)*space(B1_B2_T_B3,4)*space(B1_B2_T_B3,5)), space(B1_B2_T_B3,3)*space(B1_B2_T_B3,4)*space(B1_B2_T_B3,5));
    global bosonic_fu_Up
    @tensor B1_B2_T_B3[:]:=B1_B2_T_B3[-1,-2,1,2,3,-4]*bosonic_fu_Up[-3,1,2,3];# new1, new2, d123, new3

    B1_B2_T_B3=bosonic_permute_neighbour_ind(B1_B2_T_B3,1,2,4);# new2, new1, d123, new3
    B1_B2_T_B3=permute(B1_B2_T_B3,(1,2,),(3,4,));# (new2, new1), (d123, new3)

    #########################################

    # big_T_compressed=bosonic_permute_neighbour_ind(B_new,1,2,3);#(D1_new, R2_new, M3_new)
    # @tensor big_T_compressed[:]:=T1_new[-1,-2,1]*big_T_compressed[1,-3,-4];#(M1_R1, d1, R2_new, M3_new)
    # big_T_compressed=bosonic_permute_neighbour_ind(big_T_compressed,2,3,4);#(M1_R1,R2_new, d1,  M3_new)
    # big_T_compressed=bosonic_permute_neighbour_ind(big_T_compressed,1,2,4);#(R2_new, M1_R1, d1,  M3_new)
    # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,-3,1]*T3_new[1,-4,-5];#(R2_new, M1_R1, d1,  d3, R3_D3)
    # @tensor big_T_compressed[:]:=T2_new[-1,-2,1]*big_T_compressed[1,-3,-4,-5,-6];#(M2_D2, d2,  M1_R1, d1,  d3, R3_D3)

    # big_T_compressed=bosonic_permute_neighbour_ind(big_T_compressed,2,3,6);#(M2_D2,  M1_R1, d2, d1,  d3, R3_D3)
    # big_T_compressed=bosonic_permute_neighbour_ind(big_T_compressed,3,4,6);#(M2_D2,  M1_R1, d1, d2,  d3, R3_D3)
    # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,1,2,3,-4]*bosonic_fu_Up[-3,1,2,3];#(new2, new1,  d123, new3)
    # big_T_compressed=permute(big_T_compressed,(1,2,),(3,4,))#(new2, new1), (d123, new3)


    return B1_B2_T_B3
end

function bosonic_tsvd_truncate(tensor, left_inds, right_inds, D_max)
    return tsvd(tensor, left_inds, right_inds; trunc=truncdim(D_max))
end

function bosonic_truncation_direct(big_T,D_max, trun_order, trun_tol)
    #big_T: (new2, new1), (d123, new3)
    global bosonic_fu_Up
    @tensor big_T[:]:=big_T[-1,-2,1,-6]*bosonic_fu_Up'[-3,-4,-5,1];# new2, new1, d1,d2,d3, new3
    #big_T=big_T/norm(big_T);
    if trun_order=="simultaneous"


        big_T=bosonic_permute_neighbour_ind(big_T,1,2,6);# new1, new2, d1,d2,d3, new3
        big_T=bosonic_permute_neighbour_ind(big_T,2,3,6);# new1, d1, new2,d2,d3, new3

        U1,S1,V1=bosonic_tsvd_truncate(big_T,(1,2,),(3,4,5,6,),D_max);#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
        U3,S3,V3=bosonic_tsvd_truncate(big_T,(1,2,3,4,),(5,6,),D_max);#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)


        big_T=bosonic_permute_neighbour_ind(big_T,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3
        big_T=bosonic_permute_neighbour_ind(big_T,1,2,6);# M2_D2, M1_R1, d1, d2, d3, R3_D3
        big_T=bosonic_permute_neighbour_ind(big_T,3,4,6);# M2_D2, M1_R1, d2, d1, d3, R3_D3
        big_T=bosonic_permute_neighbour_ind(big_T,2,3,6);# M2_D2, d2, M1_R1, d1, d3, R3_D3
        # T1_T2_B_T3=T1_T2_B_T3/norm(T1_T2_B_T3);
        U2,S2,V2=bosonic_tsvd_truncate(big_T,(1,2,),(3,4,5,6,),D_max);#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)

        # λ_2_new=permute(S2,(2,),(1,));

        # @tensor T2_new[:]:=T2_res[-1,1]*U2[1,-2,-3];#(M2_D2, d2, R2_new)
        # T1_B_T3=S2*V2;#(R2_new, M1_R1, d1, d3, R3_D3)
        # @tensor T1_B[:]:=T1_B_T3[-1,-2,-3,1,2]*V3'[1,2,-4];#(R2_new, M1_R1, d1, M3_new)
        # @tensor T3_new[:]:=V3[-1,-2,1]*T3_res[1,-3];#(M3_new, d3, R3_D3)
        # λ_3_new=S3;

        # T1_B=bosonic_permute_neighbour_ind(T1_B,1,2,4);#(M1_R1, R2_new, d1, M3_new)
        # T1_B=bosonic_permute_neighbour_ind(T1_B,2,3,4);#(M1_R1, d1, R2_new, M3_new)
        # @tensor B_new[:]:=U1'[-1,1,2]*T1_B[1,2,-2,-3];#(D1_new, R2_new, M3_new)
        # @tensor T1_new[:]:=T1_res[-1,1]*U1[1,-2,-3];#(M1_R1, d1, D1_new)
        # λ_1_new=permute(S1,(2,),(1,));

        # #B_new: (D1_new, R2_new, M3_new) => (R2, D1, M3)
        # B_new=bosonic_permute_neighbour_ind(B_new,1,2,3);#(R2_new, D1_new, M3_new)
        # B_new=permute(B_new,(1,2,),(3,));

        # #T1_new: (M1_R1, d1, D1_new) => (M1, d1, R1, D1)
        # @tensor T1_new[:]:=T1_new[1,-3,-4]*Ut1'[-1,-2,1];#(M1, R1, d1, D1_new)
        # T1_new=bosonic_permute_neighbour_ind(T1_new,2,3,4);#(M1, d1, R1, D1_new)

        # #T2_new: (M2_D2, d2, R2_new) => (M2, d2, R2, D2)
        # @tensor T2_new[:]:=T2_new[1,-3,-4]*Ut2'[-1,-2,1];#(M2, D2, d2, R2_new)
        # T2_new=bosonic_permute_neighbour_ind(T2_new,2,3,4);#(M2, d2, D2, R2_new)
        # T2_new=bosonic_permute_neighbour_ind(T2_new,3,4,4);#(M2, d2, R2_new, D2)

        # #T3_new: (M3_new, d3, R3_D3) => (M3, d3, R3, D3)
        # @tensor T3_new[:]:=T3_new[-1,-2,1]*Ut3'[-3,-4,1];#(M3_new, d3, R3, D3)



        T2_new=U2*sqrt(S2);#(M2_D2, d2, R2_new)     #here how to absorb S doesn't matter, as we will redetermine S by svd after sweep optimization
        T2_new=permute(T2_new,(1,2,),(3,));
        B_new=sqrt(S2)*V2;#(R2_new, M1_R1, d1, d3, R3_D3)
        λ_2_new=sqrt(S2);

        λ_3_new=sqrt(S3);
        λ_3_new_inv=my_pinv(λ_3_new);
        @tensor B_new[:]:=B_new[-1,-2,-3,1,2]*V3'[1,2,3]*λ_3_new_inv'[3,-4];#(R2_new, M1_R1, d1, M3_new)
        T3_new=sqrt(S3)*V3;#(M3_new, d3, R3_D3)
        T3_new=permute(T3_new,(1,2,),(3,));

        B_new=bosonic_permute_neighbour_ind(B_new,1,2,4);#(M1_R1, R2_new, d1, M3_new)
        B_new=bosonic_permute_neighbour_ind(B_new,2,3,4);#(M1_R1, d1, R2_new, M3_new)
        λ_1_new=sqrt(S1);
        λ_1_new_inv=my_pinv(λ_1_new);
        @tensor B_new[:]:=λ_1_new_inv'[-1,3]*U1'[3,1,2]*B_new[1,2,-2,-3];#(D1_new, R2_new, M3_new)
        T1_new=U1*sqrt(S1);#(M1_R1, d1, D1_new)
        T1_new=permute(T1_new,(1,2,),(3,));

        #B_new: (D1_new, R2_new, M3_new) => (R2, D1, M3)
        B_new=bosonic_permute_neighbour_ind(B_new,1,2,3);#(R2_new, D1_new, M3_new)
        B_new=permute(B_new,(1,2,),(3,));

        method=2;#two methods are the same
        if method==1
            # big_T_compressed=bosonic_permute_neighbour_ind(B_new,1,2,3);#(D1_new, R2_new, M3_new)
            # @tensor big_T_compressed[:]:=T1_new[-1,-2,1]*big_T_compressed[1,-3,-4];#(M1_R1, d1, R2_new, M3_new)
            # big_T_compressed=bosonic_permute_neighbour_ind(big_T_compressed,2,3,4);#(M1_R1,R2_new, d1,  M3_new)
            # big_T_compressed=bosonic_permute_neighbour_ind(big_T_compressed,1,2,4);#(R2_new, M1_R1, d1,  M3_new)
            # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,-3,1]*T3_new[1,-4,-5];#(R2_new, M1_R1, d1,  d3, R3_D3)
            # @tensor big_T_compressed[:]:=T2_new[-1,-2,1]*big_T_compressed[1,-3,-4,-5,-6];#(M2_D2, d2,  M1_R1, d1,  d3, R3_D3)

            # big_T_compressed=bosonic_permute_neighbour_ind(big_T_compressed,2,3,6);#(M2_D2,  M1_R1, d2, d1,  d3, R3_D3)
            # big_T_compressed=bosonic_permute_neighbour_ind(big_T_compressed,3,4,6);#(M2_D2,  M1_R1, d1, d2,  d3, R3_D3)
            # @tensor big_T_compressed[:]:=big_T_compressed[-1,-2,1,2,3,-4]*bosonic_fu_Up[-3,1,2,3];#(new2, new1,  d123, new3)
            # big_T_compressed=permute(big_T_compressed,(1,2,),(3,4,))#(new2, new1), (d123, new3)


        else
            big_T_compressed=bosonic_build_triangle_from_4tensors(B_new,T1_new,T2_new,T3_new)
        end


        return B_new,T1_new,T2_new,T3_new, big_T_compressed
    elseif trun_order=="successive"
    end
end

function bosonic_truncation_env_gauge(env_top,env_bot, big_T,D_max, trun_order, trun_tol)
    #big_T: (2, 1), (d123, 3)
    #env_bot: new_ind,2,3,1
    #env_top: 2,3,1, new_ind
    u,s,v=tsvd(env_bot,(1,2,3,),(4,));
    gauge1=s*v;#11,1
    gauge1_inv=v'*my_pinv(s);
    u,s,v=tsvd(env_bot,(1,3,4,),(2,));
    gauge2=s*v;#2,2
    gauge2_inv=v'*my_pinv(s);
    u,s,v=tsvd(env_bot,(1,2,4,),(3,));
    gauge3=s*v;#33,3
    gauge3_inv=v'*my_pinv(s);

    @tensor big_T_new[:]:=big_T[2,1,-3,3]*gauge1[-2,1]*gauge2[-1,2]*gauge3[-4,3];
    big_T_new=permute(big_T_new,(1,2,),(3,4,));

    B_new,T1_new,T2_new,T3_new, big_T_compressed=bosonic_truncation_direct(big_T_new,D_max, trun_order, trun_tol);
    @tensor big_T_compressed[:]:=big_T_compressed[2,1,-3,3]*gauge1_inv[-2,1]*gauge2_inv[-1,2]*gauge3_inv[-4,3];
    big_T_compressed=permute(big_T_compressed,(1,2,),(3,4,));

    #T1_new: (M1_R1, d1, D1_new)
    @tensor T1_new[:]:=T1_new[1,-2,-3]*gauge1_inv[-1,1];
    T1_new=permute(T1_new,(1,2,),(3,));
    #T2_new: (M2_D2, d2, R2_new)
    @tensor T2_new[:]:=T2_new[1,-2,-3]*gauge2_inv[-1,1];
    T2_new=permute(T2_new,(1,2,),(3,));
    #T3_new: (M3_new, d3, R3_D3)
    @tensor T3_new[:]:=T3_new[-1,-2,1]*gauge3_inv[-3,1];
    T3_new=permute(T3_new,(1,2,),(3,));

    return B_new,T1_new,T2_new,T3_new, big_T_compressed
end



function bosonic_triangle_full_update(
    gate,
    B_set,
    T_set,
    CTM_cell,
    Lx,
    Ly,
    coord,
    D_max,
    trun_order,
    trun_tol,
    n_sweep,
)
    # """
    #          M1     R1
    #            \   /
    #             \ /....d1
    #              |                   T1 =  |M1, d1><D1, R1|=|M1, d1><|R1, D1
    #              |D1

    #              |                B=|R2, D1><M3|
    #             / \

    #   M2\   /R2    M3\   /R3
    #      \ /....d2    \ /....d3
    #       |            |
    #       |D2          |D3

    #       T2           T3

    # T2=|M2, d2><D2, R2|=|M2, d2><|R2, D2
    # T3=|M3, d3><D3, R3|=|M3, d3><|R3, D3
    # """

    (c1,c2)=coord;
    gates_ru_ld_rd=gate;


    B1_res, B1_keep, B2_res, B2_keep, B3_res, B3_keep,  B1_B2_T_B3, B1_B2_T_B3_op = bosonic_split_3tensors(T_set[mod1(c1+1,Lx),c2], T_set[c1,mod1(c2+1,Ly)], T_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], B_set[mod1(c1+1,Lx),mod1(c2+1,Ly)], gates_ru_ld_rd);


    T_LU=B_set[c1,c2];
    T_double_LU, _ = bosonic_build_double_layer_noswap_Tm(T_LU',T_LU, false);#L M U
    B_LU=T_set[c1,c2];
    B_double_LU, _ = bosonic_build_double_layer_noswap_Bm(B_LU',B_LU, true);#D R M

    T_RU=B_set[mod1(c1+1,Lx),c2];
    T_double_RU, _ = bosonic_build_double_layer_noswap_Tm(T_RU',T_RU, false);#L M U

    T_LD=B_set[c1,mod1(c2+1,Ly)];
    T_double_LD, _ = bosonic_build_double_layer_noswap_Tm(T_LD',T_LD, false);#L M U

    B_double_RU, _ = bosonic_build_double_layer_noswap_Bm(B1_res',B1_res,false);#D R M
    B_double_LD, _ = bosonic_build_double_layer_noswap_Bm(B2_res',B2_res,false);#D R M
    B_double_RD, _ = bosonic_build_double_layer_noswap_Bm(B3_res',B3_res,false);#D R M
    BigTriangle_double_Noswap, U_L,U_D,U_U = bosonic_build_double_layer_noswap_Tm(B1_B2_T_B3',B1_B2_T_B3_op, true);#L M U = L D U
    BigTriangle_double_env=bosonic_contract_triangle_env(CTM_cell, T_double_LU, T_double_RU, T_double_LD, B_double_LU, B_double_RU, B_double_LD, B_double_RD, mod1(c1,Lx),mod1(c2,Ly), Lx, Ly);#L M U = L D U

    @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env[1,2,3]*U_L[1,-1,-4]*U_D[2,-2,-5]*U_U[-3,-6,3]; # storage order: L', D', U',   L, D, U,  fermionic order: L',L,U',U,D,D'


        BigTriangle_double_env_expand=permute(BigTriangle_double_env_expand,(1,2,3,),(4,5,6,));# storage order: L', D', U',       L, D, U
        BigTriangle_double_env_expand=permute(BigTriangle_double_env_expand,(1,2,3,),(4,5,6,));

        #eu,ev=eigen(BigTriangle_double_env_expand);
        eu,ev=eigh(BigTriangle_double_env_expand);
        eu=bosonic_check_positive(eu);

        #M=ev*eu*ev';
        # env_bot=ev';#new_ind,1,2,3
        # env_top=ev;# 1,2,3, new_ind
        env_bot=sqrt(eu)*ev';#new_ind,2,3,1
        env_top=ev*sqrt(eu);# 2,3,1, new_ind


    # @tensor BigTriangle_double_env_expand[:]:=BigTriangle_double_env_expand[1,2,3,4,5,6]*U_L'[1,4,-1]*U_D'[2,5,-2]*U_U'[-3,3,6];
    # ov1=@tensor BigTriangle_double_env_expand[1,2,3]*BigTriangle_double_Noswap[1,2,3];
    # ov2=ob_2x2(CTM_cell,AA_cell[c1][c2],AA_cell[mod1(c1+1,Lx)][c2],AA_cell[c1][mod1(c2+1,Ly)],AA_cell[mod1(c1+1,Lx)][mod1(c2+1,Ly)],mod1(c1-1,Lx),mod1(c2-1,Ly));
    # E=E+ov1/ov2;

    ############################################
    #direct truncation
    B_new,T1_new,T2_new,T3_new, big_T_compressed=bosonic_truncation_direct(B1_B2_T_B3_op,D_max, trun_order, trun_tol)
    println("direct truncation:"*string(space(B_new)))

    # #test overlap without environment
    ov12=bosonic_get_overlap_env(env_top,env_bot,big_T_compressed',B1_B2_T_B3_op);
    ov11=bosonic_get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=bosonic_get_overlap_env(env_top,env_bot,big_T_compressed',big_T_compressed);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap without optimization:"*string(norm(ov)))

    println("overlap with environmen after optimization:");
    B_new_a,T1_new_a,T2_new_a,T3_new_a,big_T_compressed_opt_a, ov_a=bosonic_sweep_optimizations(n_sweep,B1_B2_T_B3_op,env_top,env_bot, B_new,T1_new,T2_new,T3_new)


    ####################################
    #truncation with env gauge
    B_new,T1_new,T2_new,T3_new, big_T_compressed=bosonic_truncation_env_gauge(env_top,env_bot, B1_B2_T_B3_op,D_max, trun_order, trun_tol)
    println("truncation with env gauge:"*string(space(B_new)))
    # #test overlap without environment
    ov12=bosonic_get_overlap_env(env_top,env_bot,big_T_compressed',B1_B2_T_B3_op);
    ov11=bosonic_get_overlap_env(env_top,env_bot,B1_B2_T_B3_op',B1_B2_T_B3_op);
    ov22=bosonic_get_overlap_env(env_top,env_bot,big_T_compressed',big_T_compressed);
    ov=ov12/sqrt(ov11*ov22);
    println("overlap without optimization:"*string(norm(ov)))

    # println([ov12,ov11,ov22])
    println("overlap with environmen after optimization:");
    B_new_b,T1_new_b,T2_new_b,T3_new_b,big_T_compressed_opt_b, ov_b=bosonic_sweep_optimizations(n_sweep,B1_B2_T_B3_op,env_top,env_bot, B_new,T1_new,T2_new,T3_new)

    #########################################
    if ov_a>ov_b
        println("direct truncation better")
        B_new_=B_new_a;
        T1_new_=T1_new_a;
        T2_new_=T2_new_a;
        T3_new_=T3_new_a;
        big_T_compressed_opt_=big_T_compressed_opt_a;
    else
        println("truncation with gauge better")
        B_new_=B_new_b;
        T1_new_=T1_new_b;
        T2_new_=T2_new_b;
        T3_new_=T3_new_b;
        big_T_compressed_opt_=big_T_compressed_opt_b;
    end
    # println(space(B_new_))
    # println(space(T1_new_))
    # println(space(T2_new_))
    # println(space(T3_new_))
    # println(space(big_T_compressed_opt_))


    #T1_new: (new1, d1),  (D1)
    #T2_new: (new2, d2),  (R2)
    #T3_new: (M3, d3), (new3)
    #B_new: (R2, D1), (M3)



    #T1=|M1, d1><D1, R1|=|M1, d1><|R1, D1
    #T1_res:(M1), (R1, new1)
    @tensor T1_new_opt[:]:=B1_res[-1,-2,1]*T1_new_[1,-3,-4];#(M1)(R1, new1), (new1, d1)(D1) ->  (M1,R1, d1,D1)
    T1_new_opt=bosonic_permute_neighbour_ind(T1_new_opt,2,3,4);#(M1,d1, R1,D1)
    T1_new_opt=permute(T1_new_opt,(1,),(2,3,4,));#(M1),(d1,R1,D1)

    #T2=|M2, d2><D2, R2|=|M2, d2><|R2, D2
    #T2_res: (M2), (new2, D2)
    B2_res=bosonic_permute_neighbour_ind(B2_res,2,3,3);#(M2)(D2,new2)
    @tensor T2_new_opt[:]:=B2_res[-1,-2,1]*T2_new_[1,-3,-4];#(M2)(D2,new2), (new2, d2)(R2)  ->  (M2,D2, d2,R2)
    T2_new_opt=bosonic_permute_neighbour_ind(T2_new_opt,2,3,4);#(M2,d2, D2,R2)
    T2_new_opt=bosonic_permute_neighbour_ind(T2_new_opt,3,4,4);#(M2,d2, R2,D2)
    T2_new_opt=permute(T2_new_opt,(1,),(2,3,4,));#(M2),(d2,R2,D2)

    #T3=|M3, d3><D3, R3|=|M3, d3><|R3, D3
    #T3_res: (new3), (R3, D3)
    @tensor T3_new_opt[:]:=T3_new_[-1,-2,1]*B3_res[1,-3,-4];#(M3, d3)(new3),  (new3)(R3, D3)  ->  (M3,d3, R3,D3)
    T3_new_opt=permute(T3_new_opt,(1,),(2,3,4,));#(M3),(d3,R3,D3)



    #B=|R2, D1><M3|
    B_new_opt=B_new_;

    @assert (length(codomain(T1_new_opt))==1)&(length(domain(T1_new_opt))==3)
    @assert (length(codomain(T2_new_opt))==1)&(length(domain(T2_new_opt))==3)
    @assert (length(codomain(T3_new_opt))==1)&(length(domain(T3_new_opt))==3)
    @assert (length(codomain(B_new_opt))==2)&(length(domain(B_new_opt))==1)

    T1_new_opt=T1_new_opt/norm(T1_new_opt);
    T2_new_opt=T2_new_opt/norm(T2_new_opt);
    T3_new_opt=T3_new_opt/norm(T3_new_opt);
    B_new_opt=B_new_opt/norm(B_new_opt);

    (c1,c2)=coord;
    T_set[mod1(c1+1,Lx),c2]=T1_new_opt;
    T_set[c1,mod1(c2+1,Ly)]=T2_new_opt;
    T_set[mod1(c1+1,Lx),mod1(c2+1,Ly)]=T3_new_opt;
    B_set[mod1(c1+1,Lx),mod1(c2+1,Ly)]=B_new_opt;

    return B_set, T_set
end


"""
    bosonic_triangle_J1_gate(dt, physical_space; J1=1, Jchi_up=0)

Construct the three-site imaginary-time gate on one simplex-centred triangle,

`exp[-dt*(J1*(S1⋅S2 + S2⋅S3 + S3⋅S1) +
Jchi_up*S1⋅(S2×S3))]`.

Only the simplex-centred/up triangles are updated. Consequently `Jchi_up`
implements a half-triangle chirality term and is zero by default.

`Hamiltonians(physical_space)` is supplied by
`src/bosonic/square/square_spin_operator.jl` and supports dense spin-1/2 and
SU(2)-symmetric physical spaces.
"""
function bosonic_triangle_J1_gate(dt, physical_space; J1=1, Jchi_up=0)
    _, H123chiral, H12, H31, H23 = Hamiltonians(physical_space)
    # The open physical legs in the bosonic Julia iPEPS convention contract
    # the transpose of the dense operator convention used by the Python
    # yastn code.  This leaves S.S invariant but reverses the purely imaginary
    # scalar-chirality operator.  Keep Jchi's sign identical across the two
    # codes by compensating that transpose here.
    H123chiral = -H123chiral
    H_triangle = J1 * (H12 + H31 + H23) + Jchi_up * H123chiral
    H_triangle = permute(H_triangle, (1, 2, 3), (4, 5, 6))
    hermiticity_error = norm(H_triangle - H_triangle') / max(norm(H_triangle), eps())
    @assert hermiticity_error < 1e-12
    eigenvalues, eigenvectors = eigh(H_triangle)
    gate = eigenvectors * exp(-dt * eigenvalues) * eigenvectors'

    # `Hamiltonians` follows the observable/RDM convention. The local iPESS
    # update contracts the three input legs with physical domain legs, so all
    # six legs must be bent to the opposite TensorKit orientation.
    return permute(gate, (4, 5, 6), (1, 2, 3))
end

function bosonic_unpack_ctm_result(result)
    if length(result) == 8
        return result
    elseif length(result) == 6
        CTM_cell, AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell = result
        return CTM_cell, AA_cell, U_L_cell, U_D_cell, U_R_cell, U_U_cell, missing, missing
    end
    error("Unexpected CTMRG_cell return length $(length(result)); expected 6 or 8")
end


"""
    bosonic_measure_J1_Jchi_energy(A_cell, AA_cell, CTM_cell; J1=1, Jchi=0)

Measure the triangular-lattice spin model from the converged bosonic CTM
environment.  The up-triangle reduced density matrix supplies all three
nearest-neighbour bonds and `chi_up`; the down-triangle density matrix
supplies `chi_down`.  Both requested energies are therefore formed from the
same contractions:

`E_half = E_J1 + Jchi*chi_up`

`E_full = E_J1 + Jchi*(chi_up + chi_down)`.
"""
function bosonic_measure_J1_Jchi_energy(
    A_cell,
    AA_cell,
    CTM_cell;
    J1=1,
    Jchi=0,
)
    isdefined(@__MODULE__, :ob_LU_RU_LD_cell) || error(
        "Load src/bosonic/square/square_model_cell.jl before measuring J1-Jchi energy."
    )
    Lx_local, Ly_local = length(A_cell), length(A_cell[1])
    @assert (Lx_local, Ly_local) == (Lx, Ly)

    _, H123chiral, H12, H31, H23 = Hamiltonians(space(A_cell[1][1], 5))
    # Match Python/yastn's epsilon_abc S1^a S2^b S3^c convention.  The Julia
    # open-RDM contraction transposes dense physical operators; this changes
    # the sign of chirality (one Sy per term) but not Heisenberg interactions.
    H123chiral = -H123chiral
    # `Hamiltonians` uses the observable convention inherited from the old
    # TensorKit code. Bend all physical legs to the current open-RDM
    # convention, exactly as in `bosonic_triangle_J1_gate`.
    H123chiral = permute(H123chiral, (4, 5, 6), (1, 2, 3))
    H12 = permute(H12, (4, 5, 6), (1, 2, 3))
    H31 = permute(H31, (4, 5, 6), (1, 2, 3))
    H23 = permute(H23, (4, 5, 6), (1, 2, 3))
    AA_open_cell = initial_tuple_cell(Lx_local, Ly_local)
    for cx in 1:Lx_local, cy in 1:Ly_local
        AA_open, _ = build_double_layer_open(A_cell[cx][cy])
        AA_open_cell = fill_tuple(AA_open_cell, AA_open, cx, cy)
    end

    V_s = space(A_cell[1][1], 5)
    V_ss = fuse(V_s' ⊗ V_s)
    U_s_s = unitary(V_ss, V_s' ⊗ V_s)'

    SS_x_set = zeros(ComplexF64, Lx_local, Ly_local)
    SS_y_set = zeros(ComplexF64, Lx_local, Ly_local)
    SS_diagonal_set = zeros(ComplexF64, Lx_local, Ly_local)
    chi_up_set = zeros(ComplexF64, Lx_local, Ly_local)
    chi_down_set = zeros(ComplexF64, Lx_local, Ly_local)

    for cx in 1:Lx_local, cy in 1:Ly_local
        pos_LU = (mod1(cx + 1, Lx_local), mod1(cy + 1, Ly_local))
        pos_RU = (mod1(cx + 2, Lx_local), mod1(cy + 1, Ly_local))
        pos_LD = (mod1(cx + 1, Lx_local), mod1(cy + 2, Ly_local))
        pos_RD = (mod1(cx + 2, Lx_local), mod1(cy + 2, Ly_local))

        # This (LU, RU, LD) RDM supplies the three inequivalent J1 bonds.
        # Python's `chi_down` uses the opposite cyclic site order, equivalent
        # to reversing these three physical legs before inserting H_chi.
        rho_bonds = ob_LU_RU_LD_cell(
            cx,
            cy,
            CTM_cell,
            AA_cell[pos_RD[1]][pos_RD[2]],
            AA_open_cell[pos_LU[1]][pos_LU[2]],
            AA_open_cell[pos_RU[1]][pos_RU[2]],
            AA_open_cell[pos_LD[1]][pos_LD[2]],
        )
        rho_chi_down = permute(rho_bonds, (3, 2, 1,))
        @tensor rho_bonds[:]:=rho_bonds[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3]
        @tensor rho_chi_down[:]:=rho_chi_down[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3]
        norm_bonds = @tensor rho_bonds[1,2,3,1,2,3]
        SS_x_set[cx, cy] = (@tensor rho_bonds[1,2,3,4,5,6] * H12[1,2,3,4,5,6]) / norm_bonds
        SS_y_set[cx, cy] = (@tensor rho_bonds[1,2,3,4,5,6] * H31[1,2,3,4,5,6]) / norm_bonds
        SS_diagonal_set[cx, cy] = (@tensor rho_bonds[1,2,3,4,5,6] * H23[1,2,3,4,5,6]) / norm_bonds
        chi_down_set[cx, cy] = (
            @tensor rho_chi_down[1,2,3,4,5,6] * H123chiral[1,2,3,4,5,6]
        ) / norm_bonds

        # Simplex-centred up triangle. Raw order is (LD, RU, RD); reorder to
        # (RU, LD, RD), a cyclic shift of Python's (LD, RD, RU) convention and
        # the same ordering used by the FU gate.
        rho_chi_up = ob_LD_RU_RD_cell(
            cx,
            cy,
            CTM_cell,
            AA_cell[pos_LU[1]][pos_LU[2]],
            AA_open_cell[pos_LD[1]][pos_LD[2]],
            AA_open_cell[pos_RU[1]][pos_RU[2]],
            AA_open_cell[pos_RD[1]][pos_RD[2]],
        )
        rho_chi_up = permute(rho_chi_up, (2, 1, 3,))
        @tensor rho_chi_up[:]:=rho_chi_up[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3]
        norm_chi_up = @tensor rho_chi_up[1,2,3,1,2,3]
        chi_up_set[cx, cy] = (
            @tensor rho_chi_up[1,2,3,4,5,6] * H123chiral[1,2,3,4,5,6]
        ) / norm_chi_up
    end

    normalization = Lx_local * Ly_local
    SS_x = real(sum(SS_x_set) / normalization)
    SS_y = real(sum(SS_y_set) / normalization)
    SS_diagonal = real(sum(SS_diagonal_set) / normalization)
    chi_up = real(sum(chi_up_set) / normalization)
    chi_down = real(sum(chi_down_set) / normalization)
    energy_J1 = real(J1) * (SS_x + SS_y + SS_diagonal)
    energy_chi_up = real(Jchi) * chi_up
    energy_chi_down = real(Jchi) * chi_down
    energy_half_chirality = energy_J1 + energy_chi_up
    energy_full_J1_Jchi = energy_half_chirality + energy_chi_down

    return (
        energy_full_J1_Jchi=energy_full_J1_Jchi,
        energy_half_chirality=energy_half_chirality,
        energy_J1=energy_J1,
        energy_chi_up=energy_chi_up,
        energy_chi_down=energy_chi_down,
        SS_x=SS_x,
        SS_y=SS_y,
        SS_diagonal=SS_diagonal,
        chi_up=chi_up,
        chi_down=chi_down,
        SS_x_set=SS_x_set,
        SS_y_set=SS_y_set,
        SS_diagonal_set=SS_diagonal_set,
        chi_up_set=chi_up_set,
        chi_down_set=chi_down_set,
    )
end


function bosonic_print_J1_Jchi_energy(energy)
    println(
        "E_full_J1_Jchi=$(energy.energy_full_J1_Jchi), " *
        "E_half_chirality=$(energy.energy_half_chirality), " *
        "E_J1=$(energy.energy_J1), " *
        "E_chi_up=$(energy.energy_chi_up), " *
        "E_chi_down=$(energy.energy_chi_down)"
    )
    println(
        "SS_x=$(energy.SS_x), SS_y=$(energy.SS_y), " *
        "SS_diagonal=$(energy.SS_diagonal), " *
        "chi_up=$(energy.chi_up), chi_down=$(energy.chi_down)"
    )
    println(
        "SS_x_set=$(vec(energy.SS_x_set)), " *
        "SS_y_set=$(vec(energy.SS_y_set)), " *
        "SS_diagonal_set=$(vec(energy.SS_diagonal_set))"
    )
    println(
        "chi_up_set=$(vec(energy.chi_up_set)), " *
        "chi_down_set=$(vec(energy.chi_down_set))"
    )
    flush(stdout)
    return nothing
end


"""
    bosonic_full_update_iPESS_J1_once(B_set, T_set, environment_chi, dt,
                                      D_max, ctm_setting; kwargs...)

Perform exactly one environment-aware full-update sweep over all
simplex-centred (up) triangles.  This is intended primarily to lift an
existing state, e.g. from `D=6` to `D=8`, before variational optimization of
the complete Hamiltonian.

The cell must have at least two entries in both directions.  A local
triangle writes three independently optimized site tensors; in a `1×N` or
`N×1` cell two or more of those sites alias the same array entry, which does
not define a consistent bond-dimension expansion.

Required functions (`convert_iPESS_to_iPEPS`, `initial_condition`, and
`CTMRG_cell`) must be loaded by the caller.  The input tensors are copied by
default.  The return value is a named tuple containing the updated state and
the final CTM environment.
"""
function bosonic_full_update_iPESS_J1_once(
    B_set,
    T_set,
    environment_chi::Integer,
    dt,
    D_max::Integer,
    ctm_setting;
    J1=1,
    Jchi_up=0,
    n_sweep::Integer=10,
    trun_order="simultaneous",
    trun_tol=1e-10,
    init_CTM=[],
    copy_state::Bool=true,
    verbose::Bool=true,
)
    @assert dt != 0 "A zero time step cannot populate the new bond sectors."
    @assert D_max > 0
    @assert n_sweep > 0
    @assert trun_order == "simultaneous" "Only simultaneous truncation is implemented."

    B_work = copy_state ? deepcopy(B_set) : B_set
    T_work = copy_state ? deepcopy(T_set) : T_set
    Lx_local, Ly_local = size(B_work)
    @assert size(T_work) == (Lx_local, Ly_local)
    (Lx_local >= 2 && Ly_local >= 2) || throw(ArgumentError(
        "Bond-expanding triangular iPESS full update requires Lx >= 2 and " *
        "Ly >= 2 so the three sites of each triangle are distinct; got " *
        "cell size $(Lx_local)x$(Ly_local). Extend the state to at least " *
        "a 2x2 cell before running this update."
    ))

    # The repository's iPESS-to-iPEPS cell converter uses these globals.
    global Lx = Lx_local
    global Ly = Ly_local

    physical_space = space(T_work[1, 1], 2)
    gate = bosonic_triangle_J1_gate(
        dt,
        physical_space;
        J1=J1,
        Jchi_up=Jchi_up,
    )

    A_cell = convert_iPESS_to_iPEPS(B_work, T_work)
    has_initial_CTM = !(
        init_CTM === nothing || (init_CTM isa AbstractVector && isempty(init_CTM))
    )
    init = initial_condition(
        init_type="PBC",
        reconstruct_CTM=!has_initial_CTM,
        reconstruct_AA=true,
    )
    CTM_cell, AA_cell, _, _, _, _, ite_num, ite_err = bosonic_unpack_ctm_result(
        CTMRG_cell(A_cell, environment_chi, init, init_CTM, ctm_setting)
    )
    verbose && println("initial CTMRG: iterations=$ite_num, error=$ite_err")

    triangle_count = Lx_local * Ly_local
    triangle_index = 0
    for cy in 1:Ly_local
        for cx in 1:Lx_local
            triangle_index += 1
            verbose && println("J1 full update triangle $triangle_index/$triangle_count at ($cx,$cy)")
            B_work, T_work = bosonic_triangle_full_update(
                gate,
                B_work,
                T_work,
                CTM_cell,
                Lx_local,
                Ly_local,
                (cx, cy),
                D_max,
                trun_order,
                trun_tol,
                n_sweep,
            )

            # As in the original FU implementation, refresh the environment
            # after every overlapping local triangle update.
            A_cell = convert_iPESS_to_iPEPS(B_work, T_work)
            init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
            CTM_cell, AA_cell, _, _, _, _, ite_num, ite_err = bosonic_unpack_ctm_result(
                CTMRG_cell(A_cell, environment_chi, init, [], ctm_setting)
            )
            verbose && println("CTMRG: iterations=$ite_num, error=$ite_err")
        end
    end

    # Match the old triangular Hubbard FU logic: measure once after a full
    # sweep, using the final CTM environment. Full- and half-chirality model
    # energies are two combinations of the same measured observables.
    energy = bosonic_measure_J1_Jchi_energy(
        A_cell,
        AA_cell,
        CTM_cell;
        J1=J1,
        Jchi=Jchi_up,
    )
    verbose && bosonic_print_J1_Jchi_energy(energy)

    bond_spaces = [space(B_work[cx, cy], leg) for cx in 1:Lx_local for cy in 1:Ly_local for leg in 1:3]
    achieved_dimensions = dim.(bond_spaces)
    if any(d -> d < D_max, achieved_dimensions)
        @warn "The requested bond dimension was not populated on every simplex leg" D_max achieved_dimensions
    end

    return (
        B_set=B_work,
        T_set=T_work,
        CTM_cell=CTM_cell,
        AA_cell=AA_cell,
        ctm_iterations=ite_num,
        ctm_error=ite_err,
        energy=energy,
        bond_dimensions=achieved_dimensions,
    )
end
