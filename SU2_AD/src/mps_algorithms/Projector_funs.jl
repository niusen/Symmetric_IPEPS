function projector_virtual(V::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})


    if V==Rep[SU₂](0=>2, 1/2=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1
        T=TensorMap(M,Rep[SU₂](0=>2),V);
        P_even[1]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1),V);
        P_odd[1]=T;
    elseif  V==Rep[SU₂](0=>2, 1/2=>1)'
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1
        T=TensorMap(M,Rep[SU₂](0=>2)',V);
        P_even[1]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1)',V);
        P_odd[1]=T;
    end


    return P_odd,P_even
end

function projector_physical(V::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})

    if V==Rep[SU₂](1/2=>1)


        P_even=[];

        M=zeros(2,2)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1),Rep[SU₂](1/2=>1));
        P_odd=T;
    elseif V==Rep[SU₂](1/2=>1)'

        P_even=[];

        M=zeros(2,2)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1)',Rep[SU₂](1/2=>1)');
        P_odd=T;
    elseif V==Rep[SU₂](0=>2,1/2=>1)
        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[SU₂](0=>2),Rep[SU₂](0=>2,1/2=>1));
        P_even=T;

        M=zeros(2,4)*im;
        M[1,2+1]=1;
        M[2,2+2]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1),Rep[SU₂](0=>2,1/2=>1));
        P_odd=T;
    elseif V==Rep[SU₂](0=>2,1/2=>1)'
        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[SU₂](0=>2)',Rep[SU₂](0=>2,1/2=>1)');
        P_even=T;

        M=zeros(2,4)*im;
        M[1,2+1]=1;
        M[2,2+2]=1;
        T=TensorMap(M,Rep[SU₂](1/2=>1)',Rep[SU₂](0=>2,1/2=>1)');
        P_odd=T;
    end

    

    return P_odd,P_even

end

function projector_virtual(V) #U1 or U1xSU2
    VV1=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'
    VV2=Rep[U₁ × SU₂]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1)
    
    if V==Rep[U₁](0=>1, 1=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,1);

        M=zeros(1,2)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁](0=>1),V);
        P_even[1]=T;

        M=zeros(1,2)*im;
        M[1,2]=1;
        T=TensorMap(M,Rep[U₁](1=>1),V);
        P_odd[1]=T;
    elseif V==Rep[U₁](0=>1, -1=>2, -2=>1)
        P_odd=Vector(undef,1);
        P_even=Vector(undef,2);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁](0=>1),V);
        P_even[1]=T;

        M=zeros(1,4)*im;
        M[1,4]=1;
        T=TensorMap(M,Rep[U₁](-2=>1),V);
        P_even[2]=T;

        M=zeros(2,4)*im;
        M[1,2]=1;
        M[2,3]=1;
        T=TensorMap(M,Rep[U₁](-1=>2),V);
        P_odd[1]=T;
    elseif V==VV1
        P_odd=Vector(undef,1);
        P_even=Vector(undef,2);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>1)',V);
        P_even[1]=T;

        M=zeros(1,4)*im;
        M[1,2]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((2, 0)=>1)',V);
        P_even[2]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((1, 1/2)=>1)',V);
        P_odd[1]=T;
    elseif V==VV2
        P_odd=Vector(undef,2);
        P_even=Vector(undef,4);

        M=zeros(1,16)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>1),V);
        P_even[1]=T;

        M=zeros(3,16)*im;
        M[1,2]=1;
        M[2,3]=1;
        M[3,4]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-2, 0)=>3),V);
        P_even[2]=T;

        M=zeros(1,16)*im;
        M[1,5]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-4, 0)=>1),V);
        P_even[3]=T;

        M=zeros(3,16)*im;
        M[1,14]=1;
        M[2,15]=1;
        M[3,16]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-2, 1)=>1),V);
        P_even[4]=T;

        M=zeros(4,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-1, 1/2)=>2),V);
        P_odd[1]=T;

        M=zeros(4,16)*im;
        M[1,10]=1;
        M[2,11]=1;
        M[3,12]=1;
        M[4,13]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-3, 1/2)=>2),V);
        P_odd[2]=T;

    end


    return P_odd,P_even
end

function projector_virtual_devided(V)
    VV1=Rep[U₁ × SU₂]((0, 0)=>1, (2, 0)=>1, (1, 1/2)=>1)'
    VV2=Rep[U₁ × SU₂]((0, 0)=>1, (-2, 0)=>3, (-4, 0)=>1, (-1, 1/2)=>2, (-3, 1/2)=>2, (-2, 1)=>1)
    
    if V==VV1
        Ps=Vector(undef,3);

        M=zeros(1,4)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>1)',V);
        Ps[1]=T;

        M=zeros(1,4)*im;
        M[1,2]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((2, 0)=>1)',V);
        Ps[2]=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((1, 1/2)=>1)',V);
        Ps[3]=T;
    elseif V==VV2
        Ps=Vector(undef,6);

        M=zeros(1,16)*im;
        M[1,1]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>1),V);
        Ps[1]=T;

        M=zeros(3,16)*im;
        M[1,2]=1;
        M[2,3]=1;
        M[3,4]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-2, 0)=>3),V);
        Ps[2]=T;

        M=zeros(1,16)*im;
        M[1,5]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-4, 0)=>1),V);
        Ps[3]=T;

        M=zeros(3,16)*im;
        M[1,14]=1;
        M[2,15]=1;
        M[3,16]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-2, 1)=>1),V);
        Ps[4]=T;

        M=zeros(4,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-1, 1/2)=>2),V);
        Ps[5]=T;

        M=zeros(4,16)*im;
        M[1,10]=1;
        M[2,11]=1;
        M[3,12]=1;
        M[4,13]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((-3, 1/2)=>2),V);
        Ps[6]=T;

    end


    return Ps
end


function projector_physical(V)#U1 or U1xSU2
    VV1=Rep[U₁ × SU₂]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (1, 1/2)=>2, (-1, 1/2)=>2, (0, 1)=>1)
    if V==Rep[U₁](0=>2, 1=>1, -1=>1)

        M=zeros(2,4)*im;
        M[1,1]=1;
        M[2,2]=1;
        T=TensorMap(M,Rep[U₁](0=>2),V);
        P_even=T;

        M=zeros(2,4)*im;
        M[1,3]=1;
        M[2,4]=1;
        T=TensorMap(M,Rep[U₁](1=>1,-1=>1),V);
        P_odd=T;
    elseif V==VV1
        M=zeros(8,16)*im;
        M[1,1]=1;
        M[2,2]=1;
        M[3,3]=1;
        M[4,4]=1;
        M[5,5]=1;
        M[6,14]=1;
        M[7,15]=1;
        M[8,16]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((0, 0)=>3, (2, 0)=>1, (-2, 0)=>1, (0, 1)=>1),V);
        P_even=T;

        M=zeros(8,16)*im;
        M[1,6]=1;
        M[2,7]=1;
        M[3,8]=1;
        M[4,9]=1;
        M[5,10]=1;
        M[6,11]=1;
        M[7,12]=1;
        M[8,13]=1;
        T=TensorMap(M,Rep[U₁ × SU₂]((1, 1/2)=>2, (-1, 1/2)=>2),V);
        P_odd=T;

    end

    

    return P_odd,P_even

end

function projector_general_SU2(V1::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}; check::Bool=true)
    Spinlist = Any[]

    for s in sectors(V1)
        Spin = s.j
        Dim = dim(V1, s)
        for _ in 1:Dim
            push!(Spinlist, Spin)
        end
    end

    L = length(Spinlist)
    Ps = Vector(undef, L)
    total_dim = sum(Int(2 * S + 1) for S in Spinlist)
    posit = 0

    for cc in 1:L
        S = Spinlist[cc]
        spin_dim = Int(2 * S + 1)
        M = zeros(ComplexF64, spin_dim, total_dim)
        for dd in 1:spin_dim
            M[dd, posit + dd] = 1
        end
        posit += spin_dim

        if V1.dual
            Ps[cc] = TensorMap(M, Rep[SU₂](S => 1)', V1)
        else
            Ps[cc] = TensorMap(M, Rep[SU₂](S => 1), V1)
        end
    end

    if check
        @assert length(Ps) > 0
        @tensor T[:] := Ps[1]'[-1, 1] * Ps[1][1, -2]
        for cc in 2:length(Ps)
            @tensor TT[:] := Ps[cc]'[-1, 1] * Ps[cc][1, -2]
            T = T + TT
        end
        @assert norm(permute(T, (1,), (2,)) - unitary(V1, V1)) < 1e-10
    end

    return Ps
end

"""
    SU2_space_effective_dimension(V; alpha=0.5)

Estimate the computational size of an SU(2)-symmetric space after taking
symmetry reduction into account.  For

    V = SU2Space(S1=>d1, S2=>d2, ...)

this returns

    sum_i d_i * (2*S_i + 1)^alpha.

`alpha=1` gives the full dense dimension.  `alpha=0` counts only reduced
degeneracy dimensions.  Values such as `alpha=0.5` are useful empirical
weights for grouping projectors: high-spin sectors are treated as more
expensive than spin-0 sectors, but not as expensive as in a fully dense
calculation.
"""
function SU2_space_effective_dimension(
    V::GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}};
    alpha::Real=0.5,
)
    effective_dim = 0.0
    for s in sectors(V)
        S = s.j
        effective_dim += dim(V, s) * (2 * Float64(S) + 1)^alpha
    end
    return effective_dim
end

"""
    combine_SU2_projector_group(projector_group; eig_tol=1e-10, check=true)

Combine a small list of fine projectors into one coarse projector.  The input
projectors must all have the same domain `V`.  The group projector is built
from

    Q = P1' * P1 + P2' * P2 + ...

which is a projector on `V`.  We diagonalize `Q`, discard the zero-eigenvalue
subspace with a truncated `eigh`, and return the isometry `P_group` satisfying

    P_group' * P_group ≈ Q.
"""
function combine_SU2_projector_group(
    projector_group;
    eig_tol::Real=1e-12,
    check::Bool=true,
)
    @assert !isempty(projector_group)

    @tensor Q_tensor[:] := projector_group[1]'[-1, 1] * projector_group[1][1, -2]
    for cc in 2:length(projector_group)
        @tensor QQ[:] := projector_group[cc]'[-1, 1] * projector_group[cc][1, -2]
        Q_tensor = Q_tensor + QQ
    end
    Q = permute(Q_tensor, (1,), (2,))

    eu_full, _ = eigh(Q)
    vals = real.(diag(convert(Array, eu_full)))
    nonzero = abs.(vals) .> eig_tol
    rank = count(nonzero)
    @assert rank > 0

    if check
        zero_vals = vals[.!nonzero]
        one_vals = vals[nonzero]
        isempty(zero_vals) || @assert maximum(abs.(zero_vals)) < eig_tol
        @assert maximum(abs.(one_vals .- 1)) < eig_tol
        @assert norm(Q * Q - Q) < eig_tol * max(norm(Q), 1)
    end

    eigh_trunc_fun = isdefined(@__MODULE__, :eigh_trunc) ? eigh_trunc : TensorKit.eigh_trunc
    eu, ev = eigh_trunc_fun(Q; trunc=truncrank(rank))
    vals_keep = real.(diag(convert(Array, eu)))
    if check
        @assert length(vals_keep) == rank
        @assert maximum(abs.(vals_keep .- 1)) < eig_tol
    end

    P_group = ev'

    if check
        @tensor Q_check_tensor[:] := P_group'[-1, 1] * P_group[1, -2]
        Q_check = permute(Q_check_tensor, (1,), (2,))
        @assert norm(Q_check - Q) < eig_tol * max(norm(Q), 1)
        @assert norm(P_group * P_group' - unitary(codomain(P_group), codomain(P_group))) <
            eig_tol * max(dim(codomain(P_group)), 1)
    end

    return P_group
end

"""
    group_SU2_projectors(projectors; max_eff_dim, alpha=0.5, eig_tol=1e-10, check=true)

Group a fine projector list, typically `projector_general_SU2(V)`, into
coarser projectors.  The input projectors are scanned in order.  A running
group is closed when adding the next projector would make

    sum(SU2_space_effective_dimension(space(P, 1); alpha=alpha))

exceed `max_eff_dim`.  Each closed group is combined by
`combine_SU2_projector_group`, i.e. via `sum(P'P)` followed by a truncated
Hermitian eigendecomposition.  When `check=true`, the grouped projectors are
verified to resolve the identity on the original space.
"""
function group_SU2_projectors(
    projectors;
    max_eff_dim::Real,
    alpha::Real=0.5,
    eig_tol::Real=1e-12,
    check::Bool=true,
)
    @assert max_eff_dim > 0
    @assert !isempty(projectors)

    V = domain(projectors[1])
    for P in projectors
        @assert domain(P) == V
    end

    grouped_projectors = Any[]
    projector_group = Any[]
    group_eff_dim = 0.0

    for P in projectors
        P_eff_dim = SU2_space_effective_dimension(space(P, 1); alpha=alpha)
        if !isempty(projector_group) && group_eff_dim + P_eff_dim > max_eff_dim
            push!(grouped_projectors, combine_SU2_projector_group(projector_group; eig_tol=eig_tol, check=check))
            empty!(projector_group)
            group_eff_dim = 0.0
        end

        push!(projector_group, P)
        group_eff_dim += P_eff_dim
    end

    if !isempty(projector_group)
        push!(grouped_projectors, combine_SU2_projector_group(projector_group; eig_tol=eig_tol, check=check))
    end

    if check
        @assert length(grouped_projectors) > 0
        @tensor T[:] := grouped_projectors[1]'[-1, 1] * grouped_projectors[1][1, -2]
        for cc in 2:length(grouped_projectors)
            @tensor TT[:] := grouped_projectors[cc]'[-1, 1] * grouped_projectors[cc][1, -2]
            T = T + TT
        end
        @assert norm(permute(T, (1,), (2,)) - unitary(V, V)) <
            eig_tol * max(dim(V), 1)
    end

    return grouped_projectors
end

function projector_general_SU2_U1(V1)
    Prime=false;
    if string(V1)[end]=='\''
        Prime=true;
    end


    Qnlist=[];
    Spinlist=[];
    
    for s in sectors(V1)
        # println(s)
        # println(dim(V1,s))
        st=replace(string(s), "Irrep[U₁]" => "a");
        st=replace(st, "⊠ Irrep[SU₂]" => "a");
        #println(st)
        left_pos,right_pos,slash_pos=QN_str_search(string(st));

        Qn=parse(Int64, st[left_pos[2]+1:right_pos[1]-1])
        if length(slash_pos)>0
            @assert length(slash_pos)==1
            Numerator=parse(Int64, st[left_pos[3]+1:slash_pos[1]-1])
            Denominator=parse(Int64, st[slash_pos[1]+1:right_pos[2]-1])
            Spin=Numerator/Denominator
        else
            Spin=Numerator=parse(Int64, st[left_pos[3]+1:right_pos[2]-1])
        end
        #println(Spin)
        Dim=dim(V1, s)
        #Dim=Int(Dim*(2*Spin+1))
        for cc=1:Dim
            Qnlist=vcat(Qnlist,Int(Qn));
            Spinlist=vcat(Spinlist,Spin);

        end
        #Qnlist=vcat(Qnlist,Int.(ones(Dim))*Qn);
        
    end
    # println(Qnlist)
    # println(Spinlist)

    L=length(Qnlist);
    Ps=Vector(undef,L);
    total_dim=Int(sum(Spinlist*2 .+1));
    posit=0;
    for cc=1:L
        S=Spinlist[cc];
        Qn=Qnlist[cc];
        M=zeros(Int(2*S+1),total_dim)
        for dd=1:Int(2*S+1)
            M[dd,posit+dd]=1;
        end
        posit=posit+Int(2*S+1);
        if Prime
            T=TensorMap(M,(Rep[U₁ × SU₂]((-Qn, S)=>1))', V1);
        else
            T=TensorMap(M,Rep[U₁ × SU₂]((Qn, S)=>1),V1);
        end
        Ps[cc]=T;
    end

    #check
    @tensor T[:]:=Ps[1]'[-1,1]*Ps[1][1,-2];
    for cc=2:length(Ps);
        @tensor TT[:]:=Ps[cc]'[-1,1]*Ps[cc][1,-2];
        T=T+TT;
    end
    @assert norm(permute(T,(1,),(2,))-unitary(V1,V1))<1e-10

    return Ps
end


function build_double_layer_NoSwap(Ap,A)
    #display(space(A))



    




    Ap=permute(Ap,(1,2,),(3,4,5))
    A=permute(A,(1,2,),(3,4,5));
    
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

