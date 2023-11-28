using LinearAlgebra
using TensorKit

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################



function truncate_multiplet_origin(s,chi,multiplet_tol,trun_tol)
    #the multiplet is not due to su(2) symmetry
    s_dense=sort(abs.(diag(convert(Array,s))),rev=true);

    # println(s_dense/s_dense[1])

    if length(s_dense)>chi
        value_trun=s_dense[chi+1];
    else
        value_trun=0;
    end
    value_max=maximum(s_dense);

    s_Dict=convert(Dict,s);
    
    space_full=space(s,1);
    for sp in sectors(space_full)

        diag_elem=abs.(diag(s_Dict[:data][string(sp)]));
        for cd=1:length(diag_elem)
            if ((diag_elem[cd]/value_max)<trun_tol) | (diag_elem[cd]<=value_trun) |(abs((diag_elem[cd]-value_trun)/value_trun)<multiplet_tol)
                diag_elem[cd]=0;
            end
        end
        s_Dict[:data][string(sp)]=diagm(diag_elem);
    end
    s=convert(TensorMap,s_Dict);

    # s_=sort(diag(convert(Array,s)),rev=true);
    # s_=s_/s_[1];
    # print(s_)
    # @assert 1+1==3
    return s
end

function delet_zero_block(U,Σ,V)

    secs=blocksectors(Σ);
    sec_length=Vector{Int}(undef,length(secs))
    U_dict = convert(Dict,U)
    Σ_dict = convert(Dict,Σ)
    V_dict = convert(Dict,V)

    #ProductSpace(Rep[SU₂](0=>3, 1/2=>4, 1=>4, 3/2=>2, 2=>1))

    for cc =1:length(secs)
        c=secs[cc];
        if (size(diag(Σ_dict[:data][string(c)]),1)>0) & (sum(abs.(diag(Σ_dict[:data][string(c)])))>0)
            inds=findall(x->(abs.(x).>0), diag(Σ_dict[:data][string(c)]));
            U_dict[:data][string(c)]=U_dict[:data][string(c)][:,inds];
            Σ_dict[:data][string(c)]=Σ_dict[:data][string(c)][inds,inds];
            V_dict[:data][string(c)]=V_dict[:data][string(c)][inds,:];

            sec_length[cc]=length(inds);
        else
            delete!(U_dict[:data], string(c))
            delete!(V_dict[:data], string(c))
            delete!(Σ_dict[:data], string(c))
            sec_length[cc]=0;
        end
    end

    #define sector string
    sec_str="ProductSpace(Rep[SU₂](" *string(((dim(secs[1])-1)/2)) * "=>" * string(sec_length[1]);
    for cc=2:length(secs)
        sec_str=sec_str*", " * string(((dim(secs[cc])-1)/2)) * "=>" * string(sec_length[cc]);
    end
    sec_str=sec_str*"))"

    U_dict[:domain]=sec_str
    V_dict[:codomain]=sec_str
    Σ_dict[:domain]=sec_str
    Σ_dict[:codomain]=sec_str

    return convert(TensorMap, U_dict), convert(TensorMap, Σ_dict), convert(TensorMap, V_dict)
end


function Truncations(uM,sM,vM,bond_dim,trun_tol)  
    sM=truncate_multiplet_origin(sM,bond_dim,1e-5,trun_tol);

    uM_new,sM_new,vM_new=delet_zero_block(uM,sM,vM);
    @assert (norm(uM_new*sM_new*vM_new-uM*sM*vM)/norm(uM*sM*vM))<1e-14
    uM=uM_new;
    sM=sM_new;
    vM=vM_new;
    sM=sM/norm(sM)
    return uM,sM,vM
end






function trotter_gate(H,dt)
        eu,ev=eigh(H);
        @assert norm(ev*eu*ev'-H)/norm(H)<1e-14 
        gate=ev*exp(-dt*eu)*ev';
        gate_half=ev*exp(-dt*eu/2)*ev';
    return gate, gate_half
end



function update_RU_triangle(Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U, gate, trun_tol,bond_dim)
    """
                lambda_m_D
                       .
                    | .
                    |.
    lambda_u_L   ---Tu---  lambda_u_R
                    |
                    |
                lambda_m_U            lambda_r_U
                       .                     .
                    | .                   | .
                    |.                    |.
    lambda_m_L   ---Tm--- lambda_m_R   ---Tr---  lambda_m_L
                    |                     |
                    |                     |
                lambda_m_D            lambda_r_D

    """



    #absord lambda
    @tensor Tu_absorbed[:]:=Tu[1,2,3,4,-5]*lambda_u_L[-1,1]*lambda_m_U[-2,2]*lambda_u_R[-3,3]*lambda_m_D[-4,4];
    @tensor Tr_absorbed[:]:=Tr[1,2,3,4,-5]*lambda_m_R[-1,1]*lambda_r_D[-2,2]*lambda_m_L[-3,3]*lambda_r_U[-4,4];
    @tensor Tm_absorbed[:]:=Tm[1,2,-3,-4,-5]*lambda_m_L[1,-1]*lambda_m_D[2,-2];

    #simplify tensors
    U,S,V=tsvd(Tr_absorbed,(1,5,),(2,3,4,));
    Tr_keep=U*S;
    Tr_res=V;
    U,S,V=tsvd(Tu_absorbed,(2,5,),(1,3,4,));
    Tu_keep=U*S;
    Tu_res=V;

    @tensor TT[:]:=Tm_absorbed[-1,-2,1,2,-5]*Tr_keep[1,-6,-3]*Tu_keep[2,-7,-4];
    @tensor TT_new[:]:=TT[-1,-2,-3,-4,1,2,3]*gate[-5,-6,-7,1,2,3];

    U,S,V=tsvd(TT_new,(1,2,3,5,6,),(4,7,); trunc=truncdim(bond_dim));
    U,S,V=Truncations(U,S,V,bond_dim,trun_tol);
    lambda_m_U_new=S;
    Tu_keep_new=permute(V,(1,3,2,));
    @tensor Tu_new[:]:=Tu_keep_new[-2,-5,1]*Tu_res[1,-1,-3,-4];

    TT_new=permute(U*S,(1,2,3,6,4,5,));
    U,S,V=tsvd(TT_new,(1,2,4,5,),(3,6,); trunc=truncdim(bond_dim));
    U,S,V=Truncations(U,S,V,bond_dim,trun_tol);
    lambda_m_R_new=S;
    @tensor Tr_new[:]:=V[-1,1,-5]*Tr_res[1,-2,-3,-4];
    Tm_new=permute(U,(1,2,5,3,4,));

    lambda_m_U_new_inv=pinv(lambda_m_U_new);
    lambda_m_L_inv=pinv(lambda_m_L);
    lambda_m_D_inv=pinv(lambda_m_D);
    @tensor Tm_new[:]:=Tm_new[1,2,-3,3,-5]*lambda_m_L_inv[1,-1]*lambda_m_D_inv[2,-2]*lambda_m_U_new_inv[3,-4];


    lambda_u_L_inv=pinv(lambda_u_L);
    lambda_u_R_inv=pinv(lambda_u_R);
    lambda_m_D_inv=pinv(lambda_m_D);
    @tensor Tu_new[:]:=Tu_new[1,-2,2,3,-5]*lambda_u_L_inv[-1,1]*lambda_u_R_inv[-3,2]*lambda_m_D_inv[-4,3];

    lambda_r_D_inv=pinv(lambda_r_D);
    lambda_m_L_inv=pinv(lambda_m_L);
    lambda_r_U_inv=pinv(lambda_r_U);
    @tensor Tr_new[:]:=Tr_new[-1,1,2,3,-5]*lambda_r_D_inv[-2,1]*lambda_m_L_inv[-3,2]*lambda_r_U_inv[-4,3];


    return Tm_new,Tr_new,Tu_new,lambda_m_L,lambda_m_D,lambda_m_R_new,lambda_m_U_new, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U
end

function update_LU_triangle(Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U, gate, trun_tol,bond_dim)
    Tm=permute(Tm,(2,3,4,1,5,));
    Tr=permute(Tr,(2,3,4,1,5,));
    Tu=permute(Tu,(2,3,4,1,5,));

    Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U=update_RU_triangle(Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U, gate, trun_tol,bond_dim);

    Tm=permute(Tm,(4,1,2,3,5,));
    Tr=permute(Tr,(4,1,2,3,5,));
    Tu=permute(Tu,(4,1,2,3,5,));

    return Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U
end

function update_LD_triangle(Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U, gate, trun_tol,bond_dim)
    Tm=permute(Tm,(3,4,1,2,5,));
    Tr=permute(Tr,(3,4,1,2,5,));
    Tu=permute(Tu,(3,4,1,2,5,));

    Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U=update_RU_triangle(Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U, gate, trun_tol,bond_dim);

    Tm=permute(Tm,(3,4,1,2,5,));
    Tr=permute(Tr,(3,4,1,2,5,));
    Tu=permute(Tu,(3,4,1,2,5,));

    return Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U
end

function update_RD_triangle(Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U, gate, trun_tol,bond_dim)
    Tm=permute(Tm,(4,1,2,3,5,));
    Tr=permute(Tr,(4,1,2,3,5,));
    Tu=permute(Tu,(4,1,2,3,5,));

    Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U=update_RU_triangle(Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U, gate, trun_tol,bond_dim);

    Tm=permute(Tm,(2,3,4,1,5,));
    Tr=permute(Tr,(2,3,4,1,5,));
    Tu=permute(Tu,(2,3,4,1,5,));

    return Tm,Tr,Tu,lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U, lambda_u_L,lambda_u_R, lambda_r_D,lambda_r_U
end

function A_site_triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gate, trun_tol,bond_dim)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    #Tm,Tr,Tu,    lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U,               lambda_u_L,lambda_u_R,      lambda_r_D,lambda_r_U


    #RU triangle: ABC
    T_A,T_B,T_C,lambda_A_L,lambda_A_D,lambda_A_R,lambda_A_U, lambda_D_R,lambda_D_L, lambda_D_U,lambda_D_D=
    update_RU_triangle(T_A,T_B,T_C,lambda_A_L,lambda_A_D,lambda_A_R,lambda_A_U, lambda_D_R,lambda_D_L, lambda_D_U,lambda_D_D, gate, trun_tol,bond_dim);

    #LU triangle: ACB
    T_A,T_C,T_B,lambda_A_D,lambda_A_R,lambda_A_U,lambda_A_L, lambda_D_U,lambda_D_D, lambda_D_L,lambda_D_R=
    update_LU_triangle(T_A,T_C,T_B,lambda_A_D,lambda_A_R,lambda_A_U,lambda_A_L, lambda_D_U,lambda_D_D, lambda_D_L,lambda_D_R, gate, trun_tol,bond_dim);

    #LD triangle: ABC
    T_A,T_B,T_C,lambda_A_R,lambda_A_U,lambda_A_L,lambda_A_D, lambda_D_L,lambda_D_R, lambda_D_D,lambda_D_U=
    update_LD_triangle(T_A,T_B,T_C,lambda_A_R,lambda_A_U,lambda_A_L,lambda_A_D, lambda_D_L,lambda_D_R, lambda_D_D,lambda_D_U, gate, trun_tol,bond_dim);

    #RD triangle: ACB
    T_A,T_C,T_B,lambda_A_U,lambda_A_L,lambda_A_D,lambda_A_R, lambda_D_D,lambda_D_U, lambda_D_R,lambda_D_L=
    update_RD_triangle(T_A,T_C,T_B,lambda_A_U,lambda_A_L,lambda_A_D,lambda_A_R, lambda_D_D,lambda_D_U, lambda_D_R,lambda_D_L, gate, trun_tol,bond_dim);

    return T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U
end

function B_site_triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gate, trun_tol,bond_dim)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    #Tm,Tr,Tu,    lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U,               lambda_u_L,lambda_u_R,      lambda_r_D,lambda_r_U

    lambda_A_R=permute(lambda_A_R,(2,),(1,));
    lambda_D_U=permute(lambda_D_U,(2,),(1,));
    lambda_A_L=permute(lambda_A_L,(2,),(1,));
    lambda_D_D=permute(lambda_D_D,(2,),(1,));
    lambda_D_L=permute(lambda_D_L,(2,),(1,));
    lambda_D_R=permute(lambda_D_R,(2,),(1,));
    lambda_A_D=permute(lambda_A_D,(2,),(1,));
    lambda_A_U=permute(lambda_A_U,(2,),(1,));

    #RU triangle: 
    T_B,T_A,T_D,  lambda_A_R,lambda_D_U,lambda_A_L,lambda_D_D,    lambda_D_L,lambda_D_R,      lambda_A_D,lambda_A_U=
    update_RU_triangle(T_B,T_A,T_D,  lambda_A_R,lambda_D_U,lambda_A_L,lambda_D_D,    lambda_D_L,lambda_D_R,      lambda_A_D,lambda_A_U, gate, trun_tol,bond_dim);

    #LU triangle: 
    T_B,T_D,T_A,  lambda_D_U,lambda_A_L,lambda_D_D,lambda_A_R,    lambda_A_D,lambda_A_U,      lambda_D_R,lambda_D_L=
    update_LU_triangle(T_B,T_D,T_A,  lambda_D_U,lambda_A_L,lambda_D_D,lambda_A_R,    lambda_A_D,lambda_A_U,      lambda_D_R,lambda_D_L, gate, trun_tol,bond_dim);

    #LD triangle: 
    T_B,T_A,T_D,  lambda_A_L,lambda_D_D,lambda_A_R,lambda_D_U,    lambda_D_R,lambda_D_L,      lambda_A_U,lambda_A_D=
    update_LD_triangle(T_B,T_A,T_D,  lambda_A_L,lambda_D_D,lambda_A_R,lambda_D_U,    lambda_D_R,lambda_D_L,      lambda_A_U,lambda_A_D, gate, trun_tol,bond_dim);

    #RD triangle: 
    T_B,T_D,T_A,  lambda_D_D,lambda_A_R,lambda_D_U,lambda_A_L,    lambda_A_U,lambda_A_D,      lambda_D_L,lambda_D_R=
    update_RD_triangle(T_B,T_D,T_A,  lambda_D_D,lambda_A_R,lambda_D_U,lambda_A_L,    lambda_A_U,lambda_A_D,      lambda_D_L,lambda_D_R, gate, trun_tol,bond_dim);

    lambda_A_R=permute(lambda_A_R,(2,),(1,));
    lambda_D_U=permute(lambda_D_U,(2,),(1,));
    lambda_A_L=permute(lambda_A_L,(2,),(1,));
    lambda_D_D=permute(lambda_D_D,(2,),(1,));
    lambda_D_L=permute(lambda_D_L,(2,),(1,));
    lambda_D_R=permute(lambda_D_R,(2,),(1,));
    lambda_A_D=permute(lambda_A_D,(2,),(1,));
    lambda_A_U=permute(lambda_A_U,(2,),(1,));

    return T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U
end

function C_site_triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gate, trun_tol,bond_dim)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    #Tm,Tr,Tu,    lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U,               lambda_u_L,lambda_u_R,      lambda_r_D,lambda_r_U

    lambda_A_R=permute(lambda_A_R,(2,),(1,));
    lambda_D_U=permute(lambda_D_U,(2,),(1,));
    lambda_A_L=permute(lambda_A_L,(2,),(1,));
    lambda_D_D=permute(lambda_D_D,(2,),(1,));
    lambda_D_L=permute(lambda_D_L,(2,),(1,));
    lambda_D_R=permute(lambda_D_R,(2,),(1,));
    lambda_A_D=permute(lambda_A_D,(2,),(1,));
    lambda_A_U=permute(lambda_A_U,(2,),(1,));

    #RU triangle: 
    T_C,T_D,T_A,  lambda_D_R,lambda_A_U,lambda_D_L,lambda_A_D,    lambda_A_L,lambda_A_R,      lambda_D_D,lambda_D_U=
    update_RU_triangle(T_C,T_D,T_A,  lambda_D_R,lambda_A_U,lambda_D_L,lambda_A_D,    lambda_A_L,lambda_A_R,      lambda_D_D,lambda_D_U, gate, trun_tol,bond_dim);

    #LU triangle: 
    T_C,T_A,T_D,  lambda_A_U,lambda_D_L,lambda_A_D,lambda_D_R,    lambda_D_D,lambda_D_U,      lambda_A_R,lambda_A_L=
    update_LU_triangle(T_C,T_A,T_D,  lambda_A_U,lambda_D_L,lambda_A_D,lambda_D_R,    lambda_D_D,lambda_D_U,      lambda_A_R,lambda_A_L, gate, trun_tol,bond_dim);

    #LD triangle: 
    T_C,T_D,T_A,  lambda_D_L,lambda_A_D,lambda_D_R,lambda_A_U,    lambda_A_R,lambda_A_L,      lambda_D_U,lambda_D_D=
    update_LD_triangle(T_C,T_D,T_A,  lambda_D_L,lambda_A_D,lambda_D_R,lambda_A_U,    lambda_A_R,lambda_A_L,      lambda_D_U,lambda_D_D, gate, trun_tol,bond_dim);

    #RD triangle: 
    T_C,T_A,T_D,  lambda_A_D,lambda_D_R,lambda_A_U,lambda_D_L,    lambda_D_U,lambda_D_D,      lambda_A_L,lambda_A_R=
    update_RD_triangle(T_C,T_A,T_D,  lambda_A_D,lambda_D_R,lambda_A_U,lambda_D_L,    lambda_D_U,lambda_D_D,      lambda_A_L,lambda_A_R, gate, trun_tol,bond_dim);

    lambda_A_R=permute(lambda_A_R,(2,),(1,));
    lambda_D_U=permute(lambda_D_U,(2,),(1,));
    lambda_A_L=permute(lambda_A_L,(2,),(1,));
    lambda_D_D=permute(lambda_D_D,(2,),(1,));
    lambda_D_L=permute(lambda_D_L,(2,),(1,));
    lambda_D_R=permute(lambda_D_R,(2,),(1,));
    lambda_A_D=permute(lambda_A_D,(2,),(1,));
    lambda_A_U=permute(lambda_A_U,(2,),(1,));

    return T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U
end

function D_site_triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gate, trun_tol,bond_dim)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    #Tm,Tr,Tu,    lambda_m_L,lambda_m_D,lambda_m_R,lambda_m_U,               lambda_u_L,lambda_u_R,      lambda_r_D,lambda_r_U
    

    #RU triangle: ABC
    T_D,T_C,T_B,lambda_D_L,lambda_D_D,lambda_D_R,lambda_D_U, lambda_A_R,lambda_A_L, lambda_A_U,lambda_A_D=
    update_RU_triangle(T_D,T_C,T_B,lambda_D_L,lambda_D_D,lambda_D_R,lambda_D_U, lambda_A_R,lambda_A_L, lambda_A_U,lambda_A_D, gate, trun_tol,bond_dim);

    #LU triangle: ACB
    T_D,T_B,T_C,lambda_D_D,lambda_D_R,lambda_D_U,lambda_D_L, lambda_A_U,lambda_A_D, lambda_A_L,lambda_A_R=
    update_LU_triangle(T_D,T_B,T_C,lambda_D_D,lambda_D_R,lambda_D_U,lambda_D_L, lambda_A_U,lambda_A_D, lambda_A_L,lambda_A_R, gate, trun_tol,bond_dim);

    #LD triangle: ABC
    T_D,T_C,T_B,lambda_D_R,lambda_D_U,lambda_D_L,lambda_D_D, lambda_A_L,lambda_A_R, lambda_A_D,lambda_A_U=
    update_LD_triangle(T_D,T_C,T_B,lambda_D_R,lambda_D_U,lambda_D_L,lambda_D_D, lambda_A_L,lambda_A_R, lambda_A_D,lambda_A_U, gate, trun_tol,bond_dim);

    #RD triangle: ACB
    T_D,T_B,T_C,lambda_D_U,lambda_D_L,lambda_D_D,lambda_D_R, lambda_A_D,lambda_A_U, lambda_A_R,lambda_A_L=
    update_RD_triangle(T_D,T_B,T_C,lambda_D_U,lambda_D_L,lambda_D_D,lambda_D_R, lambda_A_D,lambda_A_U, lambda_A_R,lambda_A_L, gate, trun_tol,bond_dim);

    return T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U
end

function itebd_step(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, trun_tol, gate, bond_dim)
    # println("one step")
    # println(space(T_u))
    # println(space(T_d))
    
    T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U= A_site_triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gate, trun_tol,bond_dim);
    T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U= B_site_triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gate, trun_tol,bond_dim);
    T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U= C_site_triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gate, trun_tol,bond_dim);
    T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U= D_site_triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gate, trun_tol,bond_dim);
    return T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U
end

function itebd(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, H, trun_tol, tau, dt, bond_dim)
    gate, gate_half=trotter_gate(H, dt)

    T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U=itebd_step(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, trun_tol, gate_half, bond_dim)
    for cs=1:Int(round(tau/dt))
        T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U=itebd_step(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, trun_tol, gate, bond_dim)

    end
    T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U=itebd_step(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, trun_tol, gate_half, bond_dim)

    return T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U
end