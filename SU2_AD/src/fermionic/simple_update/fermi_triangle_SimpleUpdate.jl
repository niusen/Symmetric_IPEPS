using LinearAlgebra:diag,I,diagm 
###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################

function convert_to_iPEPS(Lx,Ly,T_set)
    A_cell=initial_tuple_cell(Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            A_cell=fill_tuple(A_cell, T_set[cx,cy], cx,cy);
        end
    end
    return A_cell
end

function initial_state_from_optimized(state)
    Lx=2;
    Ly=2;
    
    T_set=Matrix{TensorMap}(undef,Lx,Ly);
    lambdax_set=Matrix{TensorMap}(undef,Lx,Ly);
    lambday_set=Matrix{TensorMap}(undef,Lx,Ly);
    for ca=1:Lx
        for cb=1:Ly
            if isa(state[1],Square_iPEPS)
                if size(state)==(2,1)
                    T_set[ca,cb]=state[ca,1].T;
                end
                Vv1=space(T_set[ca,cb],1);
                Vv2=space(T_set[ca,cb],2);
                lambdax_set[ca,cb]=unitary(Vv1,Vv1);
                lambday_set[ca,cb]=unitary(Vv2',Vv2');
            end
        end
    end
    return T_set,lambdax_set,lambday_set
end

function initial_iPEPS_Z2(Lx,Ly,Vp,Vv)
    lambdax_set=Matrix{Any}(undef,Lx,Ly);#to the left of site (ca,cb) 
    lambday_set=Matrix{Any}(undef,Lx,Ly);#to the bot of site (ca,cb)
    T_set=Matrix{Any}(undef,Lx,Ly)

    for ca=1:Lx
        for cb=1:Ly
            T=TensorMap(randn,Vv*Vv*Vv'*Vv',Vp');
            T=permute(T,(1,2,3,4,5,));
            T_set[ca,cb]=T;
        end
    end

    for ca=1:Lx
        for cb=1:Ly
            lambdax_set[ca,cb]=unitary(Vv,Vv);
            lambday_set[ca,cb]=unitary(Vv',Vv');
        end
    end

    return T_set,lambdax_set,lambday_set
end

function initial_iPEPS_SU2(Lx,Ly,Vp,Vv)
    lambdax_set=Matrix{Any}(undef,Lx,Ly);#to the left of site (ca,cb) 
    lambday_set=Matrix{Any}(undef,Lx,Ly);#to the bot of site (ca,cb)
    T_set=Matrix{Any}(undef,Lx,Ly)

    for ca=1:Lx
        for cb=1:Ly
            T=TensorMap(randn,Vv*Vv*Vv'*Vv',Vp');
            T=permute(T,(1,2,3,4,5,));
            T_set[ca,cb]=T;
        end
    end

    for ca=1:Lx
        for cb=1:Ly
            lambdax_set[ca,cb]=unitary(Vv,Vv);
            lambday_set[ca,cb]=unitary(Vv',Vv');
        end
    end

    return T_set,lambdax_set,lambday_set
end

function initial_iPEPS_U1_SU2(Lx,Ly,Vp,Vv_set)
    global VDummy_set
    VDummy1=VDummy_set[1];
    VDummy2=VDummy_set[2];
    lambdax_set=Matrix{Any}(undef,Lx,Ly);#to the left of site (ca,cb) 
    lambday_set=Matrix{Any}(undef,Lx,Ly);#to the bot of site (ca,cb)
    T_set=Matrix{Any}(undef,Lx,Ly)

    for ca=1:Lx
        for cb=1:Ly
            if mod1(ca,2)==1
                Vp_=fuse(Vp*VDummy1);
            elseif mod1(ca,2)==2
                Vp_=fuse(Vp*VDummy2);
            end
            T=TensorMap(randn,Vv_set[ca][1]*Vv_set[ca][2]*Vv_set[ca][3]*Vv_set[ca][4],Vp_');
            T=permute(T,(1,2,3,4,5,));
            T_set[ca,cb]=T;
        end
    end

    for ca=1:Lx
        for cb=1:Ly
            lambdax_set[ca,cb]=unitary(Vv_set[ca][1],Vv_set[ca][1]);
            lambday_set[ca,cb]=unitary(Vv_set[ca][2]',Vv_set[ca][2]');
        end
    end

    return T_set,lambdax_set,lambday_set
end


function Rank(T::TensorMap)
    return length(domain(T))+length(codomain(T))
end


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
    if isa(space(Σ,1),ComplexSpace)
        # pos=findall(diag(Σ).>0);

        # if Rank(U)==6
        # end
        # if Rank(V)==3
        # end

        # println(space(V))
        return U,Σ,V
    else
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



function evo_hopping_RU_LD_RD(op_LD_RD_RU, A_RU0, A_LD0, A_RD0, Dmax)

    function move_RU(T)
        T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
        T=permute_neighbour_ind(T,3,4,5);#L,U,R,  d,D,

        U,S,V=tsvd(T,(1,2,3,),(4,5,));
        RU_res=U;#(L_ru,U_ru,R_ru, virtual_ru)
        RU_keep=S*V; #(virtual_ru, d_ru,D_ru)
        return RU_res, RU_keep
    end
    function move_LD(T)
        T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
        T=permute_neighbour_ind(T,4,5,5);#L,U,d,D,R,
        T=permute_neighbour_ind(T,3,4,5);#L,U,D,  d,R,

        U,S,V=tsvd(T,(1,2,3,),(4,5,));
        LD_res=U;#(L_ld,U_ld,D_ld, virtual_ld)
        LD_keep=S*V;#(virtual_ld, d_ld,R_ld)
        return LD_res,LD_keep

    end

    function update_RD(T_RU,T_LD, T_RD, op)
        RU_res, RU_keep=move_RU(T_RU);
        LD_res,LD_keep=move_LD(T_LD);

        T_RD=permute(T_RD,(1,4,5,3,2,));#L_rd,U_rd,d_rd,R_rd,D_rd,
        @tensor T_LD_RD[:]:=LD_keep[-1,-2,1]*T_RD[1,-3,-4,-5,-6];#(virtual_ld, d_ld,     U_rd,d_rd,R_rd,D_rd,)
        T_LD_RD=permute_neighbour_ind(T_LD_RD,3,4,6);#(virtual_ld, d_ld,   d_rd,U_rd,R_rd,D_rd,)
        T_LD_RD=permute_neighbour_ind(T_LD_RD,4,5,6);#(virtual_ld, d_ld,   d_rd,R_rd,U_rd,D_rd,)
        T_LD_RD=permute_neighbour_ind(T_LD_RD,5,6,6);#(virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd,)
        U_RD=unitary(fuse(space(T_LD_RD,4)*space(T_LD_RD,5)), space(T_LD_RD,4)*space(T_LD_RD,5));
        @tensor T_LD_RD[:]:=T_LD_RD[-1,-2,-3,1,2,-5]*U_RD[-4,1,2];#(virtual_ld, d_ld,   d_rd,R_D_rd,U_rd,)
        T_LD_RD=permute_neighbour_ind(T_LD_RD,3,4,5);#(virtual_ld, d_ld,   R_D_rd,d_rd,U_rd,)
        T_LD_RD=permute_neighbour_ind(T_LD_RD,2,3,5);#(virtual_ld,  R_D_rd, d_ld, d_rd, U_rd,)

        RU_keep=permute_neighbour_ind(RU_keep,2,3,3);#(virtual_ru, D_ru, d_ru)
        RU_keep=permute_neighbour_ind(RU_keep,1,2,3);#(D_ru, virtual_ru,  d_ru)
        RU_keep=permute_neighbour_ind(RU_keep,2,3,3);#(D_ru, d_ru, virtual_ru)
        gate=parity_gate(RU_keep,1);
        @tensor RU_keep[:]:=RU_keep[1,-2,-3]*gate[-1,1];
        @tensor T_RU_LD_RD[:]:=T_LD_RD[-1,-2,-3,-4,1]*RU_keep[1,-5,-6];# (virtual_ld,  R_D_rd, d_ld, d_rd, U_rd,)    (D_ru, d_ru, virtual_ru)

        @tensor T_RU_LD_RD[:]:=T_RU_LD_RD[-1,-2,1,2,3,-6]*op[-3,-4,-5,1,2,3];
        return RU_res,LD_res,T_RU_LD_RD, U_RD
    end

    function back_RD(T_RU_LD_RD, U_RD)
        #######################
        U1,S1,V1=tsvd(permute(T_RU_LD_RD,(1,2,3,4,),(5,6,)); trunc=truncdim(Dmax));#(virtual_ld,  R_D_rd, d_ld, d_rd, U_rd_new),  (D_ru_new, d_ru, virtual_ru) 
        S1=S1/norm(S1);
        #######################
        RU_keep=permute(sqrt(S1)*V1,(1,2,3,));#(D_ru_new, d_ru, virtual_ru)
        gate=parity_gate(RU_keep,1);
        @tensor RU_keep[:]:=RU_keep[1,-2,-3]*gate[-1,1];
        RU_keep=permute_neighbour_ind(RU_keep,2,3,3);#(D_ru_new,  virtual_ru, d_ru)
        RU_keep=permute_neighbour_ind(RU_keep,1,2,3);#(virtual_ru, D_ru_new, d_ru)
        RU_keep=permute_neighbour_ind(RU_keep,2,3,3);#(virtual_ru, d_ru, D_ru_new)

        T_LD_RD=permute(U1*S1,(1,2,3,4,5,));#(virtual_ld,  R_D_rd, d_ld, d_rd, U_rd_new) 
        T_LD_RD=permute_neighbour_ind(T_LD_RD,2,3,5);#(virtual_ld,  d_ld, R_D_rd,  d_rd, U_rd_new) 
        T_LD_RD=permute_neighbour_ind(T_LD_RD,3,4,5);#(virtual_ld,  d_ld, d_rd, R_D_rd,  U_rd_new) 
        @tensor T_LD_RD[:]:=T_LD_RD[-1,-2,-3,1,-6]*U_RD'[-4,-5,1];#(virtual_ld, d_ld,  d_rd, R_rd, D_rd,U_rd_new,)
        T_LD_RD=permute_neighbour_ind(T_LD_RD,5,6,6);#(virtual_ld, d_ld,  d_rd,R_rd,U_rd_new,D_rd,) 
        T_LD_RD=permute_neighbour_ind(T_LD_RD,4,5,6);#(virtual_ld, d_ld,  d_rd,U_rd_new,R_rd,D_rd,) 
        T_LD_RD=permute_neighbour_ind(T_LD_RD,3,4,6);#(virtual_ld, d_ld,  U_rd_new,d_rd,R_rd,D_rd,) 
        ###################
        U3,S3,V3=tsvd(permute(T_LD_RD,(1,2,),(3,4,5,6,)); trunc=truncdim(Dmax));
        S3=S3/norm(S3);
        ###################
        LD_keep=permute(U3*sqrt(S3),(1,2,3,));#(virtual_ld, d_ld, R_ld)
        T_RD=permute(sqrt(S3)*V3,(1,2,3,4,5,));#(L_rd, U_rd,d_rd,R_rd,D_rd,) 
        T_RD=permute(T_RD,(1,5,4,2,3,));

        S1_sqrt_inv=sqrt(my_pinv(S1));
        @tensor T_RD[:]:=T_RD[-1,-2,-3,1,-5]*S1_sqrt_inv[1,-4];
        lambda_RU_RD=S1;
        lambda_LD_RD=S3;
        return RU_keep,LD_keep, T_RD, lambda_RU_RD, lambda_LD_RD
    end

    function back_LD(LD_res,LD_keep)
        #(L_ld,U_ld,D_ld, virtual_ld)
        #(virtual_ld, d_ld, R_ld)
        @tensor T_LD[:]:=LD_res[-1,-2,-3,1]*LD_keep[1,-4,-5];#(L_ld,U_ld,D_ld, d_ld, R_ld)
        T_LD=permute_neighbour_ind(T_LD,3,4,5);#(L,U, d,D, R)
        T_LD=permute_neighbour_ind(T_LD,4,5,5);#(L,U, d,R, D)
        T_LD=permute(T_LD,(1,5,4,2,3,));
        return T_LD
    end

    function back_RU(RU_res,RU_keep)
        #(L_ru,U_ru,R_ru, virtual_ru)
        #(virtual_ru, d_ru, D_ru)
        @tensor T_RU[:]:=RU_res[-1,-2,-3,1]*RU_keep[1,-4,-5];#(L_ru,U_ru,R_ru,  d_ru, D_ru)
        T_RU=permute_neighbour_ind(T_RU,3,4,5);#(L,U,d, R, D)
        T_RU=permute(T_RU,(1,5,4,2,3,));
        return T_RU
    end

    RU_res,LD_res,A_RU_LD_RD, U_RD=update_RD(A_RU0, A_LD0, A_RD0, op_LD_RD_RU);
    RU_keep,LD_keep, A_RD, lambda_RU_RD, lambda_LD_RD=back_RD(A_RU_LD_RD, U_RD);
    A_LD=back_LD(LD_res,LD_keep);
    A_RU=back_RU(RU_res,RU_keep);
    return A_RU, A_LD, A_RD, lambda_RU_RD, lambda_LD_RD
end





function evo_hopping_LU_RU_LD(op_LD_LU_RU, A_RU0, A_LD0, A_LU0, Dmax)

    function move_RU(T)
        T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
        T=permute_neighbour_ind(T,2,3,5);#L,d,U,R,D,

        U,S,V=tsvd(T,(1,2,),(3,4,5,));
        RU_res=V;#(virtual_ru, U_ru,R_ru,D_ru)
        RU_keep=U*S; #(L_ru,d_ru, virtual_ru)
        return RU_res, RU_keep
    end
    function move_LD(T)
        T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
        T=permute_neighbour_ind(T,3,4,5);#L,U,R,d,D,
        T=permute_neighbour_ind(T,4,5,5);#L,U,R,D,d,
        T=permute_neighbour_ind(T,2,3,5);#L,R,U,D,d,
        T=permute_neighbour_ind(T,3,4,5);#L,R,D,U,d,
        T=permute_neighbour_ind(T,4,5,5);#L,R,D,  d,U,

        U,S,V=tsvd(T,(1,2,3,),(4,5,));
        LD_res=U;#(L_ld,R_ld,D_ld, virtual_ld)
        LD_keep=S*V;#(virtual_ld, d_ld,U_ld)
        return LD_res,LD_keep

    end

    function update_LU(T_RU,T_LD, T_LU, op)
        RU_res, RU_keep=move_RU(T_RU);
        LD_res,LD_keep=move_LD(T_LD);

        T_LU=permute(T_LU,(1,4,5,3,2,));#L_lu,U_lu,d_lu,R_lu,D_lu,
        T_LU=permute_neighbour_ind(T_LU,4,5,5);#L_lu,U_lu,d_lu,D_lu,R_lu,
        @tensor T_LU_RU[:]:=T_LU[-1,-2,-3,-4,1]*RU_keep[1,-5,-6];#(L_lu,U_lu,d_lu,D_lu,R_lu,     L_ru,d_ru, virtual_ru)
        U_LU=unitary(fuse(space(T_LU_RU,1)*space(T_LU_RU,2)), space(T_LU_RU,1)*space(T_LU_RU,2));
        @tensor T_LU_RU[:]:=T_LU_RU[1,2,-2,-3,-4,-5]*U_LU[-1,1,2];#(L_U_lu, d_lu, D_lu, d_ru, virtual_ru)
        T_LU_RU=permute_neighbour_ind(T_LU_RU,2,3,5);#(L_U_lu, D_lu, d_lu, d_ru, virtual_ru)
        T_LU_RU=permute_neighbour_ind(T_LU_RU,1,2,5);#(D_lu, L_U_lu, d_lu, d_ru, virtual_ru)
        T_LU_RU=permute_neighbour_ind(T_LU_RU,2,3,5);#(D_lu, d_lu, L_U_lu, d_ru, virtual_ru)
        T_LU_RU=permute_neighbour_ind(T_LU_RU,3,4,5);#(D_lu, d_lu, d_ru, L_U_lu, virtual_ru)

        gate=parity_gate(LD_keep,3);
        @tensor LD_keep[:]:=LD_keep[-1,-2,1]*gate[-3,1];
        @tensor T_LU_RU_LD[:]:=LD_keep[-1,-2,1]*T_LU_RU[1,-3,-4,-5,-6];# (virtual_ld, d_ld,U_ld)  (D_lu, d_lu, d_ru, L_U_lu, virtual_ru)
        # T_LU_RU_LD=permute_neighbour_ind(T_LU_RU_LD,3,4,6);# (virtual_ld, d_ld, d_ru, d_lu, L_U_lu, virtual_ru)

        @tensor T_LU_RU_LD[:]:=T_LU_RU_LD[-1,1,2,3,-5,-6]*op[-2,-3,-4,1,2,3];
        # T_LU_RU_LD=permute_neighbour_ind(T_LU_RU_LD,3,4,6);# (virtual_ld, d_ld, d_lu, d_ru, L_U_lu, virtual_ru)
        return RU_res,LD_res,T_LU_RU_LD, U_LU
    end

    function back_LU(T_LU_RU_LD, U_LU)
        #######################
        U1,S1,V1=tsvd(permute(T_LU_RU_LD,(1,2,),(3,4,5,6,)); trunc=truncdim(Dmax));#(virtual_ld, d_ld,U_ld_new)  (D_lu_new, d_lu, d_ru, L_U_lu, virtual_ru)
        S1=S1/norm(S1);
        #######################
        LD_keep=permute(U1*sqrt(S1),(1,2,3,));#(virtual_ld, d_ld,U_ld_new)
        gate=parity_gate(LD_keep,3);
        @tensor LD_keep[:]:=LD_keep[-1,-2,1]*gate[-3,1];

        
        T_LU_RU=permute(S1*V1,(1,2,3,4,5,));#(D_lu_new, d_lu, d_ru, L_U_lu, virtual_ru) 
        T_LU_RU=permute_neighbour_ind(T_LU_RU,3,4,5);#(D_lu_new, d_lu, L_U_lu, d_ru, virtual_ru) 
        @tensor T_LU_RU[:]:=T_LU_RU[-1,-2,1,-5,-6]*U_LU'[-3,-4,1];#(D_lu_new, d_lu, L_lu, U_lu, d_ru, virtual_ru)
        ###################
        U3,S3,V3=tsvd(permute(T_LU_RU,(1,2,3,4,),(5,6,)); trunc=truncdim(Dmax));#(D_lu_new, d_lu, L_lu, U_lu, R_lu_new) (L_ru_new, d_ru, virtual_ru)
        S3=S3/norm(S3);
        ###################
        RU_keep=permute(sqrt(S3)*V3,(1,2,3,));#(L_ru_new, d_ru, virtual_ru)
        T_LU=permute(U3*sqrt(S3),(1,2,3,4,5,));#(D_lu_new, d_lu, L_lu, U_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,2,3,5); #(D_lu_new, L_lu, d_lu, U_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,1,2,5); #(L_lu, D_lu_new, d_lu, U_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,3,4,5); #(L_lu, D_lu_new, U_lu, d_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,2,3,5); #(L_lu, U_lu, D_lu_new, d_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,3,4,5); #(L_lu, U_lu, d_lu, D_lu_new, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,4,5,5); #(L_lu, U_lu, d_lu,, R_lu_new D_lu_new) 
        T_LU=permute(T_LU,(1,5,4,2,3,));

        
        S1_sqrt_inv=sqrt(my_pinv(S1));
        @tensor T_LU[:]:=T_LU[-1,1,-3,-4,-5]*S1_sqrt_inv[-2,1];
        lambda_LD_LU=S1;
        lambda_LU_RU=S3;
        return LD_keep, RU_keep, T_LU, lambda_LD_LU, lambda_LU_RU
    end

    function back_LD(LD_res,LD_keep)
        #(L_ld,R_ld,D_ld, virtual_ld)
        #(virtual_ld, d_ld,U_ld_new)
        @tensor T_LD[:]:=LD_res[-1,-2,-3,1]*LD_keep[1,-4,-5];#(L_ld,R_ld,D_ld, d_ld,U_ld_new)
        T_LD=permute_neighbour_ind(T_LD,4,5,5);#(L, R, D, U, d)
        T_LD=permute_neighbour_ind(T_LD,3,4,5);#(L, R, U, D, d)
        T_LD=permute_neighbour_ind(T_LD,2,3,5);#(L, U, R, D, d)
        T_LD=permute_neighbour_ind(T_LD,4,5,5);#(L, U, R, d, D)
        T_LD=permute_neighbour_ind(T_LD,3,4,5);#(L, U, d, R, D)
        T_LD=permute(T_LD,(1,5,4,2,3,));
        return T_LD
    end

    function back_RU(RU_res,RU_keep)
        #(virtual_ru, U_ru,R_ru,D_ru)
        #(L_ru_new, d_ru, virtual_ru)
        @tensor T_RU[:]:=RU_keep[-1,-2,1]*RU_res[1,-3,-4,-5];#(L_ru_new, d_ru, U_ru,R_ru,D_ru)
        T_RU=permute_neighbour_ind(T_RU,2,3,5);#(L, U, d, R, D)
        T_RU=permute(T_RU,(1,5,4,2,3,));
        return T_RU
    end

    RU_res,LD_res,T_LU_RU_LD, U_LU=update_LU(A_RU0, A_LD0, A_LU0, op_LD_LU_RU);
    LD_keep,RU_keep, A_LU, lambda_LD_LU, lambda_LU_RU=back_LU(T_LU_RU_LD, U_LU);
    A_LD=back_LD(LD_res,LD_keep);
    A_RU=back_RU(RU_res,RU_keep);
    return A_RU, A_LD, A_LU, lambda_LD_LU, lambda_LU_RU
end



function triangle_update(ct,Tset, lambdaxset,lambdayset,  gates_ru_ld_rd, gates_lu_ru_ld, Dmax)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    Lx,Ly=size(Tset);
    global update_triangle1, update_triangle2

for ca=1:Lx
    for cb=1:Ly
        if update_triangle1
            # B
            #CD
            pos1=[mod1(ca+1,Lx),mod1(cb+1,Ly)];
            pos2=[mod1(ca+1,Lx),mod1(cb,Ly)];
            pos3=[mod1(ca,Lx),mod1(cb,Ly)];

            T1=Tset[pos1[1],pos1[2]];
            T2=Tset[pos2[1],pos2[2]];
            T3=Tset[pos3[1],pos3[2]];
            λ1=lambdaxset[pos1[1],pos1[2]];
            λ2=lambdaxset[mod1(pos1[1]+1,Lx),pos1[2]];
            λ3=lambdayset[pos1[1],mod1(pos1[2]+1,Ly)];
            λ4=lambdayset[pos2[1],pos2[2]];
            λ5=lambdaxset[mod1(pos2[1]+1,Lx),pos2[2]];
            λ6=lambdaxset[pos3[1],pos3[2]];
            λ7=lambdayset[pos3[1],pos3[2]];
            λ8=lambdayset[pos3[1],mod1(pos3[2]+1,Ly)];
    
            @tensor T1[:]:=T1[1,-2,3,4,-5]*λ1[-1,1]*λ2[3,-3]*λ3[-4,4];
            @tensor T2[:]:=T2[-1,2,3,-4,-5]*λ4[2,-2]*λ5[3,-3];
            @tensor T3[:]:=T3[1,2,-3,4,-5]*λ6[-1,1]*λ7[2,-2]*λ8[-4,4];
            A_RU=T1;
            A_LD=T3;
            A_RD=T2;
            A_RU,A_LD,A_RD,s1,s3=evo_hopping_RU_LD_RD(gates_ru_ld_rd[mod1(ca,2)], A_RU,A_LD,A_RD, Dmax);
            T1_new=A_RU;
            T3_new=A_LD;
            T2_new=A_RD;

            λ1_inv=my_pinv(λ1);
            λ2_inv=my_pinv(λ2);
            λ3_inv=my_pinv(λ3);
            λ4_inv=my_pinv(λ4);
            λ5_inv=my_pinv(λ5);
            λ6_inv=my_pinv(λ6);
            λ7_inv=my_pinv(λ7);
            λ8_inv=my_pinv(λ8);
            @tensor T1_new[:]:=T1_new[1,-2,3,4,-5]*λ1_inv[-1,1]*λ2_inv[3,-3]*λ3_inv[-4,4];
            @tensor T2_new[:]:=T2_new[-1,2,3,-4,-5]*λ4_inv[2,-2]*λ5_inv[3,-3];
            @tensor T3_new[:]:=T3_new[1,2,-3,4,-5]*λ6_inv[-1,1]*λ7_inv[2,-2]*λ8_inv[-4,4];
    
            s1=s1/norm(s1);
            s3=s3/norm(s3);
            lambdayset[pos1[1],pos1[2]]=permute(sqrt(s1),(2,),(1,));
            lambdaxset[pos2[1],pos2[2]]=sqrt(s3);

            T1_new=T1_new/norm(T1_new);
            T2_new=T2_new/norm(T2_new);
            T3_new=T3_new/norm(T3_new);
            Tset[pos1[1],pos1[2]]=T1_new;
            Tset[pos2[1],pos2[2]]=T2_new;
            Tset[pos3[1],pos3[2]]=T3_new;
        end
        #############################
        if update_triangle2
            #AB
            #C
            pos1=[mod1(ca,Lx),mod1(cb,Ly)];
            pos2=[mod1(ca,Lx),mod1(cb+1,Ly)];
            pos3=[mod1(ca+1,Lx),mod1(cb+1,Ly)];
            T1=T_set[pos1[1],pos1[2]];
            T2=T_set[pos2[1],pos2[2]];
            T3=T_set[pos3[1],pos3[2]];
            λ1=lambdax_set[pos1[1],pos1[2]];
            λ2=lambday_set[pos1[1],pos1[2]];
            λ3=lambdax_set[mod1(pos1[1]+1,Lx),pos1[2]];
            λ4=lambdax_set[pos2[1],pos2[2]];
            λ5=lambday_set[pos2[1],mod1(pos2[2]+1,Ly)];
            λ6=lambday_set[pos3[1],pos3[2]];
            λ7=lambdax_set[mod1(pos3[1]+1,Lx),pos3[2]];
            λ8=lambday_set[pos3[1],mod1(pos3[2]+1,Ly)];

            @tensor T1[:]:=T1[1,2,3,-4,-5]*λ1[-1,1]*λ2[2,-2]*λ3[3,-3];
            @tensor T2[:]:=T2[1,-2,-3,4,-5]*λ4[-1,1]*λ5[-4,4];
            @tensor T3[:]:=T3[-1,2,3,4,-5]*λ6[2,-2]*λ7[3,-3]*λ8[-4,4];

            A_LD=T1;
            A_LU=T2;
            A_RU=T3;
            A_RU, A_LD, A_LU, s1,s3=evo_hopping_LU_RU_LD(gates_lu_ru_ld[mod1(ca,2)], A_RU, A_LD, A_LU, Dmax);
            T1_new=A_LD;
            T2_new=A_LU;
            T3_new=A_RU;

            λ1_inv=my_pinv(λ1);
            λ2_inv=my_pinv(λ2);
            λ3_inv=my_pinv(λ3);
            λ4_inv=my_pinv(λ4);
            λ5_inv=my_pinv(λ5);
            λ6_inv=my_pinv(λ6);
            λ7_inv=my_pinv(λ7);
            λ8_inv=my_pinv(λ8);
            @tensor T1_new[:]:=T1_new[1,2,3,-4,-5]*λ1_inv[-1,1]*λ2_inv[2,-2]*λ3_inv[3,-3];
            @tensor T2_new[:]:=T2_new[1,-2,-3,4,-5]*λ4_inv[-1,1]*λ5_inv[-4,4];
            @tensor T3_new[:]:=T3_new[-1,2,3,4,-5]*λ6_inv[2,-2]*λ7_inv[3,-3]*λ8_inv[-4,4];

            s1=s1/norm(s1);
            s3=s3/norm(s3);
            lambday_set[pos2[1],pos2[2]]=permute(sqrt(s1),(2,),(1,));
            lambdax_set[pos3[1],pos3[2]]=sqrt(s3);

            T1_new=T1_new/norm(T1_new);
            T2_new=T2_new/norm(T2_new);
            T3_new=T3_new/norm(T3_new);
            T_set[pos1[1],pos1[2]]=T1_new;
            T_set[pos2[1],pos2[2]]=T2_new;
            T_set[pos3[1],pos3[2]]=T3_new;

        end

        if mod(ct,20)==0
            println(space(s1))
            println(space(s3))
        end
    end
end




    return Tset, lambdaxset,lambdayset
end

function gate_RU_LD_RD_Hofstadter(energy_setting,parameters,dt, space_type,Lx,Ly)
    @assert mod(Lx,energy_setting.Magnetic_cell)==0;
    if space_type==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
    elseif space_type==GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
        if mod(energy_setting.Magnetic_cell,2)==1 #odd number of sites in unitcell
            @assert mod(Ly,2)==0;
            #if use U1 symmetry, use different dummy physical space along y direction along Ly, where Ly should be even number
        end
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
    elseif space_type==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
        if (energy_setting.model == "Triangle_Hofstadter_Hubbard")|(energy_setting.model == "spinful_triangle_lattice")
            Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_Z2();
        elseif (energy_setting.model == "Triangle_Hofstadter_spinless")
            Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=Hamiltonians_spinless_Z2();
        end
    end
    
    pasrmeters_site=@ignore_derivatives get_Hofstadter_coefficients(Lx,Ly,parameters,energy_setting);
    tx_coe_set=pasrmeters_site["tx_coe_set"]/2;
    ty_coe_set=pasrmeters_site["ty_coe_set"]/2;
    t2_coe_set=pasrmeters_site["t2_coe_set"]/2;
    U_coe_set=pasrmeters_site["U_coe_set"]/6;
    μ_coe_set=pasrmeters_site["μ_coe_set"]/6;


    gate_set=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx
        for cy=1:Ly
            ####################
            O1=Cdag_set[mod1(cy+1,2)];
            O2=C_set[mod1(cy+1,2)];
            @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
            op=op*tx_coe_set[cx,cy];
            op=permute(op,(1,2,),(3,4,));
            hh=op+op';
            Id=unitary(space(Cdag_set[mod1(cy,2)],2),space(Cdag_set[mod1(cy,2)],2));
            @tensor hh_tx[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
            ######################
            O1=Cdag_set[mod1(cy,2)];
            O2=C_set[mod1(cy+1,2)];
            @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
            op=op*ty_coe_set[cx,cy]';#be careful about the order of sites here
            op=permute(op,(1,2,),(3,4,));
            hh=op+op';
            Id=unitary(space(Cdag_set[mod1(cy+1,2)],2),space(Cdag_set[mod1(cy+1,2)],2));
            @tensor hh_ty[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
            #####################
            O1=Cdag_set[mod1(cy+1,2)];
            O2=C_set[mod1(cy,2)];
            @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
            op=-op;#!!!!!!! somehow this minus sign is required
            op=op*t2_coe_set[cx,cy];
            op=permute(op,(1,2,),(3,4,));
            hh=op+op';
            Id=unitary(space(Cdag_set[mod1(cy+1,2)],2),space(Cdag_set[mod1(cy+1,2)],2));
            @tensor hh[:]:=hh[-1,-3,-4,-6]*Id[-2,-5];
            sgate=swap_gate(hh,2,3);
            @tensor hh_t2[:]:=sgate[-2,-3,1,2]*hh[-1,1,2,-4,3,4]*sgate'[3,4,-5,-6];
            #################
            OU_LD=n_double_set[mod1(cy+1,2)]-(1/2)*N_occu_set[mod1(cy+1,2)]+(1/4)*Ident_set[mod1(cy+1,2)];
            OU_RU=n_double_set[mod1(cy,2)]-(1/2)*N_occu_set[mod1(cy,2)]+(1/4)*Ident_set[mod1(cy,2)];
            OU_RD=n_double_set[mod1(cy+1,2)]-(1/2)*N_occu_set[mod1(cy+1,2)]+(1/4)*Ident_set[mod1(cy+1,2)];
            Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
            Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
            Id_RD=unitary(space(OU_RD,1),space(OU_RD,1));
            @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
            @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
            @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
            hh_U=(hh_LD+hh_RU+hh_RD)*U_coe_set[cx,cy];
            #################
            OU_LD=N_occu_set[mod1(cy+1,2)];
            OU_RU=N_occu_set[mod1(cy,2)];
            OU_RD=N_occu_set[mod1(cy+1,2)];
            Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
            Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
            Id_RD=unitary(space(OU_RD,1),space(OU_RD,1));
            @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
            @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
            @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
            hh_μ=(hh_LD+hh_RU+hh_RD)*μ_coe_set[cx,cy];
            #################
            hh=permute(hh_tx+hh_ty+hh_t2+hh_U-hh_μ,(1,2,3,),(4,5,6,));
            eu,ev=eigh(hh);
            gate=ev*exp(-dt*eu)*ev';
            gate_set[cx,cy]=gate;
        end
    end
    return gate_set
end

function gate_RU_LD_RD(parameters,dt, space_type,Lx)

    if space_type==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
    elseif space_type==GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
    elseif space_type==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_Z2();
    end
    
    t1=parameters["t1"];
    t2=parameters["t2"];
    ϕ=parameters["ϕ"];
    U=parameters["U"];

    tx_coe_set=[exp(im*ϕ),exp(im*ϕ)]*t1/2;
    # ty_coe_set=[-1,1]*t1/2;
    # t2_coe_set=[-1,1]*t2/2;
    ty_coe_set=[1,-1]*t1/2;
    t2_coe_set=[1,-1]*t2/2;
    U_coe=U/6;

    gate_set=Matrix{TensorMap}(undef,2,1);
    for cx=1:2;
        ####################
        # O1=Cdag_set[mod1(cx,Lx)];
        # O2=C_set[mod1(cx+1,Lx)];
        O1=Cdag_set[mod1(cx,2)];
        O2=C_set[mod1(cx+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*tx_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh_tx[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        ######################
        # O1=Cdag_set[mod1(cx+1,Lx)];
        # O2=C_set[mod1(cx+1,Lx)];
        O1=Cdag_set[mod1(cx+1,2)];
        O2=C_set[mod1(cx+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*ty_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        # Id=unitary(space(Cdag_set[mod1(cx,Lx)],2),space(Cdag_set[mod1(cx,Lx)],2));
        Id=unitary(space(Cdag_set[mod1(cx,2)],2),space(Cdag_set[mod1(cx,2)],2));
        @tensor hh_ty[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        #####################
        # O1=Cdag_set[mod1(cx,Lx)];
        # O2=C_set[mod1(cx+1,Lx)];
        O1=Cdag_set[mod1(cx,2)];
        O2=C_set[mod1(cx+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=-op;#!!!!!!! somehow this minus sign is required
        op=op*t2_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh[:]:=hh[-1,-3,-4,-6]*Id[-2,-5];
        sgate=swap_gate(hh,2,3);
        @tensor hh_t2[:]:=sgate[-2,-3,1,2]*hh[-1,1,2,-4,3,4]*sgate'[3,4,-5,-6];
        #################
        # OU_LD=n_double_set[mod1(cx,Lx)]-(1/2)*N_occu_set[mod1(cx,Lx)]+(1/4)*Ident_set[mod1(cx,Lx)];
        # OU_RU=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        # OU_RD=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        OU_LD=n_double_set[mod1(cx,2)]-(1/2)*N_occu_set[mod1(cx,2)]+(1/4)*Ident_set[mod1(cx,2)];
        OU_RU=n_double_set[mod1(cx+1,2)]-(1/2)*N_occu_set[mod1(cx+1,2)]+(1/4)*Ident_set[mod1(cx+1,2)];
        OU_RD=n_double_set[mod1(cx+1,2)]-(1/2)*N_occu_set[mod1(cx+1,2)]+(1/4)*Ident_set[mod1(cx+1,2)];
        Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
        Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
        Id_RD=unitary(space(OU_RD,1),space(OU_RD,1));
        @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
        @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
        @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
        hh_U=(hh_LD+hh_RU+hh_RD)*U_coe;
        
        #################
        hh=permute(hh_tx+hh_ty+hh_t2+hh_U,(1,2,3,),(4,5,6,));#hh_tx+hh_ty+hh_t2+hh_U
        eu,ev=eigh(hh);
        gate=ev*exp(-dt*eu)*ev';
        gate_set[cx]=gate;
    end
    return gate_set
end


function gate_LU_RU_LD(parameters,dt, space_type, Lx)

    if space_type==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
    elseif space_type==GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
    elseif space_type==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_Z2();
    end
    t1=parameters["t1"];
    t2=parameters["t2"];
    ϕ=parameters["ϕ"];
    U=parameters["U"];

    tx_coe_set=[exp(im*ϕ),exp(im*ϕ)]*t1/2;
    ty_coe_set=[-1,1]*t1/2;
    t2_coe_set=[1,-1]*t2/2;
    U_coe=U/6;
    
    gate_set=Matrix{TensorMap}(undef,2,1);
    for cx=1:2
        ####################
        # O1=Cdag_set[mod1(cx,Lx)];
        # O2=C_set[mod1(cx+1,Lx)];
        O1=Cdag_set[mod1(cx,2)];
        O2=C_set[mod1(cx+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*tx_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,1),space(hh,1));
        @tensor hh_tx[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        ######################
        # O1=Cdag_set[mod1(cx,Lx)];
        # O2=C_set[mod1(cx,Lx)];
        O1=Cdag_set[mod1(cx,2)];
        O2=C_set[mod1(cx,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*ty_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        # Id=unitary(space(Cdag_set[mod1(cx+1,Lx)],2),space(Cdag_set[mod1(cx+1,Lx)],2));
        Id=unitary(space(Cdag_set[mod1(cx+1,2)],2),space(Cdag_set[mod1(cx+1,2)],2));
        @tensor hh_ty[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        #####################
        # O1=Cdag_set[mod1(cx,Lx)];
        # O2=C_set[mod1(cx+1,Lx)];
        O1=Cdag_set[mod1(cx,2)];
        O2=C_set[mod1(cx+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=-op;#!!!!!!! somehow this minus sign is required
        op=op*t2_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,1),space(hh,1));
        @tensor hh[:]:=hh[-1,-3,-4,-6]*Id[-2,-5];
        sgate=swap_gate(hh,2,3);
        @tensor hh_t2[:]:=sgate[-2,-3,1,2]*hh[-1,1,2,-4,3,4]*sgate'[3,4,-5,-6];##op_LD_LU_RU
        #################
        # OU_LD=n_double_set[mod1(cx,Lx)]-(1/2)*N_occu_set[mod1(cx,Lx)]+(1/4)*Ident_set[mod1(cx,Lx)];
        # OU_RU=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        # OU_LU=n_double_set[mod1(cx,Lx)]-(1/2)*N_occu_set[mod1(cx,Lx)]+(1/4)*Ident_set[mod1(cx,Lx)];
        OU_LD=n_double_set[mod1(cx,2)]-(1/2)*N_occu_set[mod1(cx,2)]+(1/4)*Ident_set[mod1(cx,2)];
        OU_RU=n_double_set[mod1(cx+1,2)]-(1/2)*N_occu_set[mod1(cx+1,2)]+(1/4)*Ident_set[mod1(cx+1,2)];
        OU_LU=n_double_set[mod1(cx,2)]-(1/2)*N_occu_set[mod1(cx,2)]+(1/4)*Ident_set[mod1(cx,2)];
        Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
        Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
        Id_LU=unitary(space(OU_LU,1),space(OU_LU,1));
        @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_LU[-2,-5]*Id_RU[-3,-6];
        @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_LU[-2,-5]*OU_RU[-3,-6];
        @tensor hh_LU[:]:=Id_LD[-1,-4]*OU_LU[-2,-5]*Id_RU[-3,-6];
        hh_U=(hh_LD+hh_RU+hh_LU)*U_coe;

        ###############################
        hh=permute(hh_tx+hh_ty+hh_t2+hh_U,(1,2,3,),(4,5,6,));#hh_tx+hh_ty+hh_t2+hh_U
        eu,ev=eigh(hh);
        gate=ev*exp(-dt*eu)*ev';
        gate_set[cx]=gate;
    end
    return gate_set
end



function H_RU_LD_RD(parameters, space_type,Lx)

    if space_type==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
    elseif space_type==GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
    elseif space_type==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_Z2();
    end
    
    t1=parameters["t1"];
    t2=parameters["t2"];
    ϕ=parameters["ϕ"];
    U=parameters["U"];

    tx_coe_set=[exp(im*ϕ),exp(im*ϕ)]*t1;
    # ty_coe_set=[-1,1]*t1;
    # t2_coe_set=[-1,1]*t2;
    ty_coe_set=[1,-1]*t1;
    t2_coe_set=[1,-1]*t2;
    U_coe=U/3;

    H_set=Matrix{TensorMap}(undef,2,1);
    for cx=1:2;
        ####################
        # O1=Cdag_set[mod1(cx,Lx)];
        # O2=C_set[mod1(cx+1,Lx)];
        O1=Cdag_set[mod1(cx,2)];
        O2=C_set[mod1(cx+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*tx_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh_tx[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        ######################
        # O1=Cdag_set[mod1(cx+1,Lx)];
        # O2=C_set[mod1(cx+1,Lx)];
        O1=Cdag_set[mod1(cx+1,2)];
        O2=C_set[mod1(cx+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*ty_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        # Id=unitary(space(Cdag_set[mod1(cx,Lx)],2),space(Cdag_set[mod1(cx,Lx)],2));
        Id=unitary(space(Cdag_set[mod1(cx,2)],2),space(Cdag_set[mod1(cx,2)],2));
        @tensor hh_ty[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        #####################
        # O1=Cdag_set[mod1(cx,Lx)];
        # O2=C_set[mod1(cx+1,Lx)];
        O1=Cdag_set[mod1(cx,2)];
        O2=C_set[mod1(cx+1,2)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=-op;#!!!!!!! somehow this minus sign is required
        op=op*t2_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh[:]:=hh[-1,-3,-4,-6]*Id[-2,-5];
        sgate=swap_gate(hh,2,3);
        @tensor hh_t2[:]:=sgate[-2,-3,1,2]*hh[-1,1,2,-4,3,4]*sgate'[3,4,-5,-6];
        #################
        # OU_LD=n_double_set[mod1(cx,Lx)]-(1/2)*N_occu_set[mod1(cx,Lx)]+(1/4)*Ident_set[mod1(cx,Lx)];
        # OU_RU=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        # OU_RD=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        OU_LD=n_double_set[mod1(cx,2)]-(1/2)*N_occu_set[mod1(cx,2)]+(1/4)*Ident_set[mod1(cx,2)];
        OU_RU=n_double_set[mod1(cx+1,2)]-(1/2)*N_occu_set[mod1(cx+1,2)]+(1/4)*Ident_set[mod1(cx+1,2)];
        OU_RD=n_double_set[mod1(cx+1,2)]-(1/2)*N_occu_set[mod1(cx+1,2)]+(1/4)*Ident_set[mod1(cx+1,2)];
        Id_LD=unitary(space(OU_LD,1),space(OU_LD,1));
        Id_RU=unitary(space(OU_RU,1),space(OU_RU,1));
        Id_RD=unitary(space(OU_RD,1),space(OU_RD,1));
        @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
        @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
        @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
        hh_U=(hh_LD+hh_RU+hh_RD)*U_coe;
        
        #################
        hh=permute(hh_tx+hh_ty+hh_t2+hh_U,(1,2,3,),(4,5,6,));#hh_tx+hh_ty+hh_t2+hh_U
        H_set[cx]=hh;
    end
    return H_set
end


function check_convergence(lambdaset,lambdaset_old)
    Lx,Ly=size(lambdaset);
    err_set=Matrix{Float64}(undef,Lx,Ly);
    for ca=1:Lx
        for cb=1:Ly
            es1=convert(Array,lambdaset[ca,cb]);
            es2=convert(Array,lambdaset_old[ca,cb]);
            if size(es1)==size(es2)
                err_set[ca,cb]=norm(es1-es2);
            else
                err_set[ca,cb]=100;
            end
        end
    end
    return err_set
end
function itebd(parameters, Tset, lambdaxset,lambdayset,  tau,dt, Dmax)
    tol=dt*1e-3;#for determining convergence 
    println("tau, dt="*string([tau,dt]))
    # println("one step")
    # println(space(T_u))
    # println(space(T_d))
    Lx,Ly=size(Tset);
    gates_ru_ld_rd=gate_RU_LD_RD(parameters,dt, typeof(space(Tset[1,1],1)),Lx);
    gates_lu_ru_ld=gate_LU_RU_LD(parameters,dt, typeof(space(Tset[1,1],1)),Lx);

    lambdaxset_old=deepcopy(lambdaxset);
    lambdayset_old=deepcopy(lambdayset);
    for ct=1:Int(round(tau/abs(dt)))

        Tset, lambdaxset,lambdayset= triangle_update(ct,Tset, lambdaxset,lambdayset, gates_ru_ld_rd, gates_lu_ru_ld, Dmax);
        err_x=check_convergence(lambdaxset,lambdaxset_old);
        err_y=check_convergence(lambdayset,lambdayset_old);
        er=max(maximum(err_x),maximum(err_y));
        if mod(ct,20)==0
            println("iteration "*string(ct)*", convergence= "*string(er));flush(stdout)
        end
        if er<tol
            break;
        end
        lambdaxset_old=deepcopy(lambdaxset);
        lambdayset_old=deepcopy(lambdayset);
    end
    return Tset, lambdaxset,lambdayset
end

