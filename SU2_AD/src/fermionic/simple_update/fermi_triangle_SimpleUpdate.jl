using LinearAlgebra:diag,I,diagm 
###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################

function Rank(T::TensorMap)
    return length(domain(T))+length(codomain(T))
end

# function mypinv(T)
#     return pinv(T)
# end
function mypinv(T)
    epsilon0 = 1e-16
    epsilon=epsilon0*maximum(diag(convert(Array,T)))
    T_new=deepcopy(T);

    for (k,dst) in blocks(T_new)
        src = blocks(T_new)[k]
        @inbounds for i in 1:size(dst,1)
            dst[i,i] = dst[i,i]/(dst[i,i]^2+epsilon)
        end
    end
    return T_new
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









function evo_hopping_diagonala(O1,O2,A_RU0,A_LD0,A_RD0,hopping_coe,dt)

    function update_hopping_diagonala(O1,O2,A_RU,A_LD,A_RD)
        if Rank(O1)==3
            @tensor A_LD[:]:= A_LD[-1,-2,-3,-4,1]*O1[-6,-5,1]
            @tensor A_RU[:]:= A_RU[-1,-2,-3,-4,1]*O2[-6,-5,1]
            O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));

            gate=@ignore_derivatives parity_gate(A_LD,1); 
            @tensor A_LD[:]:=A_LD[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=@ignore_derivatives parity_gate(A_LD,2); 
            @tensor A_LD[:]:=A_LD[-1,1,-3,-4,-5,-6]*gate[-2,1];
            gate=@ignore_derivatives parity_gate(A_LD,4); 
            @tensor A_LD[:]:=A_LD[-1,-2,-3,1,-5,-6]*gate[-4,1];

            gate=@ignore_derivatives parity_gate(A_RD,2); 
            @tensor A_RD[:]:=A_RD[-1,1,-3,-4,-5]*gate[-2,1];
            gate=@ignore_derivatives parity_gate(A_RD,3); 
            @tensor A_RD[:]:=A_RD[-1,-2,1,-4,-5]*gate[-3,1];
            gate=@ignore_derivatives parity_gate(A_RD,5); 
            @tensor A_RD[:]:=A_RD[-1,-2,-3,-4,1]*gate[-5,1];

            gate=@ignore_derivatives parity_gate(A_RU,3); 
            @tensor A_RU[:]:=A_RU[-1,-2,1,-4,-5,-6]*gate[-3,1];
            gate=@ignore_derivatives parity_gate(A_RU,5); 
            @tensor A_RU[:]:=A_RU[-1,-2,-3,-4,1,-6]*gate[-5,1];


            U1=@ignore_derivatives unitary(fuse(space(A_LD,3)⊗space(A_LD,6)), space(A_LD,3)⊗space(A_LD,6)); 
            U2=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
            @tensor A_LD[:]:=A_LD[-1,-2,1,-4,-5,2]*U1[-3,1,2];
            @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U2[-2,1,2];
            @tensor A_RD[:]:=A_RD[1,-2,-3,3,-5]*O_string[4,2]*U1'[1,2,-1]*U2'[3,4,-4];

        elseif Rank(O1)==2
            @tensor A_LD[:]:= A_LD[-1,-2,-3,-4,1]*O1[-5,1]
            @tensor A_RU[:]:= A_RU[-1,-2,-3,-4,1]*O2[-5,1]
            A_RD=A_RD;
        end
        return A_LD,A_RU,A_RD
    end

    

    function move_RU(T,Um)
        T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
        T=permute_neighbour_ind(T,3,4,5);#L,U,R,  d,D,
        if Um==nothing
            U,S,V=tsvd(T,(1,2,3,),(4,5,));
            RU_res=U;#(L_ru,U_ru,R_ru, virtual_ru)
            RU_keep=S*V; #(virtual_ru, d_ru,D_ru)
            return RU_res, RU_keep,U
        else
            RU_res=Um;
            RU_keep=Um'*permute(T,(1,2,3,),(4,5,)); 
            return RU_res, RU_keep,nothing
        end
    end
    function move_LD(T,Um)
        T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
        T=permute_neighbour_ind(T,4,5,5);#L,U,d,D,R,
        T=permute_neighbour_ind(T,3,4,5);#L,U,D,  d,R,
        if Um==nothing
            U,S,V=tsvd(T,(1,2,3,),(4,5,));
            LD_res=U;#(L_ld,U_ld,D_ld, virtual_ld)
            LD_keep=S*V;#(virtual_ld, d_ld,R_ld)
            return LD_res,LD_keep,U
        else
            LD_res=Um;
            LD_keep=Um'*permute(T,(1,2,3,),(4,5,)); 
            return LD_res,LD_keep,nothing
        end
    end

    function update_RD(T_RU,Ua,T_LD,Ub, T_RD)#most expensive part
        RU_res, RU_keep,Ua=move_RU(T_RU,Ua);
        LD_res,LD_keep,Ub=move_LD(T_LD,Ub);

        T_RD=permute(T_RD,(1,4,5,3,2,));#L_rd,U_rd,d_rd,R_rd,D_rd,
        @tensor T_LD_RD[:]:=LD_keep[-1,-2,1]*T_RD[1,-3,-4,-5,-6];#(virtual_ld, d_ld,     U_rd,d_rd,R_rd,D_rd,)
        T_LD_RD=permute_neighbour_ind(T_LD_RD,3,4,6);#(virtual_ld, d_ld,   d_rd,U_rd,R_rd,D_rd,)
        T_LD_RD=permute_neighbour_ind(T_LD_RD,4,5,6);#(virtual_ld, d_ld,   d_rd,R_rd,U_rd,D_rd,)
        T_LD_RD=permute_neighbour_ind(T_LD_RD,5,6,6);#(virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd,)

        RU_keep=permute_neighbour_ind(RU_keep,2,3,3);#(virtual_ru, D_ru, d_ru)
        RU_keep=permute_neighbour_ind(RU_keep,1,2,3);#(D_ru, virtual_ru,  d_ru)
        gate=parity_gate(RU_keep,1);
        @tensor RU_keep[:]:=RU_keep[1,-2,-3]*gate[-1,1];
        Unita=unitary(fuse(space(T_LD_RD,1)*space(T_LD_RD,2)*space(T_LD_RD,3)), space(T_LD_RD,1)*space(T_LD_RD,2)*space(T_LD_RD,3));
        ## @tensor T_RU_LD_RD[:]:=T_LD_RD[-1,-2,-3,-4,-5,1]*RU_keep[1,-6,-7];# (virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd,)    (D_ru, virtual_ru,  d_ru)
        @tensor T_RU_LD_RD[:]:=Unita[-1,1,2,3]*T_LD_RD[1,2,3,-4,-5,4]*RU_keep[4,-6,-7];# (virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd,)    (D_ru, virtual_ru,  d_ru)

        return RU_res,LD_res,T_RU_LD_RD,Ua,Ub,Unita
    end

    function back_RD(T_RU_LD_RD,Unita)
        global D_max
        #######################
        # U1,S1,V1=tsvd(permute(T_RU_LD_RD,(1,2,3,4,5,),(6,7,)); trunc=truncdim(D_max));#(virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd_new),  (D_ru_new, virtual_ru,  d_ru) 
        U1,S1,V1=tsvd(permute(T_RU_LD_RD,(1,2,3,),(4,5,)); trunc=truncdim(D_max));#(virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd_new),  (D_ru_new, virtual_ru,  d_ru) 
        S1=S1/norm(S1);
        @tensor U1[:]:=U1[1,-4,-5,-6]*Unita'[-1,-2,-3,1];
        U1=permute(U1,(1,2,3,4,5,),(6,));

        #######################
        RU_keep=permute(V1,(1,2,3,));#(D_ru_new, virtual_ru,  d_ru)
        gate=parity_gate(RU_keep,1);
        @tensor RU_keep[:]:=RU_keep[1,-2,-3]*gate[-1,1];
        RU_keep=permute_neighbour_ind(RU_keep,1,2,3);#(virtual_ru, D_ru_new, d_ru)
        RU_keep=permute_neighbour_ind(RU_keep,2,3,3);#(virtual_ru, d_ru, D_ru_new)

        T_LD_RD=permute(U1*S1,(1,2,3,4,5,6,));#(virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd_new) 
        T_LD_RD=permute_neighbour_ind(T_LD_RD,5,6,6);#(virtual_ld, d_ld,   d_rd,R_rd,U_rd_new,D_rd,) 
        T_LD_RD=permute_neighbour_ind(T_LD_RD,4,5,6);#(virtual_ld, d_ld,   d_rd,U_rd_new,R_rd,D_rd,) 
        T_LD_RD=permute_neighbour_ind(T_LD_RD,3,4,6);#(virtual_ld, d_ld,   U_rd_new,d_rd,R_rd,D_rd,) 
        ###################
        U2,S2,V2=tsvd(permute(T_LD_RD,(1,2,),(3,4,5,6,)); trunc=truncdim(D_max));
        S2=S2/norm(S2);

        ###################
        LD_keep=permute(U2,(1,2,3,));#(virtual_ld, d_ld, R_ld)
        T_RD=permute(V2,(1,2,3,4,5,));#(L_rd, U_rd,d_rd,R_rd,D_rd,) 
        T_RD=permute(T_RD,(1,5,4,2,3,));

        @tensor T_RD[:]:=T_RD[-1,-2,-3,1,-5]*mypinv(S1)[1,-4];
        lambda_RU_RD=S1;
        lambda_LD_RD=S2;
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

    
    A_LD,A_RU,A_RD=update_hopping_diagonala(O1[1],O2[1],A_RU0,A_LD0,A_RD0);
    RU_res,LD_res,A_RU_LD_RD,Ua,Ub,Unita=update_RD(A_RU,nothing,A_LD,nothing, A_RD);

    A_LD_tem,A_RU_tem,A_RD_tem=update_hopping_diagonala(dt*hopping_coe*O1[2],O2[2],A_RU0,A_LD0,A_RD0);
    RU_res_tem,LD_res_tem,A_RU_LD_RD_tem,_=update_RD(A_RU_tem,Ua,A_LD_tem,Ub, A_RD_tem);
    A_RU_LD_RD=A_RU_LD_RD+A_RU_LD_RD_tem;

    A_LD_tem,A_RU_tem,A_RD_tem=update_hopping_diagonala(dt*(-hopping_coe')*O1[3],O2[3],A_RU0,A_LD0,A_RD0);
    RU_res_tem,LD_res_tem,A_RU_LD_RD_tem,_=update_RD(A_RU_tem,Ua,A_LD_tem,Ub, A_RD_tem);
    A_RU_LD_RD=A_RU_LD_RD+A_RU_LD_RD_tem;

    RU_keep,LD_keep, A_RD, lambda_RU_RD, lambda_LD_RD=back_RD(A_RU_LD_RD,Unita);
    A_LD=back_LD(LD_res,LD_keep);
    A_RU=back_RU(RU_res,RU_keep);

    return A_RU, A_LD, A_RD, lambda_RU_RD, lambda_LD_RD
end



function evo_hopping_RU_LD_RD(op_LD_RD_RU, A_RU0, A_LD0, A_RD0)

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
        global D_max
        #######################
        U1,S1,V1=tsvd(permute(T_RU_LD_RD,(1,2,3,4,),(5,6,)); trunc=truncdim(D_max));#(virtual_ld,  R_D_rd, d_ld, d_rd, U_rd_new),  (D_ru_new, d_ru, virtual_ru) 
        S1=S1/norm(S1);
        #######################
        RU_keep=permute(V1,(1,2,3,));#(D_ru_new, d_ru, virtual_ru)
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
        U2,S2,V2=tsvd(permute(T_LD_RD,(1,2,),(3,4,5,6,)); trunc=truncdim(D_max));
        S2=S2/norm(S2);
        ###################
        LD_keep=permute(U2,(1,2,3,));#(virtual_ld, d_ld, R_ld)
        T_RD=permute(V2,(1,2,3,4,5,));#(L_rd, U_rd,d_rd,R_rd,D_rd,) 
        T_RD=permute(T_RD,(1,5,4,2,3,));

        @tensor T_RD[:]:=T_RD[-1,-2,-3,1,-5]*mypinv(S1)[1,-4];
        lambda_RU_RD=S1;
        lambda_LD_RD=S2;
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





function evo_hopping_LU_RU_LD(op_LD_LU_RU, A_RU0, A_LD0, A_LU0)

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
        global D_max
        #######################
        U1,S1,V1=tsvd(permute(T_LU_RU_LD,(1,2,),(3,4,5,6,)); trunc=truncdim(D_max));#(virtual_ld, d_ld,U_ld_new)  (D_lu_new, d_lu, d_ru, L_U_lu, virtual_ru)
        S1=S1/norm(S1);
        #######################
        LD_keep=permute(U1,(1,2,3,));#(virtual_ld, d_ld,U_ld_new)
        gate=parity_gate(LD_keep,3);
        @tensor LD_keep[:]:=LD_keep[-1,-2,1]*gate[-3,1];

        
        T_LU_RU=permute(S1*V1,(1,2,3,4,5,));#(D_lu_new, d_lu, d_ru, L_U_lu, virtual_ru) 
        T_LU_RU=permute_neighbour_ind(T_LU_RU,3,4,5);#(D_lu_new, d_lu, L_U_lu, d_ru, virtual_ru) 
        @tensor T_LU_RU[:]:=T_LU_RU[-1,-2,1,-5,-6]*U_LU'[-3,-4,1];#(D_lu_new, d_lu, L_lu, U_lu, d_ru, virtual_ru)
        ###################
        U2,S2,V2=tsvd(permute(T_LU_RU,(1,2,3,4,),(5,6,)); trunc=truncdim(D_max));#(D_lu_new, d_lu, L_lu, U_lu, R_lu_new) (L_ru_new, d_ru, virtual_ru)
        S2=S2/norm(S2);
        ###################
        RU_keep=permute(V2,(1,2,3,));#(L_ru_new, d_ru, virtual_ru)
        T_LU=permute(U2,(1,2,3,4,5,));#(D_lu_new, d_lu, L_lu, U_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,2,3,5); #(D_lu_new, L_lu, d_lu, U_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,1,2,5); #(L_lu, D_lu_new, d_lu, U_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,3,4,5); #(L_lu, D_lu_new, U_lu, d_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,2,3,5); #(L_lu, U_lu, D_lu_new, d_lu, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,3,4,5); #(L_lu, U_lu, d_lu, D_lu_new, R_lu_new) 
        T_LU=permute_neighbour_ind(T_LU,4,5,5); #(L_lu, U_lu, d_lu,, R_lu_new D_lu_new) 
        T_LU=permute(T_LU,(1,5,4,2,3,));

        @tensor T_LU[:]:=T_LU[-1,1,-3,-4,-5]*mypinv(S1)[-2,1];
        lambda_LD_LU=S1;
        lambda_LU_RU=S2;
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







function triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gates_ru_ld_rd, gates_lu_ru_ld)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    global update_triangle1, update_triangle2
    if update_triangle1
        # B
        #CD
        @tensor A_RU[:]:=T_B[1,-2,3,4,-5]*lambda_A_R[-1,1]*lambda_A_L[-3,3]*lambda_D_D[-4,4];
        @tensor A_LD[:]:=T_C[1,2,-3,4,-5]*lambda_D_R[-1,1]*lambda_A_U[-2,2]*lambda_A_D[-4,4];
        @tensor A_RD[:]:=T_D[1,2,3,4,-5]*lambda_D_L[1,-1]*lambda_D_D[2,-2]*lambda_D_R[3,-3]*lambda_D_U[4,-4];
        A_RU,A_LD,A_RD,lambda_RU_RD, lambda_LD_RD=evo_hopping_RU_LD_RD(gates_ru_ld_rd[1], A_RU,A_LD,A_RD);
        @tensor T_B[:]:=A_RU[1,-2,3,4,-5]*mypinv(lambda_A_R)[-1,1]*mypinv(lambda_A_L)[-3,3]*mypinv(lambda_D_D)[-4,4];
        @tensor T_C[:]:=A_LD[1,2,-3,4,-5]*mypinv(lambda_D_R)[-1,1]*mypinv(lambda_A_U)[-2,2]*mypinv(lambda_A_D)[-4,4];
        @tensor T_D[:]:=A_RD[-1,2,3,-4,-5]*mypinv(lambda_D_D)[2,-2]*mypinv(lambda_D_R)[3,-3];
        lambda_D_U=lambda_RU_RD; 
        lambda_D_L=permute(lambda_LD_RD,(2,),(1,));
        println(space(lambda_D_U))



        # A
        #DC
        @tensor A_RU[:]:=T_A[1,-2,3,4,-5]*lambda_A_L[1,-1]*lambda_A_R[3,-3]*lambda_A_U[4,-4];
        @tensor A_LD[:]:=T_D[1,2,-3,4,-5]*lambda_D_L[1,-1]*lambda_D_D[2,-2]*lambda_D_U[4,-4];
        @tensor A_RD[:]:=T_C[1,2,3,4,-5]*lambda_D_R[-1,1]*lambda_A_U[-2,2]*lambda_D_L[-3,3]*lambda_A_D[-4,4];
        A_RU,A_LD,A_RD,lambda_RU_RD, lambda_LD_RD=evo_hopping_RU_LD_RD(gates_ru_ld_rd[2], A_RU,A_LD,A_RD);
        @tensor T_A[:]:=A_RU[1,-2,3,4,-5]*mypinv(lambda_A_L)[1,-1]*mypinv(lambda_A_R)[3,-3]*mypinv(lambda_A_U)[4,-4];
        @tensor T_D[:]:=A_LD[1,2,-3,4,-5]*mypinv(lambda_D_L)[1,-1]*mypinv(lambda_D_D)[2,-2]*mypinv(lambda_D_U)[4,-4];
        @tensor T_C[:]:=A_RD[-1,2,3,-4,-5]*mypinv(lambda_A_U)[-2,2]*mypinv(lambda_D_L)[-3,3];
        lambda_A_D=permute(lambda_RU_RD,(2,),(1,)); 
        lambda_D_R=lambda_LD_RD;

        
        

        # D
        #AB
        @tensor A_RU[:]:=T_D[1,-2,3,4,-5]*lambda_D_L[1,-1]*lambda_D_R[3,-3]*lambda_D_U[4,-4];
        @tensor A_LD[:]:=T_A[1,2,-3,4,-5]*lambda_A_L[1,-1]*lambda_A_D[2,-2]*lambda_A_U[4,-4];
        @tensor A_RD[:]:=T_B[1,2,3,4,-5]*lambda_A_R[-1,1]*lambda_D_U[-2,2]*lambda_A_L[-3,3]*lambda_D_D[-4,4];
        A_RU,A_LD,A_RD,lambda_RU_RD, lambda_LD_RD=evo_hopping_RU_LD_RD(gates_ru_ld_rd[1], A_RU,A_LD,A_RD);
        @tensor T_D[:]:=A_RU[1,-2,3,4,-5]*mypinv(lambda_D_L)[1,-1]*mypinv(lambda_D_R)[3,-3]*mypinv(lambda_D_U)[4,-4];
        @tensor T_A[:]:=A_LD[1,2,-3,4,-5]*mypinv(lambda_A_L)[1,-1]*mypinv(lambda_A_D)[2,-2]*mypinv(lambda_A_U)[4,-4];
        @tensor T_B[:]:=A_RD[-1,2,3,-4,-5]*mypinv(lambda_D_U)[-2,2]*mypinv(lambda_A_L)[-3,3];
        lambda_D_D=permute(lambda_RU_RD,(2,),(1,)); 
        lambda_A_R=lambda_LD_RD;

        

        # C
        #BA
        @tensor A_RU[:]:=T_C[1,-2,3,4,-5]*lambda_D_R[-1,1]*lambda_D_L[-3,3]*lambda_A_D[-4,4];
        @tensor A_LD[:]:=T_B[1,2,-3,4,-5]*lambda_A_R[-1,1]*lambda_D_U[-2,2]*lambda_D_D[-4,4];
        @tensor A_RD[:]:=T_A[1,2,3,4,-5]*lambda_A_L[1,-1]*lambda_A_D[2,-2]*lambda_A_R[3,-3]*lambda_A_U[4,-4];
        A_RU,A_LD,A_RD,lambda_RU_RD, lambda_LD_RD=evo_hopping_RU_LD_RD(gates_ru_ld_rd[2], A_RU,A_LD,A_RD);
        @tensor T_C[:]:=A_RU[1,-2,3,4,-5]*mypinv(lambda_D_R)[-1,1]*mypinv(lambda_D_L)[-3,3]*mypinv(lambda_A_D)[-4,4];
        @tensor T_B[:]:=A_LD[1,2,-3,4,-5]*mypinv(lambda_A_R)[-1,1]*mypinv(lambda_D_U)[-2,2]*mypinv(lambda_D_D)[-4,4];
        @tensor T_A[:]:=A_RD[-1,2,3,-4,-5]*mypinv(lambda_A_D)[2,-2]*mypinv(lambda_A_R)[3,-3];
        lambda_A_U=lambda_RU_RD; 
        lambda_A_L=permute(lambda_LD_RD,(2,),(1,));
    end
    


    #############################
    if update_triangle2
        #AB
        #C
        @tensor A_RU[:]:=T_B[-1,2,3,4,-5]*lambda_D_U[-2,2]*lambda_A_L[-3,3]*lambda_D_D[-4,4];
        @tensor A_LD[:]:=T_C[1,2,3,-4,-5]*lambda_D_R[-1,1]*lambda_A_U[-2,2]*lambda_D_L[-3,3];
        @tensor A_LU[:]:=T_A[1,2,3,4,-5]*lambda_A_L[1,-1]*lambda_A_D[2,-2]*lambda_A_R[3,-3]*lambda_A_U[4,-4];
        A_RU, A_LD, A_LU, lambda_LD_LU, lambda_LU_RU=evo_hopping_LU_RU_LD(gates_lu_ru_ld[1], A_RU, A_LD, A_LU);
        @tensor T_B[:]:=A_RU[-1,2,3,4,-5]*mypinv(lambda_D_U)[-2,2]*mypinv(lambda_A_L)[-3,3]*mypinv(lambda_D_D)[-4,4];
        @tensor T_C[:]:=A_LD[1,2,3,-4,-5]*mypinv(lambda_D_R)[-1,1]*mypinv(lambda_A_U)[-2,2]*mypinv(lambda_D_L)[-3,3];
        @tensor T_A[:]:=A_LU[1,-2,-3,4,-5]*mypinv(lambda_A_L)[1,-1]*mypinv(lambda_A_U)[4,-4];
        lambda_A_D=permute(lambda_LD_LU,(2,),(1,)); 
        lambda_A_R=lambda_LU_RU;

        #BA
        #D
        @tensor A_RU[:]:=T_A[-1,2,3,4,-5]*lambda_A_D[2,-2]*lambda_A_R[3,-3]*lambda_A_U[4,-4];
        @tensor A_LD[:]:=T_D[1,2,3,-4,-5]*lambda_D_L[1,-1]*lambda_D_D[2,-2]*lambda_D_R[3,-3];
        @tensor A_LU[:]:=T_B[1,2,3,4,-5]*lambda_A_R[-1,1]*lambda_D_U[-2,2]*lambda_A_L[-3,3]*lambda_D_D[-4,4];
        A_RU, A_LD, A_LU, lambda_LD_LU, lambda_LU_RU=evo_hopping_LU_RU_LD(gates_lu_ru_ld[2], A_RU, A_LD, A_LU);
        @tensor T_A[:]:=A_RU[-1,2,3,4,-5]*mypinv(lambda_A_D)[2,-2]*mypinv(lambda_A_R)[3,-3]*mypinv(lambda_A_U)[4,-4];
        @tensor T_D[:]:=A_LD[1,2,3,-4,-5]*mypinv(lambda_D_L)[1,-1]*mypinv(lambda_D_D)[2,-2]*mypinv(lambda_D_R)[3,-3];
        @tensor T_B[:]:=A_LU[1,-2,-3,4,-5]*mypinv(lambda_A_R)[-1,1]*mypinv(lambda_D_D)[-4,4];
        lambda_D_U=lambda_LD_LU; 
        lambda_A_L=permute(lambda_LU_RU,(2,),(1,));

        #CD
        #A
        @tensor A_RU[:]:=T_D[-1,2,3,4,-5]*lambda_D_D[2,-2]*lambda_D_R[3,-3]*lambda_D_U[4,-4];
        @tensor A_LD[:]:=T_A[1,2,3,-4,-5]*lambda_A_L[1,-1]*lambda_A_D[2,-2]*lambda_A_R[3,-3];
        @tensor A_LU[:]:=T_C[1,2,3,4,-5]*lambda_D_R[-1,1]*lambda_A_U[-2,2]*lambda_D_L[-3,3]*lambda_A_D[-4,4];
        A_RU, A_LD, A_LU, lambda_LD_LU, lambda_LU_RU=evo_hopping_LU_RU_LD(gates_lu_ru_ld[1], A_RU, A_LD, A_LU);
        @tensor T_D[:]:=A_RU[-1,2,3,4,-5]*mypinv(lambda_D_D)[2,-2]*mypinv(lambda_D_R)[3,-3]*mypinv(lambda_D_U)[4,-4];
        @tensor T_A[:]:=A_LD[1,2,3,-4,-5]*mypinv(lambda_A_L)[1,-1]*mypinv(lambda_A_D)[2,-2]*mypinv(lambda_A_R)[3,-3];
        @tensor T_C[:]:=A_LU[1,-2,-3,4,-5]*mypinv(lambda_D_R)[-1,1]*mypinv(lambda_A_D)[-4,4];
        lambda_A_U=lambda_LD_LU; 
        lambda_D_L=permute(lambda_LU_RU,(2,),(1,));

        #DC
        #B
        @tensor A_RU[:]:=T_C[-1,2,3,4,-5]*lambda_A_U[-2,2]*lambda_D_L[-3,3]*lambda_A_D[-4,4];
        @tensor A_LD[:]:=T_B[1,2,3,-4,-5]*lambda_A_R[-1,1]*lambda_D_U[-2,2]*lambda_A_L[-3,3];
        @tensor A_LU[:]:=T_D[1,2,3,4,-5]*lambda_D_L[1,-1]*lambda_D_D[2,-2]*lambda_D_R[3,-3]*lambda_D_U[4,-4];
        A_RU, A_LD, A_LU, lambda_LD_LU, lambda_LU_RU=evo_hopping_LU_RU_LD(gates_lu_ru_ld[2], A_RU, A_LD, A_LU);
        @tensor T_C[:]:=A_RU[-1,2,3,4,-5]*mypinv(lambda_A_U)[-2,2]*mypinv(lambda_D_L)[-3,3]*mypinv(lambda_A_D)[-4,4];
        @tensor T_B[:]:=A_LD[1,2,3,-4,-5]*mypinv(lambda_A_R)[-1,1]*mypinv(lambda_D_U)[-2,2]*mypinv(lambda_A_L)[-3,3];
        @tensor T_D[:]:=A_LU[1,-2,-3,4,-5]*mypinv(lambda_D_L)[1,-1]*mypinv(lambda_D_U)[4,-4];
        lambda_D_D=permute(lambda_LD_LU,(2,),(1,)); 
        lambda_D_R=lambda_LU_RU;
    end

    T_A=T_A/norm(T_A);
    T_B=T_B/norm(T_B);
    T_C=T_C/norm(T_C);
    T_D=T_D/norm(T_D);
    lambda_A_L=lambda_A_L/norm(lambda_A_L);
    lambda_A_D=lambda_A_D/norm(lambda_A_D);
    lambda_A_R=lambda_A_R/norm(lambda_A_R);
    lambda_A_U=lambda_A_U/norm(lambda_A_U);
    lambda_D_L=lambda_D_L/norm(lambda_D_L);
    lambda_D_D=lambda_D_D/norm(lambda_D_D);
    lambda_D_R=lambda_D_R/norm(lambda_D_R);
    lambda_D_U=lambda_D_U/norm(lambda_D_U);

    return T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U
end


function gate_RU_LD_RD(parameters,dt, space_type)

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
        O1=Cdag_set[mod1(cx,Lx)];
        O2=C_set[mod1(cx+1,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*tx_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh_tx[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        ######################
        O1=Cdag_set[mod1(cx+1,Lx)];
        O2=C_set[mod1(cx+1,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*ty_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(Cdag_set[mod1(cx,Lx)],2),space(Cdag_set[mod1(cx,Lx)],2));
        @tensor hh_ty[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        #####################
        O1=Cdag_set[mod1(cx,Lx)];
        O2=C_set[mod1(cx+1,Lx)];
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
        OU_LD=n_double_set[mod1(cx,Lx)]-(1/2)*N_occu_set[mod1(cx,Lx)]+(1/4)*Ident_set[mod1(cx,Lx)];
        OU_RU=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        OU_RD=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
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


function gate_LU_RU_LD(parameters,dt, space_type)

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
        O1=Cdag_set[mod1(cx,Lx)];
        O2=C_set[mod1(cx+1,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*tx_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,1),space(hh,1));
        @tensor hh_tx[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        ######################
        O1=Cdag_set[mod1(cx,Lx)];
        O2=C_set[mod1(cx,Lx)];
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*ty_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(Cdag_set[mod1(cx+1,Lx)],2),space(Cdag_set[mod1(cx+1,Lx)],2));
        @tensor hh_ty[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        #####################
        O1=Cdag_set[mod1(cx,Lx)];
        O2=C_set[mod1(cx+1,Lx)];
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
        OU_LD=n_double_set[mod1(cx,Lx)]-(1/2)*N_occu_set[mod1(cx,Lx)]+(1/4)*Ident_set[mod1(cx,Lx)];
        OU_RU=n_double_set[mod1(cx+1,Lx)]-(1/2)*N_occu_set[mod1(cx+1,Lx)]+(1/4)*Ident_set[mod1(cx+1,Lx)];
        OU_LU=n_double_set[mod1(cx,Lx)]-(1/2)*N_occu_set[mod1(cx,Lx)]+(1/4)*Ident_set[mod1(cx,Lx)];
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

function itebd(parameters, T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U,  tau,dt)
    # println("one step")
    # println(space(T_u))
    # println(space(T_d))

    gates_ru_ld_rd=gate_RU_LD_RD(parameters,dt, typeof(space(T_A,1)));
    gates_lu_ru_ld=gate_LU_RU_LD(parameters,dt, typeof(space(T_A,1)));


    for ct=1:Int(round(tau/abs(dt)))
        println("iteration "*string(ct));flush(stdout)
        T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U= triangle_update(T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U, gates_ru_ld_rd, gates_lu_ru_ld);
    end
    return T_A, T_B, T_C, T_D, lambda_A_L, lambda_A_D, lambda_A_R, lambda_A_U, lambda_D_L, lambda_D_D, lambda_D_R, lambda_D_U
end

