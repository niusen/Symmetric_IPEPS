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






function trotter_gate(H,dt)
        eu,ev=eigh(H);
        @assert norm(ev*eu*ev'-H)/norm(H)<1e-14 
        gate=ev*exp(-dt*eu)*ev';
        gate_half=ev*exp(-dt*eu/2)*ev';
    return gate, gate_half
end


function verify_evo_hopping_diagonala(CTM,O1,O2,A_cell,AA_cell,cx,cy,hopping_coe,dt)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    function update_hopping_diagonala(O1,O2,A_cell,cx,cy)
        if Rank(O1)==3
            @tensor A_LD[:]:= A_cell[pos_LD[1]][pos_LD[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
            @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
            O_string=@ignore_derivatives unitary(space(O1,1),space(O1,1));

            gate=@ignore_derivatives parity_gate(A_LD,1); 
            @tensor A_LD[:]:=A_LD[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=@ignore_derivatives parity_gate(A_LD,2); 
            @tensor A_LD[:]:=A_LD[-1,1,-3,-4,-5,-6]*gate[-2,1];
            gate=@ignore_derivatives parity_gate(A_LD,4); 
            @tensor A_LD[:]:=A_LD[-1,-2,-3,1,-5,-6]*gate[-4,1];

            gate=@ignore_derivatives parity_gate(A_cell[pos_RD[1]][pos_RD[2]],2); 
            @tensor A_RD[:]:=A_cell[pos_RD[1]][pos_RD[2]][-1,1,-3,-4,-5]*gate[-2,1];
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
            @tensor A_LD[:]:= A_cell[pos_LD[1]][pos_LD[2]][-1,-2,-3,-4,1]*O1[-5,1]
            @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-5,1]
            A_RD=A_cell[pos_RD[1]][pos_RD[2]];
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

    function update_RD(T_RU,Ua,T_LD,Ub, T_RD)
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
        @tensor T_RU_LD_RD[:]:=T_LD_RD[-1,-2,-3,-4,-5,1]*RU_keep[1,-6,-7];# (virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd,)    (D_ru, virtual_ru,  d_ru)

        return RU_res,LD_res,T_RU_LD_RD,Ua,Ub
    end

    function back_RD(T_RU_LD_RD)
        global D_max
        #######################
        U1,S1,V1=tsvd(permute(T_RU_LD_RD,(1,2,3,4,5,),(6,7,)); trunc=truncdim(D_max));#(virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd_new),  (D_ru_new, virtual_ru,  d_ru) 
        # println(S1)
        # U2,S2,V2=tsvd(permute(T_RU_LD_RD,(1,2,),(3,4,5,6,7,)); trunc=truncdim(D_max));#(virtual_ld, d_ld,   d_rd,R_rd,D_rd,U_rd_new),  (D_ru_new, virtual_ru,  d_ru) 
        # println(S2)
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
        ###################
        LD_keep=permute(U2*S2,(1,2,3,));#(virtual_ld, d_ld, R_ld)
        T_RD=permute(V2,(1,2,3,4,5,));#(L_rd, U_rd,d_rd,R_rd,D_rd,) 
        T_RD=permute(T_RD,(1,5,4,2,3,));
        return RU_keep,LD_keep, T_RD
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


    A_LD,A_RU,A_RD=update_hopping_diagonala(O1[1],O2[1],A_cell,cx,cy);
    RU_res,LD_res,A_RU_LD_RD,Ua,Ub=update_RD(A_RU,nothing,A_LD,nothing, A_RD);

    A_LD_tem,A_RU_tem,A_RD_tem=update_hopping_diagonala(dt*hopping_coe*O1[2],O2[2],A_cell,cx,cy);
    RU_res_tem,LD_res_tem,A_RU_LD_RD_tem,_=update_RD(A_RU_tem,Ua,A_LD_tem,Ub, A_RD_tem);
    A_RU_LD_RD=A_RU_LD_RD+A_RU_LD_RD_tem;



    A_LD_tem,A_RU_tem,A_RD_tem=update_hopping_diagonala(dt*(-hopping_coe')*O1[3],O2[3],A_cell,cx,cy);
    RU_res_tem,LD_res_tem,A_RU_LD_RD_tem,_=update_RD(A_RU_tem,Ua,A_LD_tem,Ub, A_RD_tem);
    A_RU_LD_RD=A_RU_LD_RD+A_RU_LD_RD_tem;



    RU_keep,LD_keep, A_RD=back_RD(A_RU_LD_RD);
    A_LD=back_LD(LD_res,LD_keep);
    A_RU=back_RU(RU_res,RU_keep);

    AA_LD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LD[1]][pos_LD[2]]',A_LD);
    AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);
    AA_RD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);


    ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_LD_double,AA_RD_double,cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob   
end



function verify_evo_hopping_x(CTM,O1,O2,A_cell,AA_cell,cx,cy,hopping_coe,dt)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    function update_hopping_x(O1,O2,A_cell,cx,cy)
        if Rank(O1)==3
            @tensor A_LU[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
            @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]
        
                
            gate=@ignore_derivatives parity_gate(A_LU,1); 
            @tensor A_LU[:]:=A_LU[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=@ignore_derivatives parity_gate(A_LU,2); 
            @tensor A_LU[:]:=A_LU[-1,1,-3,-4,-5,-6]*gate[-2,1];
            gate=@ignore_derivatives parity_gate(A_LU,4); 
            @tensor A_LU[:]:=A_LU[-1,-2,-3,1,-5,-6]*gate[-4,1];
        
            gate=@ignore_derivatives parity_gate(A_RU,1); 
            @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=@ignore_derivatives parity_gate(A_RU,4); 
            @tensor A_RU[:]:=A_RU[-1,-2,-3,1,-5,-6]*gate[-4,1];
        
        
            U=@ignore_derivatives unitary(fuse(space(A_LU,3)⊗space(A_LU,6)), space(A_LU,3)⊗space(A_LU,6)); 
            @tensor A_LU[:]:=A_LU[-1,-2,1,-4,-5,2]*U[-3,1,2];
            @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,2]*U'[1,2,-1];
        elseif Rank(O1)==2
            @tensor A_LU[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-5,1]
            @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O2[-5,1]
        end
        return A_LU,A_RU
    end

    function move_LU(T,Um)
        T=permute(T,(1,4,5,3,2,));#L,U,d,R,D,
        T=permute_neighbour_ind(T,4,5,5);#L,U,d,D,R,
        T=permute_neighbour_ind(T,3,4,5);#L,U,D,  d,R,
        if Um==nothing
            U,S,V=tsvd(T,(1,2,3,),(4,5,));
            LU_res=U;#(L_lu,U_lu,D_lu, virtual_lu)
            LU_keep=S*V;#(virtual_lu, d_lu,R_lu)
            return LU_res,LU_keep,U
        else
            LU_res=Um;
            LU_keep=Um'*permute(T,(1,2,3,),(4,5,)); 
            return LU_res,LU_keep,nothing
        end
    end

    
    function update_RU(T_LU,Um, T_RU)
        LU_res, LU_keep,Um=move_LU(T_LU,Um);
    
        T_RU=permute(T_RU,(1,4,5,3,2,));#L_ru,U_ru,d_ru,R_ru,D_ru,
        @tensor T_LU_RU[:]:=LU_keep[-1,-2,1]*T_RU[1,-3,-4,-5,-6];#(virtual_lu, d_lu,     U_ru,d_ru,R_ru,D_ru,)
        return LU_res,T_LU_RU,Um
    end

    function back_RU(T_LU_RU)
        global D_max
        ###################
        U,S,V=tsvd(permute(T_LU_RU,(1,2,),(3,4,5,6,)); trunc=truncdim(D_max));#(virtual_lu, d_lu,R_lu_new,    L_ru_new, U_ru,d_ru,R_ru,D_ru,)
        ###################
        LU_keep=permute(U*S,(1,2,3,));#(virtual_lu, d_lu,R_lu_new,)
        T_RU=permute(V,(1,2,3,4,5,));#(L_ru_new, U_ru,d_ru,R_ru,D_ru,) 
        T_RU=permute(T_RU,(1,5,4,2,3,));
        return LU_keep, T_RU
    end

    function back_LU(LU_res,LU_keep)
        #(L_lu,U_lu,D_lu, virtual_lu)
        #(virtual_lu, d_lu, R_lu)
        @tensor T_LU[:]:=LU_res[-1,-2,-3,1]*LU_keep[1,-4,-5];#(L_lu,U_lu,D_lu, d_lu, R_lu)
        T_LU=permute_neighbour_ind(T_LU,3,4,5);#(L,U, d,D, R)
        T_LU=permute_neighbour_ind(T_LU,4,5,5);#(L,U, d,R, D)
        T_LU=permute(T_LU,(1,5,4,2,3,));
        return T_LU
    end

    
    A_LU,A_RU=update_hopping_x(O1[1],O2[1],A_cell,cx,cy);
    LU_res,A_LU_RU,Um=update_RU(A_LU,nothing, A_RU);

    A_LU_tem,A_RU_tem=update_hopping_x(dt*hopping_coe*O1[2],O2[2],A_cell,cx,cy);
    LU_res_tem,A_LU_RU_tem,_=update_RU(A_LU_tem,Um, A_RU_tem);
    A_LU_RU=A_LU_RU+A_LU_RU_tem;

    A_LU_tem,A_RU_tem=update_hopping_x(dt*(-hopping_coe')*O1[3],O2[3],A_cell,cx,cy);
    LU_res_tem,A_LU_RU_tem,_=update_RU(A_LU_tem,Um, A_RU_tem);
    A_LU_RU=A_LU_RU+A_LU_RU_tem;

    LU_keep, A_RU=back_RU(A_LU_RU);
    A_LU=back_LU(LU_res,LU_keep);

    ###############################################
    AA_LU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LU[1]][pos_LU[2]]',A_LU);
    AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);

    ob=ob_2x2(CTM,AA_LU_double,AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
end


function verify_evo_hopping_y(CTM,O1,O2,A_cell,AA_cell,cx,cy,hopping_coe,dt)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    function update_hopping_y(O1,O2,A_cell,cx,cy)
        if Rank(O1)==3
            #the first index of O is dummy
            @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O1[-6,-5,1]
            @tensor A_RD[:]:= A_cell[pos_RD[1]][pos_RD[2]][-1,-2,-3,-4,1]*O2[-6,-5,1]

            gate=@ignore_derivatives parity_gate(A_RU,1); 
            @tensor A_RU[:]:=A_RU[1,-2,-3,-4,-5,-6]*gate[-1,1];
            gate=@ignore_derivatives parity_gate(A_RU,2); 
            @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,-6]*gate[-2,1];
            gate=@ignore_derivatives parity_gate(A_RU,4); 
            @tensor A_RU[:]:=A_RU[-1,-2,-3,1,-5,-6]*gate[-4,1];

            U1=@ignore_derivatives unitary(fuse(space(A_RU,2)⊗space(A_RU,6)), space(A_RU,2)⊗space(A_RU,6)); 
            @tensor A_RU[:]:=A_RU[-1,1,-3,-4,-5,2]*U1[-2,1,2];
            @tensor A_RD[:]:=A_RD[-1,-2,-3,1,-5,2]*U1'[1,2,-4];
        elseif Rank(O1)==2
            @tensor A_RU[:]:= A_cell[pos_RU[1]][pos_RU[2]][-1,-2,-3,-4,1]*O1[-5,1]
            @tensor A_RD[:]:= A_cell[pos_RD[1]][pos_RD[2]][-1,-2,-3,-4,1]*O2[-5,1]
        end

        return A_RU,A_RD
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

    function update_RD(T_RU,Um, T_RD)
        RU_res, RU_keep,Um=move_RU(T_RU,Um);
    
        T_RD=permute(T_RD,(1,4,5,3,2,));#L_rd,U_rd,d_rd,R_rd,D_rd,
        T_RD=permute_neighbour_ind(T_RD,1,2,5);#U_rd,L_rd,d_rd,R_rd,D_rd,
        @tensor T_RU_RD[:]:=RU_keep[-1,-2,1]*T_RD[1,-3,-4,-5,-6];#(virtual_ru, d_ru,D_ru),     (U_rd,L_rd,d_rd,R_rd,D_rd)
        return RU_res,T_RU_RD,Um
    end

    
    function back_RD(T_RU_RD)
        global D_max
        ###################
        U,S,V=tsvd(permute(T_RU_RD,(1,2,),(3,4,5,6,)); trunc=truncdim(D_max));#(virtual_ru, d_ru,D_ru_new),     (U_rd_new,L_rd,d_rd,R_rd,D_rd)
        ###################
        RU_keep=permute(U*S,(1,2,3,));#(virtual_ru, d_ru,D_ru_new)
        T_RD=permute(V,(1,2,3,4,5,));#(U_rd_new,L_rd,d_rd,R_rd,D_rd) 
        T_RD=permute_neighbour_ind(T_RD,1,2,5);#(L_rd,U_rd_new,d_rd,R_rd,D_rd) 
        T_RD=permute(T_RD,(1,5,4,2,3,));
        return RU_keep, T_RD
    end

    function back_RU(RU_res,RU_keep)
        #(L_ru,U_ru,R_ru, virtual_ru)
        #(virtual_ru, d_ru,D_ru_new)
        @tensor T_RU[:]:=RU_res[-1,-2,-3,1]*RU_keep[1,-4,-5];#(L_ru,U_ru,R_ru, virtual_ru),  (virtual_ru, d_ru,D_ru_new)
        T_RU=permute_neighbour_ind(T_RU,3,4,5);#(L,U,d,R,D)
        T_RU=permute(T_RU,(1,5,4,2,3,));
        return T_RU
    end


    A_RU,A_RD=update_hopping_y(O1[1],O2[1],A_cell,cx,cy);
    RU_res,A_RU_RD,Um=update_RD(A_RU,nothing, A_RD);

    A_RU_tem,A_RD_tem=update_hopping_y(dt*hopping_coe*O1[2],O2[2],A_cell,cx,cy);
    RU_res_tem,A_RU_RD_tem,_=update_RD(A_RU_tem,Um, A_RD_tem);
    A_RU_RD=A_RU_RD+A_RU_RD_tem;

    A_RU_tem,A_RD_tem=update_hopping_y(dt*(-hopping_coe')*O1[3],O2[3],A_cell,cx,cy);
    RU_res_tem,A_RU_RD_tem,_=update_RD(A_RU_tem,Um, A_RD_tem);
    A_RU_RD=A_RU_RD+A_RU_RD_tem;

    RU_keep, A_RD=back_RD(A_RU_RD);
    A_RU=back_RU(RU_res,RU_keep);
######################################################
    AA_RD_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RD[1]][pos_RD[2]]',A_RD);
    AA_RU_double,_,_,_,_=build_double_layer_swap(A_cell[pos_RU[1]][pos_RU[2]]',A_RU);

    ob=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_RU_double,AA_cell[pos_LD[1]][pos_LD[2]],AA_RD_double,cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
end



function verify_evo_onsite(CTM,O1,A_cell,AA_cell,cx,cy,h_coe,dt)
    global Lx,Ly
    pos_LU=[mod1(cx+1,Lx),mod1(cy+1,Ly)];
    pos_RU=[mod1(cx+2,Lx),mod1(cy+1,Ly)];
    pos_LD=[mod1(cx+1,Lx),mod1(cy+2,Ly)];
    pos_RD=[mod1(cx+2,Lx),mod1(cy+2,Ly)];

    function update_onsite(O1,A_cell,cx,cy)
        @tensor A1[:]:= A_cell[pos_LU[1]][pos_LU[2]][-1,-2,-3,-4,1]*O1[-5,1]
        return A1
    end


    A_tem1=update_onsite(O1[1],A_cell,cx,cy);

    A_tem2=update_onsite(dt*h_coe*O1[2],A_cell,cx,cy);
    A1=A_tem1+A_tem2;
    ###############################
    A1_double,_,_,_,_=build_double_layer_swap(A_cell[pos_LU[1]][pos_LU[2]]',A1);
        
    ob=ob_2x2(CTM,A1_double, AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    Norm=ob_2x2(CTM,AA_cell[pos_LU[1]][pos_LU[2]],AA_cell[pos_RU[1]][pos_RU[2]],AA_cell[pos_LD[1]][pos_LD[2]],AA_cell[pos_RD[1]][pos_RD[2]],cx,cy);
    ob=ob/Norm;
    return ob
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
    #U,S,V=tsvd(Tr_absorbed,(1,5,),(2,3,4,));
    Tr_absorbed=permute_neighbour_ind(Tr_absorbed,4,5,5);#L,D,R,d,U
    Tr_absorbed=permute_neighbour_ind(Tr_absorbed,3,4,5);#L,D,d,R,U
    Tr_absorbed=permute_neighbour_ind(Tr_absorbed,2,3,5);#L,d,D,R,U
    U,S,V=tsvd(Tr_absorbed,(1,2,),(3,4,5,));#(L_r,d_r),  (D_r,R_r,U_r)
    Tr_keep=U*S;#(L_r,d_r, virtual_r)
    Tr_res=V;#(virtual_r, D_r,R_r,U_r)

    #U,S,V=tsvd(Tu_absorbed,(1,3,4,),(2,5,));#(L_u,R_u,U_u, virtual_u), (virtual_u, D_u,d_u) 
    Tu_absorbed=permute_neighbour_ind(Tu_absorbed,2,3,5);#L,R,D,U,d
    Tu_absorbed=permute_neighbour_ind(Tu_absorbed,3,4,5);#L,R,U,D,d
    Tu_keep=S*V;#(virtual_u, D_u,d_u)
    Tu_res=U;#(L_u,R_u,U_u, virtual_u)

    @tensor TT[:]:=Tu_keep[-4,2,-7]*Tm_absorbed[-1,-2,1,2,-5]*Tr_keep[1,-6,-3];#(virtual_u, D_u,d_u),  (L_m,D_m,R_m,U_m,d_m), (L_r,d_r, virtual_r)
    @tensor TT_new[:]:=TT[-1,-2,-3,-4,1,2,3]*gate[-5,-6,-7,1,2,3];

    U,S,V=tsvd(TT_new,(1,2,3,5,6,),(4,7,); trunc=truncdim(bond_dim));
    U,S,V=Truncations(U,S,V,bond_dim,trun_tol);
    lambda_m_U_new=S;
    Tu_keep_new=permute(V,(1,3,2,));
    @tensor Tu_new[:]:=Tu_keep_new[-2,-5,1]*Tu_res[-1,-3,-4,1];

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