
using LinearAlgebra:diag,I,diagm 


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

# function delet_zero_block(U,Σ,V)
#     if isa(space(Σ,1),ComplexSpace)
#         # pos=findall(diag(Σ).>0);

#         # if Rank(U)==6
#         # end
#         # if Rank(V)==3
#         # end

#         # println(space(V))
#         return U,Σ,V
#     else
#         secs=blocksectors(Σ);
#         sec_length=Vector{Int}(undef,length(secs))
#         U_dict = convert(Dict,U)
#         Σ_dict = convert(Dict,Σ)
#         V_dict = convert(Dict,V)

#         #ProductSpace(Rep[SU₂](0=>3, 1/2=>4, 1=>4, 3/2=>2, 2=>1))

#         for cc =1:length(secs)
#             c=secs[cc];
#             if (size(diag(Σ_dict[:data][string(c)]),1)>0) & (sum(abs.(diag(Σ_dict[:data][string(c)])))>0)
#                 inds=findall(x->(abs.(x).>0), diag(Σ_dict[:data][string(c)]));
#                 U_dict[:data][string(c)]=U_dict[:data][string(c)][:,inds];
#                 Σ_dict[:data][string(c)]=Σ_dict[:data][string(c)][inds,inds];
#                 V_dict[:data][string(c)]=V_dict[:data][string(c)][inds,:];

#                 sec_length[cc]=length(inds);
#             else
#                 delete!(U_dict[:data], string(c))
#                 delete!(V_dict[:data], string(c))
#                 delete!(Σ_dict[:data], string(c))
#                 sec_length[cc]=0;
#             end
#         end

#         #define sector string
#         sec_str="ProductSpace(Rep[SU₂](" *string(((dim(secs[1])-1)/2)) * "=>" * string(sec_length[1]);
#         for cc=2:length(secs)
#             sec_str=sec_str*", " * string(((dim(secs[cc])-1)/2)) * "=>" * string(sec_length[cc]);
#         end
#         sec_str=sec_str*"))"

#         U_dict[:domain]=sec_str
#         V_dict[:codomain]=sec_str
#         Σ_dict[:domain]=sec_str
#         Σ_dict[:codomain]=sec_str

#         return convert(TensorMap, U_dict), convert(TensorMap, Σ_dict), convert(TensorMap, V_dict)
#     end
# end
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
function check_convergence(lambdaset,lambdaset_old,Lx,Ly)
    Lx,Ly=size(lambdaset);
    err_set=Matrix{Float64}(undef,Lx,Ly);
    for ca=1:Lx
        for cb=1:Ly
            if isassigned(lambdaset_old,ca,cb)
                es1=convert(Array,lambdaset[ca,cb]);
                es2=convert(Array,lambdaset_old[ca,cb]);
                if size(es1)==size(es2)
                    err_set[ca,cb]=norm(es1-es2);
                else
                    err_set[ca,cb]=100;
                end
            else
                err_set[ca,cb]=0;
            end
        end
    end

    return err_set
end


function H_RU_LD_RD(tx,ty,t2,ϕ,U, dt, space_type,Lx)

    if space_type==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
    elseif space_type==GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
    elseif space_type==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_Z2();
    end
    
    # t1=parameters["t1"];
    # t2=parameters["t2"];
    # ϕ=parameters["ϕ"];
    # U=parameters["U"];

    tx_coe_set=[exp(im*ϕ),exp(im*ϕ)]*tx;
    ty_coe_set=[1,-1]*ty;
    t2_coe_set=[1,-1]*t2;
    U_coe=U;

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
        # @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
        # @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
        @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
        # hh_U=(hh_LD+hh_RU+hh_RD)*U_coe;
        hh_U=(hh_RD)*U_coe;
        
        #################
        hh=permute(hh_tx+hh_ty+hh_t2+hh_U,(1,2,3,),(4,5,6,));#hh_tx+hh_ty+hh_t2+hh_U
        gate_set[cx]=hh;
    end
    return gate_set
end

function gate_RU_LD_RD(tx,ty,t2,ϕ,U, dt, space_type,Lx)

    if space_type==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
    elseif space_type==GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
    elseif space_type==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
        Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_Z2();
    end
    
    # t1=parameters["t1"];
    # t2=parameters["t2"];
    # ϕ=parameters["ϕ"];
    # U=parameters["U"];

    tx_coe_set=[exp(im*ϕ),exp(im*ϕ)]*tx;
    ty_coe_set=[1,-1]*ty;
    t2_coe_set=[1,-1]*t2;
    U_coe=U;

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
        # @tensor hh_LD[:]:=OU_LD[-1,-4]*Id_RD[-2,-5]*Id_RU[-3,-6];
        # @tensor hh_RU[:]:=Id_LD[-1,-4]*Id_RD[-2,-5]*OU_RU[-3,-6];
        @tensor hh_RD[:]:=Id_LD[-1,-4]*OU_RD[-2,-5]*Id_RU[-3,-6];
        # hh_U=(hh_LD+hh_RU+hh_RD)*U_coe;
        hh_U=(hh_RD)*U_coe;
        
        #################
        hh=permute(hh_tx+hh_ty+hh_t2+hh_U,(1,2,3,),(4,5,6,));#hh_tx+hh_ty+hh_t2+hh_U
        eu,ev=eigh(hh);
        gate=ev*exp(-dt*eu)*ev';
        gate_set[cx]=gate;
    end
    return gate_set
end


function spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V, dt, space_type,Lx)

    if space_type==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
        # Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
    # elseif space_type==GradedSpace{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{TensorKit.ProductSector{Tuple{U1Irrep, SU2Irrep}}, Int64}}
    #     Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_U1_SU2();
    elseif space_type==GradedSpace{Z2Irrep, Tuple{Int64, Int64}}
        Ident, N_occu, Cdag, C, _ =operators_spinless_Z2();
    elseif space_type==GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}
        Ident, N_occu, Cdag, C, _ =operators_spinless_U1();
    end
    
    # t1=parameters["t1"];
    # t2=parameters["t2"];
    # ϕ=parameters["ϕ"];
    # V=parameters["V"];

    tx_coe_set=[exp(im*ϕ),exp(im*ϕ)]*tx;
    ty_coe_set=[1,-1]*ty;
    t2_coe_set=[1,-1]*t2;
    V_coe=V;

    gate_set=Matrix{TensorMap}(undef,2,1);
    for cx=1:2;
        ####################
        O1=Cdag;
        O2=C;
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*tx_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh_tx[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        ######################
        O1=Cdag;
        O2=C;
        @tensor op[:]:=O1[1,-1,-3]*O2[1,-2,-4];
        op=op*ty_coe_set[cx];
        op=permute(op,(1,2,),(3,4,));
        hh=op+op';
        # Id=unitary(space(Cdag_set[mod1(cx,Lx)],2),space(Cdag_set[mod1(cx,Lx)],2));
        Id=unitary(space(Cdag,2),space(Cdag,2));
        @tensor hh_ty[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        #####################
        O1=Cdag;
        O2=C;
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


        ####################
        O1=N_occu;
        O2=N_occu;
        @tensor op[:]:=O1[-1,-3]*O2[-2,-4];
        op=op*V_coe;
        op=permute(op,(1,2,),(3,4,));
        hh=op;
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh_nnx[:]:=hh[-1,-2,-4,-5]*Id[-3,-6];
        ######################
        O1=N_occu;
        O2=N_occu;
        @tensor op[:]:=O1[-1,-3]*O2[-2,-4];
        op=op*V_coe;
        op=permute(op,(1,2,),(3,4,));
        hh=op;
        Id=unitary(space(Cdag,2),space(Cdag,2));
        @tensor hh_nny[:]:=hh[-2,-3,-5,-6]*Id[-1,-4];
        #####################
        O1=N_occu;
        O2=N_occu;
        @tensor op[:]:=O1[-1,-3]*O2[-2,-4];
        op=op*V_coe;
        op=permute(op,(1,2,),(3,4,));
        hh=op;
        Id=unitary(space(hh,2),space(hh,2));
        @tensor hh_nn_ld_ru[:]:=hh[-1,-3,-4,-6]*Id[-2,-5];
        #################
        
        #################
        hh=permute(hh_tx+hh_ty+hh_t2+hh_nnx+hh_nny+hh_nn_ld_ru,(1,2,3,),(4,5,6,));
        eu,ev=eigh(hh);
        gate=ev*exp(-dt*eu)*ev';
        gate_set[cx]=gate;
    end
    return gate_set
end

function get_triangles(Lx,Ly)
    #  A
    # /|
    #B-C
    #save coordinate of B site in a triangle

    triangle_set1=Vector{Tuple}(undef,0);
    triangle_set2=Vector{Tuple}(undef,0);
    triangle_set3=Vector{Tuple}(undef,0);
    triangle_set4=Vector{Tuple}(undef,0);

    for cx=0:Lx-1
        for cy=1:Ly
            if mod(cx,2)==0
                if mod(cy,2)==0
                    triangle_set1=vcat(triangle_set1,(cx,cy,));
                elseif mod(cy,2)==1
                    triangle_set2=vcat(triangle_set2,(cx,cy,));
                end
            elseif mod(cx,2)==1
                if mod(cy,2)==0
                    triangle_set3=vcat(triangle_set3,(cx,cy,));
                elseif mod(cy,2)==1
                    triangle_set4=vcat(triangle_set4,(cx,cy,));
                end
            end
        end
    end

    @assert length(triangle_set1)+length(triangle_set2)+length(triangle_set3)+length(triangle_set4)==Lx*Ly;
    all_triangles=(triangle_set1,triangle_set2,triangle_set3,triangle_set4,)
    return all_triangles
end

function triangle_gate_iPESS_simplified(D_max, op_LD_RD_RU, T1, T2, T3, B, trun_tol)
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

    T1=permute_neighbour_ind(T1,2,3,4);#M1, R1, d1,  D1
    Ut1=unitary(fuse(space(T1,1)*space(T1,2)), space(T1,1)*space(T1,2));
    @tensor T1[:]:=Ut1[-1,1,2]*T1[1,2,-2,-3];#M1_R1, d1,  D1
    uu,ss,vv=tsvd(permute(T1,(1,),(2,3,)));
    T1_res=uu;
    T1_keep=ss*vv;

    T2=permute_neighbour_ind(T2,3,4,4);#M2, d2, D2, R2
    T2=permute_neighbour_ind(T2,2,3,4);#M2, D2, d2, R2
    Ut2=unitary(fuse(space(T2,1)*space(T2,2)), space(T2,1)*space(T2,2));
    @tensor T2[:]:=Ut2[-1,1,2]*T2[1,2,-2,-3];#M2_D2, d2, R2
    uu,ss,vv=tsvd(permute(T2,(1,),(2,3,)));
    T2_res=uu;
    T2_keep=ss*vv;

    T3=T3;#M3, d3, R3, D3 
    Ut3=unitary(fuse(space(T3,3)*space(T3,4)), space(T3,3)*space(T3,4));
    @tensor T3[:]:=T3[-1,-2,1,2]*Ut3[-3,1,2];#M3, d3, R3_D3 
    uu,ss,vv=tsvd(permute(T3,(1,2,),(3,)));
    T3_keep=uu*ss;
    T3_res=vv;

    @tensor T2_B[:]:=T2_keep[-1,-2,1]*B[1,-3,-4];     #(M2_D2, d2, R2),  (R2, D1, M3) => (M2_D2, d2, D1, M3)
    T2_B=permute_neighbour_ind(T2_B,2,3,4);#(M2_D2, D1, d2, M3)
    T2_B=permute_neighbour_ind(T2_B,1,2,4);#(D1, M2_D2, d2, M3)
    @tensor T1_T2_B[:]:=T1_keep[-1,-2,1]*T2_B[1,-3,-4,-5];#(M1_R1, d1,  D1), (D1, M2_D2, d2, M3) => (M1_R1, d1, M2_D2, d2, M3)

    @tensor T1_T2_B_T3[:]:=T1_T2_B[-1,-2,-3,-4,1]*T3_keep[1,-5,-6];#(M1_R1, d1, M2_D2, d2, M3), (M3, d3, R3_D3) => (M1_R1, d1, M2_D2, d2, d3, R3_D3)
    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3

    #d2',d3',d1', d2,d3,d1
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,5,6,6);#d2',d3',d1', d2,d1,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,4,5,6);#d2',d3',d1', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,2,3,6);#d2',d1',d3', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,1,2,6);#d1',d2',d3', d1,d2,d3
    @tensor T1_T2_B_T3[:]:=T1_T2_B_T3[-1,-2,1,2,3,-6]*op_LD_RD_RU[-3,-4,-5,1,2,3];


    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, d1, M2_D2, d2, d3, R3_D3
    T1_T2_B_T3=T1_T2_B_T3/norm(T1_T2_B_T3);
    if isa(space(T1,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        U1,S1,V1=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
        U3,S3,V3=tsvd(T1_T2_B_T3,(1,2,3,4,),(5,6,);trunc=truncdim(D_max));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
    elseif isa(space(T1,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        U1,S1,V1=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
        U1,S1,V1=Truncations(U1,S1,V1,D_max,trun_tol);#println(norm(U1*S1*V1-M_old)/norm(M_old))
        U3,S3,V3=tsvd(T1_T2_B_T3,(1,2,3,4,),(5,6,));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
        U3,S3,V3=Truncations(U3,S3,V3,D_max,trun_tol);#println(norm(U3*S3*V3-M_old)/norm(M_old))
    elseif isa(space(T1,1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        U1,S1,V1=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
        U1,S1,V1=Truncations(U1,S1,V1,D_max,trun_tol);#println(norm(U1*S1*V1-M_old)/norm(M_old))
        U3,S3,V3=tsvd(T1_T2_B_T3,(1,2,3,4,),(5,6,));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
        U3,S3,V3=Truncations(U3,S3,V3,D_max,trun_tol);#println(norm(U3*S3*V3-M_old)/norm(M_old))
    end


    # T1_T2_B_T3_a=permute_neighbour_ind(T1_T2_B_T3,1,2,6);
    # T1_T2_B_T3_b=permute_neighbour_ind_Rank6_fast(T1_T2_B_T3,1,2);
    # @assert norm(T1_T2_B_T3_a-T1_T2_B_T3_b)/norm(T1_T2_B_T3_a)<1e-14;

    # T1_T2_B_T3_a=permute_neighbour_ind(T1_T2_B_T3,2,3,6);
    # T1_T2_B_T3_b=permute_neighbour_ind_Rank6_fast(T1_T2_B_T3,2,3);
    # @assert norm(T1_T2_B_T3_a-T1_T2_B_T3_b)/norm(T1_T2_B_T3_a)<1e-14;

    # T1_T2_B_T3_a=permute_neighbour_ind(T1_T2_B_T3,3,4,6);
    # T1_T2_B_T3_b=permute_neighbour_ind_Rank6_fast(T1_T2_B_T3,3,4);
    # @assert norm(T1_T2_B_T3_a-T1_T2_B_T3_b)/norm(T1_T2_B_T3_a)<1e-14;

    # T1_T2_B_T3_a=permute_neighbour_ind(T1_T2_B_T3,4,5,6);
    # T1_T2_B_T3_b=permute_neighbour_ind_Rank6_fast(T1_T2_B_T3,4,5);
    # @assert norm(T1_T2_B_T3_a-T1_T2_B_T3_b)/norm(T1_T2_B_T3_a)<1e-14;

    # T1_T2_B_T3_a=permute_neighbour_ind(T1_T2_B_T3,5,6,6);
    # T1_T2_B_T3_b=permute_neighbour_ind_Rank6_fast(T1_T2_B_T3,5,6);
    # @assert norm(T1_T2_B_T3_a-T1_T2_B_T3_b)/norm(T1_T2_B_T3_a)<1e-14;



    # T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3
    # T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,1,2,6);# M2_D2, M1_R1, d1, d2, d3, R3_D3
    # T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,3,4,6);# M2_D2, M1_R1, d2, d1, d3, R3_D3
    # T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M2_D2, d2, M1_R1, d1, d3, R3_D3

    T1_T2_B_T3=permute_neighbour_ind_Rank6_fast(T1_T2_B_T3,2,3);# M1_R1, M2_D2, d1, d2, d3, R3_D3
    T1_T2_B_T3=permute_neighbour_ind_Rank6_fast(T1_T2_B_T3,1,2);# M2_D2, M1_R1, d1, d2, d3, R3_D3
    T1_T2_B_T3=permute_neighbour_ind_Rank6_fast(T1_T2_B_T3,3,4);# M2_D2, M1_R1, d2, d1, d3, R3_D3
    T1_T2_B_T3=permute_neighbour_ind_Rank6_fast(T1_T2_B_T3,2,3);# M2_D2, d2, M1_R1, d1, d3, R3_D3

    T1_T2_B_T3=T1_T2_B_T3/norm(T1_T2_B_T3);
    if isa(space(T1,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        U2,S2,V2=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
    elseif isa(space(T1,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        U2,S2,V2=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
        U2,S2,V2=Truncations(U2,S2,V2,D_max,trun_tol);#println(norm(U2*S2*V2-M_old)/norm(M_old))
    elseif isa(space(T1,1), GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
        U2,S2,V2=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
        U2,S2,V2=Truncations(U2,S2,V2,D_max,trun_tol);#println(norm(U2*S2*V2-M_old)/norm(M_old))
    end

    λ_2_new=permute(S2,(2,),(1,));

    @tensor T2_new[:]:=T2_res[-1,1]*U2[1,-2,2]*S2[2,-3];#(M2_D2, d2, R2_new)
    T1_B_T3=V2;#(R2_new, M1_R1, d1, d3, R3_D3)

    S3_inv=my_pinv(S3);
    @tensor T1_B[:]:=T1_B_T3[-1,-2,-3,1,2]*V3'[1,2,3]*S3_inv'[3,-4];#(R2_new, M1_R1, d1, M3_new) 
    @tensor T3_new[:]:=S3[-1,2]*V3[2,-2,1]*T3_res[1,-3];#(M3_new, d3, R3_D3)
    λ_3_new=S3;

    T1_B=permute_neighbour_ind(T1_B,1,2,4);#(M1_R1, R2_new, d1, M3_new) 
    T1_B=permute_neighbour_ind(T1_B,2,3,4);#(M1_R1, d1, R2_new, M3_new) 
    S1_inv=my_pinv(S1);
    @tensor B_new[:]:=S1_inv'[-1,3]*U1'[3,1,2]*T1_B[1,2,-2,-3];#(D1_new, R2_new, M3_new) 
    @tensor T1_new[:]:=T1_res[-1,1]*U1[1,-2,2]*S1[2,-3];#(M1_R1, d1, D1_new) 
    λ_1_new=permute(S1,(2,),(1,));

    #B_new: (D1_new, R2_new, M3_new) => (R2, D1, M3)
    B_new=permute_neighbour_ind(B_new,1,2,3);#(R2_new, D1_new, M3_new)
    B_new=permute(B_new,(1,2,),(3,));

    #T1_new: (M1_R1, d1, D1_new) => (M1, d1, R1, D1)
    @tensor T1_new[:]:=T1_new[1,-3,-4]*Ut1'[-1,-2,1];#(M1, R1, d1, D1_new)
    T1_new=permute_neighbour_ind(T1_new,2,3,4);#(M1, d1, R1, D1_new)

    #T2_new: (M2_D2, d2, R2_new) => (M2, d2, R2, D2) 
    @tensor T2_new[:]:=T2_new[1,-3,-4]*Ut2'[-1,-2,1];#(M2, D2, d2, R2_new)
    T2_new=permute_neighbour_ind(T2_new,2,3,4);#(M2, d2, D2, R2_new)
    T2_new=permute_neighbour_ind(T2_new,3,4,4);#(M2, d2, R2_new, D2)

    #T3_new: (M3_new, d3, R3_D3) => (M3, d3, R3, D3)
    @tensor T3_new[:]:=T3_new[-1,-2,1]*Ut3'[-3,-4,1];#(M3_new, d3, R3, D3)

    return T1_new, T2_new, T3_new, B_new, λ_1_new, λ_2_new, λ_3_new
end




function triangle_update_iPESS(D_max,ct,Bset, Tset, lambdaset1, lambdaset2, lambdaset3, gates_bulk,gates_left,gates_top,gates_left_top, trun_tol)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    Lx,Ly=size(Bset);
    all_triangles=get_triangles(Lx,Ly);
    for tran_set_id=1:length(all_triangles)
        trangle_set=all_triangles[tran_set_id];

        for cs=1:length(trangle_set)
            trangle_coord=trangle_set[cs];
            # B
            #CD
            posTB=[trangle_coord[1]+1,trangle_coord[2]+1];
            posTD=[trangle_coord[1]+1,trangle_coord[2]];
            posTC=[trangle_coord[1],trangle_coord[2]];
            posBond=posTD;

            #absorb λ tensors into physical tensors
            # λ tensors not used for update, only used for checking convergence
            if (trangle_coord[1]>0)&&(trangle_coord[2]<Ly) #bulk triangle

                TB=Tset[posTB[1],posTB[2]];
                TC=Tset[posTC[1],posTC[2]];
                TD=Tset[posTD[1],posTD[2]];

                B=Bset[posBond[1],posBond[2]];

                TB, TC, TD, B, lambda_1, lambda_2, lambda_3=triangle_gate_iPESS_simplified(D_max,gates_bulk[mod1(trangle_coord[1],2)], TB, TC, TD, B, trun_tol);

                TB=permute(TB,(1,),(2,3,4,));
                TC=permute(TC,(1,),(2,3,4,));
                TD=permute(TD,(1,),(2,3,4,));

                TB=TB/norm(TB);
                TC=TC/norm(TC);
                TD=TD/norm(TD);
                B=B/norm(B);
                lambda_1=lambda_1/norm(lambda_1);
                lambda_2=lambda_2/norm(lambda_2);
                lambda_3=lambda_3/norm(lambda_3);
                
                lambdaset1[posTD[1],posTD[2]]=lambda_1;
                lambdaset2[posTD[1],posTD[2]]=lambda_2;
                lambdaset3[posTD[1],posTD[2]]=lambda_3;
                Tset[posTB[1],posTB[2]]=TB;
                Tset[posTC[1],posTC[2]]=TC;
                Tset[posTD[1],posTD[2]]=TD;
                Bset[posBond[1],posBond[2]]=B;

            elseif (trangle_coord[1]==0)&&(trangle_coord[2]<Ly) #left triangle
                #TC is dummy
                TB=Tset[posTB[1],posTB[2]];
                #TC=Tset[posTC[1],posTC[2]];
                TD=Tset[posTD[1],posTD[2]];

                B=Bset[posBond[1],posBond[2]];

                #define trivial TC:
                Vp=space(TD)[2];
                Vv=space(B)[1];
                if isa(Vp,GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
                    V_trivial=Rep[SU₂](0=>1);
                elseif isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
                    V_trivial=Rep[U₁](0=>1);
                end
                TC=TensorMap(randn,V_trivial,Vp'*Vv*V_trivial);
                TC=TC/norm(TC);
                
                TB, TC, TD, B, lambda_1, lambda_2, lambda_3=triangle_gate_iPESS_simplified(D_max,gates_left[mod1(trangle_coord[1],2)], TB, TC, TD, B, trun_tol);

                TB=permute(TB,(1,),(2,3,4,));
                #TC=permute(TC,(1,),(2,3,4,));
                TD=permute(TD,(1,),(2,3,4,));

                TB=TB/norm(TB);
                #TC=TC/norm(TC);
                TD=TD/norm(TD);
                B=B/norm(B);

                Tset[posTB[1],posTB[2]]=TB;
                #Tset[posTC[1],posTC[2]]=TC;
                Tset[posTD[1],posTD[2]]=TD;
                Bset[posBond[1],posBond[2]]=B;

            elseif (trangle_coord[1]>0)&&(trangle_coord[2]==Ly) #top triangle
                #TB is dummy
                #TB=Tset[posTB[1],posTB[2]];
                TC=Tset[posTC[1],posTC[2]];
                TD=Tset[posTD[1],posTD[2]];

                B=Bset[posBond[1],posBond[2]];

                #define trivial TB:
                Vp=space(TD)[2];
                Vv=space(B)[2];
                if isa(Vp,GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
                    V_trivial=Rep[SU₂](0=>1);
                elseif isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
                    V_trivial=Rep[U₁](0=>1);
                end
                TB=TensorMap(randn,V_trivial,Vp'*V_trivial*Vv);
                TB=TB/norm(TB);

                
                TB, TC, TD, B, lambda_1, lambda_2, lambda_3=triangle_gate_iPESS_simplified(D_max,gates_top[mod1(trangle_coord[1],2)], TB, TC, TD, B, trun_tol);

                #TB=permute(TB,(1,),(2,3,4,));
                TC=permute(TC,(1,),(2,3,4,));
                TD=permute(TD,(1,),(2,3,4,));

                #TB=TB/norm(TB);
                TC=TC/norm(TC);
                TD=TD/norm(TD);
                B=B/norm(B);

                #Tset[posTB[1],posTB[2]]=TB;
                Tset[posTC[1],posTC[2]]=TC;
                Tset[posTD[1],posTD[2]]=TD;
                Bset[posBond[1],posBond[2]]=B;

            elseif (trangle_coord[1]==0)&&(trangle_coord[2]==Ly) #left-top corner

                #TB, TC are dummy
                #TB=Tset[posTB[1],posTB[2]];
                #TC=Tset[posTC[1],posTC[2]];
                TD=Tset[posTD[1],posTD[2]];

                B=Bset[posBond[1],posBond[2]];

                #define trivial TC:
                Vp=space(TD)[2];
                Vv=space(B)[2];
                if isa(Vp,GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
                    V_trivial=Rep[SU₂](0=>1);
                elseif isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
                    V_trivial=Rep[U₁](0=>1);
                end
                TC=TensorMap(randn,V_trivial,Vp'*Vv*V_trivial);
                TC=TC/norm(TC);

                #define trivial TB:
                Vp=space(TD)[2];
                Vv=space(B)[1];
                if isa(Vp,GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
                    V_trivial=Rep[SU₂](0=>1);
                elseif isa(Vp,GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}})
                    V_trivial=Rep[U₁](0=>1);
                end
                TB=TensorMap(randn,V_trivial,Vp'*V_trivial*Vv);
                TB=TB/norm(TB);


                
                TB, TC, TD, B, lambda_1, lambda_2, lambda_3=triangle_gate_iPESS_simplified(D_max,gates_left_top[mod1(trangle_coord[1],2)], TB, TC, TD, B, trun_tol);

                #TB=permute(TB,(1,),(2,3,4,));
                #TC=permute(TC,(1,),(2,3,4,));
                TD=permute(TD,(1,),(2,3,4,));

                #TB=TB/norm(TB);
                #TC=TC/norm(TC);
                TD=TD/norm(TD);
                B=B/norm(B);

                #Tset[posTB[1],posTB[2]]=TB;
                #Tset[posTC[1],posTC[2]]=TC;
                Tset[posTD[1],posTD[2]]=TD;
                Bset[posBond[1],posBond[2]]=B;
            else
                error("unknown case")
            end



        end
    end

    return Bset, Tset, lambdaset1, lambdaset2, lambdaset3
end






function tebd_PESS(parameters, Bset, Tset, lambdaset1, lambdaset2, lambdaset3,  tau, dt, Dmax, trun_tol)
    tol=dt*1e-3;#for determining convergence 
    println("tau, dt="*string([tau,dt]))
    # println("one step")
    # println(space(T_u))
    # println(space(T_d))
    Lx,Ly=size(Tset);
    lambdaset1_old=deepcopy(lambdaset1);
    lambdaset2_old=deepcopy(lambdaset2);
    lambdaset3_old=deepcopy(lambdaset3);

    tx=parameters["t1"];
    ty=parameters["t1"];
    t2=parameters["t2"];
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_bulk=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=parameters["t1"];
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_top=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=parameters["t1"];
    t2=0;
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_left=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_left_top=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    for ct=1:Int(round(tau/abs(dt)))
        #println("iteration "*string(ct));flush(stdout)
        Bset, Tset, lambdaset1, lambdaset2, lambdaset3= triangle_update_iPESS(Dmax, ct, Bset, Tset, lambdaset1, lambdaset2, lambdaset3, gates_ru_ld_rd_bulk,gates_ru_ld_rd_left,gates_ru_ld_rd_top,gates_ru_ld_rd_left_top, trun_tol);
        err_1=check_convergence(lambdaset1,lambdaset1_old,Lx,Ly);
        err_2=check_convergence(lambdaset2,lambdaset2_old,Lx,Ly);
        err_3=check_convergence(lambdaset3,lambdaset3_old,Lx,Ly);
        er=max(maximum(err_1),maximum(err_2),maximum(err_3));
        if mod(ct,20)==0
            println("iteration "*string(ct)*", convergence= "*string(er));flush(stdout)
        end
        if mod(ct,40)==0
            for tt in Bset
                println(space(tt))
            end
        end
        if er<tol
            break;
        end
        lambdaset1_old=deepcopy(lambdaset1);
        lambdaset2_old=deepcopy(lambdaset2);
        lambdaset3_old=deepcopy(lambdaset3);
    end
    return Bset, Tset, lambdaset1, lambdaset2, lambdaset3
end



function tebd_PESS_spinless(parameters, Bset, Tset, lambdaset1, lambdaset2, lambdaset3,  tau, dt, Dmax, trun_tol)
    tol=dt*1e-3;#for determining convergence 
    println("tau, dt="*string([tau,dt]))
    # println("one step")
    # println(space(T_u))
    # println(space(T_d))
    Lx,Ly=size(Tset);
    lambdaset1_old=deepcopy(lambdaset1);
    lambdaset2_old=deepcopy(lambdaset2);
    lambdaset3_old=deepcopy(lambdaset3);

    tx=parameters["t1"];
    ty=parameters["t1"];
    t2=parameters["t2"];
    ϕ=parameters["ϕ"];
    V=parameters["V"];
    gates_ru_ld_rd_bulk=spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V,dt, typeof(space(Bset[1],1)),Lx);

    tx=parameters["t1"];
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    V=parameters["V"];
    gates_ru_ld_rd_top=spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=parameters["t1"];
    t2=0;
    ϕ=parameters["ϕ"];
    V=parameters["V"];
    gates_ru_ld_rd_left=spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    V=parameters["V"];
    gates_ru_ld_rd_left_top=spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V,dt, typeof(space(Bset[1],1)),Lx);

    for ct=1:Int(round(tau/abs(dt)))
        #println("iteration "*string(ct));flush(stdout)
        Bset, Tset, lambdaset1, lambdaset2, lambdaset3= triangle_update_iPESS(Dmax, ct, Bset, Tset, lambdaset1, lambdaset2, lambdaset3, gates_ru_ld_rd_bulk,gates_ru_ld_rd_left,gates_ru_ld_rd_top,gates_ru_ld_rd_left_top, trun_tol);
        err_1=check_convergence(lambdaset1,lambdaset1_old,Lx,Ly);
        err_2=check_convergence(lambdaset2,lambdaset2_old,Lx,Ly);
        err_3=check_convergence(lambdaset3,lambdaset3_old,Lx,Ly);
        er=max(maximum(err_1),maximum(err_2),maximum(err_3));
        if mod(ct,20)==0
            println("iteration "*string(ct)*", convergence= "*string(er));flush(stdout)
        end
        if mod(ct,40)==0
            for tt in Bset
                println(space(tt))
            end
        end
        if er<tol
            break;
        end
        lambdaset1_old=deepcopy(lambdaset1);
        lambdaset2_old=deepcopy(lambdaset2);
        lambdaset3_old=deepcopy(lambdaset3);
    end
    return Bset, Tset, lambdaset1, lambdaset2, lambdaset3
end


function tebd_noHam_spinless(Bset, Tset, lambdaset1, lambdaset2, lambdaset3, ite_num, trun_tol)
    Dmax1=get_max_dim_general(Bset);
    Dmax2=get_max_dim_general(Tset);
    Dmax=max(Dmax1,Dmax2);

    tol=1e-6;#for determining convergence 

    # println("one step")
    # println(space(T_u))
    # println(space(T_d))
    Lx,Ly=size(Tset);
    lambdaset1_old=deepcopy(lambdaset1);
    lambdaset2_old=deepcopy(lambdaset2);
    lambdaset3_old=deepcopy(lambdaset3);



    tx=0;
    ty=0;
    t2=0;
    ϕ=0;
    V=0;
    dt=0;
    gates_ru_ld_rd=spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V,dt, typeof(space(Bset[1],1)),Lx);

    for ct=1:ite_num
        #println("iteration "*string(ct));flush(stdout)
        Bset, Tset, lambdaset1, lambdaset2, lambdaset3= triangle_update_iPESS(Dmax, ct, Bset, Tset, lambdaset1, lambdaset2, lambdaset3, gates_ru_ld_rd,gates_ru_ld_rd,gates_ru_ld_rd,gates_ru_ld_rd, trun_tol);
        err_1=check_convergence(lambdaset1,lambdaset1_old,Lx,Ly);
        err_2=check_convergence(lambdaset2,lambdaset2_old,Lx,Ly);
        err_3=check_convergence(lambdaset3,lambdaset3_old,Lx,Ly);
        er=max(maximum(err_1),maximum(err_2),maximum(err_3));
        if mod(ct,20)==0
            println("iteration "*string(ct)*", convergence= "*string(er));flush(stdout)
        end

        if mod(ct,40)==0
            for tt in Bset
                println(space(tt))
            end
        end
        if er<tol
            break;
        end
        lambdaset1_old=deepcopy(lambdaset1);
        lambdaset2_old=deepcopy(lambdaset2);
        lambdaset3_old=deepcopy(lambdaset3);
    end
    return Bset, Tset, lambdaset1, lambdaset2, lambdaset3
end



function tebd_real_time_PESS_spinless(save_filenm, parameters, Bset, Tset, lambdaset1, lambdaset2, lambdaset3,  tau, dt, Dmax, trun_tol)
    println("real time dynamics with extra particle")
    tol=dt*1e-3;#for determining convergence 
    println("tau, dt="*string([tau,dt]))
    # println("one step")
    # println(space(T_u))
    # println(space(T_d))
    Lx,Ly=size(Tset);
    lambdaset1_old=deepcopy(lambdaset1);
    lambdaset2_old=deepcopy(lambdaset2);
    lambdaset3_old=deepcopy(lambdaset3);

    tx=parameters["t1"];
    ty=parameters["t1"];
    t2=parameters["t2"];
    ϕ=parameters["ϕ"];
    V=parameters["V"];
    gates_ru_ld_rd_bulk=spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V,dt*im, typeof(space(Bset[1],1)),Lx);

    tx=parameters["t1"];
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    V=parameters["V"];
    gates_ru_ld_rd_top=spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V,dt*im, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=parameters["t1"];
    t2=0;
    ϕ=parameters["ϕ"];
    V=parameters["V"];
    gates_ru_ld_rd_left=spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V,dt*im, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    V=parameters["V"];
    gates_ru_ld_rd_left_top=spinless_gate_RU_LD_RD(tx,ty,t2,ϕ,V,dt*im, typeof(space(Bset[1],1)),Lx);

    obs_times=Vector{Number}(undef,0);
    obs_hop_x=Vector{Matrix}(undef,0);
    obs_hop_y=Vector{Matrix}(undef,0);
    obs_hop_ld_ru=Vector{Matrix}(undef,0);
    obs_hop_occu=Vector{Matrix}(undef,0);
    obs_E=Vector{Number}(undef,0);
    for ct=0:Int(round(tau/abs(dt)))
        println("iteration "*string(ct));flush(stdout)


        if abs(rem(ct*dt,0.1, RoundNearest))<1e-12
            psi=B_T_sets_to_PESS(Bset,Tset);
            psi_PEPS=PESS_to_PEPS_matrix(psi);
            psi_double,UL_set,UD_set,UR_set,UU_set=construct_double_layer_swap(psi_PEPS,Lx,Ly);
            E_total,Ex_set,Ey_set,E_ld_ru_set, NNx_set,NNy_set,NN_ld_ru_set, occu_set=energy_disk_old(psi_PEPS,psi_double)

            push!(obs_times,ct*dt);
            push!(obs_hop_x,Ex_set);
            push!(obs_hop_y,Ey_set);
            push!(obs_hop_ld_ru,E_ld_ru_set);
            push!(obs_hop_occu,occu_set);
            push!(obs_E,E_total);

            matwrite(save_filenm*".mat", Dict(
                "obs_times" => obs_times,
                "obs_hop_x"=>obs_hop_x,
                "obs_hop_y"=>obs_hop_y,
                "obs_hop_ld_ru"=>obs_hop_ld_ru,
                "obs_hop_occu"=>obs_hop_occu,
                "obs_E"=>obs_E,
              ); compress = false)
        end

          





        Bset, Tset, lambdaset1, lambdaset2, lambdaset3= triangle_update_iPESS(Dmax, ct, Bset, Tset, lambdaset1, lambdaset2, lambdaset3, gates_ru_ld_rd_bulk,gates_ru_ld_rd_left,gates_ru_ld_rd_top,gates_ru_ld_rd_left_top, trun_tol);
        err_1=check_convergence(lambdaset1,lambdaset1_old,Lx,Ly);
        err_2=check_convergence(lambdaset2,lambdaset2_old,Lx,Ly);
        err_3=check_convergence(lambdaset3,lambdaset3_old,Lx,Ly);
        er=max(maximum(err_1),maximum(err_2),maximum(err_3));
        if mod(ct,20)==0
            println("iteration "*string(ct)*", convergence= "*string(er));flush(stdout)
        end
        if mod(ct,40)==0
            for tt in Bset
                println(space(tt))
            end
        end
        if er<tol
            break;
        end
        lambdaset1_old=deepcopy(lambdaset1);
        lambdaset2_old=deepcopy(lambdaset2);
        lambdaset3_old=deepcopy(lambdaset3);






    end
    return Bset, Tset, lambdaset1, lambdaset2, lambdaset3
end




function get_max_dim_general(psi::Matrix{TensorMap})
    #get maximum bond dimension of a PEPS
    dim_max=0;
    for cc in eachindex(psi)
        T=psi[cc];
        if Rank(T)==5 # rank 5 PEPS tensor
            dim_m=maximum([dim(space(T,1)), dim(space(T,2)), dim(space(T,3)), dim(space(T,4))]);
            dim_max=max(dim_max,dim_m);
        elseif Rank(T)==4 #PESS tensor
            dim_m=maximum([dim(space(T,1)), dim(space(T,3)), dim(space(T,4))]);
            dim_max=max(dim_max,dim_m);
        elseif Rank(T)==3 #PESS tensor
            dim_m=maximum([dim(space(T,1)), dim(space(T,2)), dim(space(T,3))]);
            dim_max=max(dim_max,dim_m);
        end
    end
    return dim_max
end


function tebd_PESS_no_Hamiltonian(parameters, Bset, Tset, lambdaset1, lambdaset2, lambdaset3,  Nstep, trun_tol)
    tol=1e-6;

    Dmax1=get_max_dim_general(Bset);
    Dmax2=get_max_dim_general(Tset);
    Dmax=max(Dmax1,Dmax2);

    # println("one step")
    # println(space(T_u))
    # println(space(T_d))
    Lx,Ly=size(Tset);
    lambdaset1_old=deepcopy(lambdaset1);
    lambdaset2_old=deepcopy(lambdaset2);
    lambdaset3_old=deepcopy(lambdaset3);

    dt=0;

    tx=parameters["t1"];
    ty=parameters["t1"];
    t2=parameters["t2"];
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_bulk=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=parameters["t1"];
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_top=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=parameters["t1"];
    t2=0;
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_left=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);

    tx=0;
    ty=0;
    t2=0;
    ϕ=parameters["ϕ"];
    U=parameters["U"];
    gates_ru_ld_rd_left_top=gate_RU_LD_RD(tx,ty,t2,ϕ,U,dt, typeof(space(Bset[1],1)),Lx);


    for ct=1:Nstep
        #println("iteration "*string(ct));flush(stdout)
        Bset, Tset, lambdaset1, lambdaset2, lambdaset3= triangle_update_iPESS(Dmax, ct, Bset, Tset, lambdaset1, lambdaset2, lambdaset3, gates_ru_ld_rd_bulk,gates_ru_ld_rd_left,gates_ru_ld_rd_top,gates_ru_ld_rd_left_top, trun_tol);
        err_1=check_convergence(lambdaset1,lambdaset1_old,Lx,Ly);
        err_2=check_convergence(lambdaset2,lambdaset2_old,Lx,Ly);
        err_3=check_convergence(lambdaset3,lambdaset3_old,Lx,Ly);
        er=max(maximum(err_1),maximum(err_2),maximum(err_3));
        if mod(ct,20)==0
            println("iteration "*string(ct)*", convergence= "*string(er));flush(stdout)
        end
        if mod(ct,40)==0
            for tt in Bset
                println(space(tt))
            end
        end
        if er<tol
            break;
        end
        lambdaset1_old=deepcopy(lambdaset1);
        lambdaset2_old=deepcopy(lambdaset2);
        lambdaset3_old=deepcopy(lambdaset3);
    end
    return Bset, Tset, lambdaset1, lambdaset2, lambdaset3
end

