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
    # sM=sM/norm(sM)
    return uM,sM,vM
end










function update_up_triangle(op_LD_RD_RU, T1, T2, T3, B, trun_tol)
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

    T2=permute_neighbour_ind(T2,3,4,4);#M2, d2, D2, R2
    T2=permute_neighbour_ind(T2,2,3,4);#M2, D2, d2, R2
    Ut2=unitary(fuse(space(T2,1)*space(T2,2)), space(T2,1)*space(T2,2));
    @tensor T2[:]:=Ut2[-1,1,2]*T2[1,2,-2,-3];#M2_D2, d2, R2

    T3=T3;#M3, d3, R3, D3 
    Ut3=unitary(fuse(space(T3,3)*space(T3,4)), space(T3,3)*space(T3,4));
    @tensor T3[:]:=T3[-1,-2,1,2]*Ut3[-3,1,2];#M3, d3, R3_D3 

    @tensor T2_B[:]:=T2[-1,-2,1]*B[1,-3,-4];     #(M2_D2, d2, R2),  (R2, D1, M3) => (M2_D2, d2, D1, M3)
    T2_B=permute_neighbour_ind(T2_B,2,3,4);#(M2_D2, D1, d2, M3)
    T2_B=permute_neighbour_ind(T2_B,1,2,4);#(D1, M2_D2, d2, M3)
    @tensor T1_T2_B[:]:=T1[-1,-2,1]*T2_B[1,-3,-4,-5];#(M1_R1, d1,  D1), (D1, M2_D2, d2, M3) => (M1_R1, d1, M2_D2, d2, M3)

    @tensor T1_T2_B_T3[:]:=T1_T2_B[-1,-2,-3,-4,1]*T3[1,-5,-6];#(M1_R1, d1, M2_D2, d2, M3), (M3, d3, R3_D3) => (M1_R1, d1, M2_D2, d2, d3, R3_D3)
    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3

    #d2',d3',d1', d2,d3,d1
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,5,6,6);#d2',d3',d1', d2,d1,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,4,5,6);#d2',d3',d1', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,2,3,6);#d2',d1',d3', d1,d2,d3
    op_LD_RD_RU=permute_neighbour_ind(op_LD_RD_RU,1,2,6);#d1',d2',d3', d1,d2,d3
    @tensor T1_T2_B_T3[:]:=T1_T2_B_T3[-1,-2,1,2,3,-6]*op_LD_RD_RU[-3,-4,-5,1,2,3];


    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, d1, M2_D2, d2, d3, R3_D3
    if isa(space(T1,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        U1,S1,V1=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
        U3,S3,V3=tsvd(T1_T2_B_T3,(1,2,3,4,),(5,6,);trunc=truncdim(D_max));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
    elseif isa(space(T1,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        U1,S1,V1=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M1_R1, d1, D1_new) (D1_new, M2_D2, d2, d3, R3_D3
        U1,S1,V1=Truncations(U1,S1,V1,D_max,trun_tol);#println(norm(U1*S1*V1-M_old)/norm(M_old))
        U3,S3,V3=tsvd(T1_T2_B_T3,(1,2,3,4,),(5,6,));#(M1_R1, d1, M2_D2, d2, M3_new) (M3_new, d3, R3_D3)
        U3,S3,V3=Truncations(U3,S3,V3,D_max,trun_tol);#println(norm(U3*S3*V3-M_old)/norm(M_old))
    end


    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M1_R1, M2_D2, d1, d2, d3, R3_D3
    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,1,2,6);# M2_D2, M1_R1, d1, d2, d3, R3_D3
    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,3,4,6);# M2_D2, M1_R1, d2, d1, d3, R3_D3
    T1_T2_B_T3=permute_neighbour_ind(T1_T2_B_T3,2,3,6);# M2_D2, d2, M1_R1, d1, d3, R3_D3
    if isa(space(T1,1), GradedSpace{Z2Irrep, Tuple{Int64, Int64}})
        U2,S2,V2=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,);trunc=truncdim(D_max));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
    elseif isa(space(T1,1), GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}})
        U2,S2,V2=tsvd(T1_T2_B_T3,(1,2,),(3,4,5,6,));#(M2_D2, d2, R2_new) (R2_new, M1_R1, d1, d3, R3_D3)
        U2,S2,V2=Truncations(U2,S2,V2,D_max,trun_tol);#println(norm(U2*S2*V2-M_old)/norm(M_old))
    end

    λ_2_new=permute(S2,(2,),(1,));

    T2_new=U2;#(M2_D2, d2, R2_new)
    T1_B_T3=S2*V2;#(R2_new, M1_R1, d1, d3, R3_D3)
    @tensor T1_B[:]:=T1_B_T3[-1,-2,-3,1,2]*V3'[1,2,-4];#(R2_new, M1_R1, d1, M3_new) 
    T3_new=V3;#(M3_new, d3, R3_D3)
    λ_3_new=S3;

    T1_B=permute_neighbour_ind(T1_B,1,2,4);#(M1_R1, R2_new, d1, M3_new) 
    T1_B=permute_neighbour_ind(T1_B,2,3,4);#(M1_R1, d1, R2_new, M3_new) 
    @tensor B_new[:]:=U1'[-1,1,2]*T1_B[1,2,-2,-3];#(D1_new, R2_new, M3_new) 
    T1_new=U1;#(M1_R1, d1, D1_new) 
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







function triangle_update(B_A, B_B, B_C, B_D, T_A, T_B, T_C, T_D, lambda_A_1, lambda_A_2, lambda_A_3, lambda_B_1, lambda_B_2, lambda_B_3, lambda_C_1, lambda_C_2, lambda_C_3, lambda_D_1, lambda_D_2, lambda_D_3, gates, trun_tol)
    """
    ABABABAB
    CDCDCDCD
    ABABABAB
    CDCDCDCD
    """
    global expon
    function tensor_power(S,expon)
        S=deepcopy(S);
        for cc=1:length(S.data.values)
            mm=S.data.values[cc];
            S.data.values[cc]=mm^expon;
        end
        return S
    end
    

    # B
    #CD
    @tensor T1[:]:=B_B[1,-2,3,-4]*tensor_power(lambda_B_3,expon)[-1,1]*tensor_power(lambda_A_2,expon)[-3,3];
    @tensor T2[:]:=B_C[1,-2,-3,4]*tensor_power(lambda_C_3,expon)[-1,1]*tensor_power(lambda_A_1,expon)[-4,4];
    @tensor T3[:]:=B_D[-1,-2,3,4]*tensor_power(lambda_C_2,expon)[-3,3]*tensor_power(lambda_B_1,expon)[-4,4];
    B=T_D;
    T1, T2, T3, B, lambda_1, lambda_2, lambda_3=update_up_triangle(gates[1], T1, T2, T3, B, trun_tol);
    @tensor B_B[:]:=T1[1,-2,3,-4]*mypinv(tensor_power(lambda_B_3,expon))[-1,1]*mypinv(tensor_power(lambda_A_2,expon))[-3,3];
    @tensor B_C[:]:=T2[1,-2,-3,4]*mypinv(tensor_power(lambda_C_3,expon))[-1,1]*mypinv(tensor_power(lambda_A_1,expon))[-4,4];
    @tensor B_D[:]:=T3[-1,-2,3,4]*mypinv(tensor_power(lambda_C_2,expon))[-3,3]*mypinv(tensor_power(lambda_B_1,expon))[-4,4];
    B_B=permute(B_B,(1,),(2,3,4,));
    B_C=permute(B_C,(1,),(2,3,4,));
    B_D=permute(B_D,(1,),(2,3,4,));
    T_D=B;
    lambda_D_1=lambda_1; 
    lambda_D_2=lambda_2; 
    lambda_D_3=lambda_3; 
    


    # A
    #DC
    @tensor T1[:]:=B_A[1,-2,3,-4]*tensor_power(lambda_A_3,expon)[-1,1]*tensor_power(lambda_B_2,expon)[-3,3];
    @tensor T2[:]:=B_D[1,-2,-3,4]*tensor_power(lambda_D_3,expon)[-1,1]*tensor_power(lambda_B_1,expon)[-4,4];
    @tensor T3[:]:=B_C[-1,-2,3,4]*tensor_power(lambda_D_2,expon)[-3,3]*tensor_power(lambda_A_1,expon)[-4,4];
    B=T_C;
    T1, T2, T3, B, lambda_1, lambda_2, lambda_3=update_up_triangle(gates[2], T1, T2, T3, B, trun_tol);
    @tensor B_A[:]:=T1[1,-2,3,-4]*mypinv(tensor_power(lambda_A_3,expon))[-1,1]*mypinv(tensor_power(lambda_B_2,expon))[-3,3];
    @tensor B_D[:]:=T2[1,-2,-3,4]*mypinv(tensor_power(lambda_D_3,expon))[-1,1]*mypinv(tensor_power(lambda_B_1,expon))[-4,4];
    @tensor B_C[:]:=T3[-1,-2,3,4]*mypinv(tensor_power(lambda_D_2,expon))[-3,3]*mypinv(tensor_power(lambda_A_1,expon))[-4,4];
    B_A=permute(B_A,(1,),(2,3,4,));
    B_D=permute(B_D,(1,),(2,3,4,));
    B_C=permute(B_C,(1,),(2,3,4,));
    T_C=B;
    lambda_C_1=lambda_1; 
    lambda_C_2=lambda_2; 
    lambda_C_3=lambda_3; 

    

    # D
    #AB
    @tensor T1[:]:=B_D[1,-2,3,-4]*tensor_power(lambda_D_3,expon)[-1,1]*tensor_power(lambda_C_2,expon)[-3,3];
    @tensor T2[:]:=B_A[1,-2,-3,4]*tensor_power(lambda_A_3,expon)[-1,1]*tensor_power(lambda_C_1,expon)[-4,4];
    @tensor T3[:]:=B_B[-1,-2,3,4]*tensor_power(lambda_A_2,expon)[-3,3]*tensor_power(lambda_D_1,expon)[-4,4];
    B=T_B;
    T1, T2, T3, B, lambda_1, lambda_2, lambda_3=update_up_triangle(gates[1], T1, T2, T3, B, trun_tol);
    @tensor B_D[:]:=T1[1,-2,3,-4]*mypinv(tensor_power(lambda_D_3,expon))[-1,1]*mypinv(tensor_power(lambda_C_2,expon))[-3,3];
    @tensor B_A[:]:=T2[1,-2,-3,4]*mypinv(tensor_power(lambda_A_3,expon))[-1,1]*mypinv(tensor_power(lambda_C_1,expon))[-4,4];
    @tensor B_B[:]:=T3[-1,-2,3,4]*mypinv(tensor_power(lambda_A_2,expon))[-3,3]*mypinv(tensor_power(lambda_D_1,expon))[-4,4];
    B_D=permute(B_D,(1,),(2,3,4,));
    B_A=permute(B_A,(1,),(2,3,4,));
    B_B=permute(B_B,(1,),(2,3,4,));
    T_B=B;
    lambda_B_1=lambda_1; 
    lambda_B_2=lambda_2; 
    lambda_B_3=lambda_3; 


    # C
    #BA
    @tensor T1[:]:=B_C[1,-2,3,-4]*tensor_power(lambda_C_3,expon)[-1,1]*tensor_power(lambda_D_2,expon)[-3,3];
    @tensor T2[:]:=B_B[1,-2,-3,4]*tensor_power(lambda_B_3,expon)[-1,1]*tensor_power(lambda_D_1,expon)[-4,4];
    @tensor T3[:]:=B_A[-1,-2,3,4]*tensor_power(lambda_B_2,expon)[-3,3]*tensor_power(lambda_C_1,expon)[-4,4];
    B=T_A;
    T1, T2, T3, B, lambda_1, lambda_2, lambda_3=update_up_triangle(gates[2], T1, T2, T3, B, trun_tol);
    @tensor B_C[:]:=T1[1,-2,3,-4]*mypinv(tensor_power(lambda_C_3,expon))[-1,1]*mypinv(tensor_power(lambda_D_2,expon))[-3,3];
    @tensor B_B[:]:=T2[1,-2,-3,4]*mypinv(tensor_power(lambda_B_3,expon))[-1,1]*mypinv(tensor_power(lambda_D_1,expon))[-4,4];
    @tensor B_A[:]:=T3[-1,-2,3,4]*mypinv(tensor_power(lambda_B_2,expon))[-3,3]*mypinv(tensor_power(lambda_C_1,expon))[-4,4];
    B_C=permute(B_C,(1,),(2,3,4,));
    B_B=permute(B_B,(1,),(2,3,4,));
    B_A=permute(B_A,(1,),(2,3,4,));
    T_A=B;
    lambda_A_1=lambda_1; 
    lambda_A_2=lambda_2; 
    lambda_A_3=lambda_3; 
    


    T_A=T_A/norm(T_A);
    T_B=T_B/norm(T_B);
    T_C=T_C/norm(T_C);
    T_D=T_D/norm(T_D);
    B_A=B_A/norm(B_A);
    B_B=B_B/norm(B_B);
    B_C=B_C/norm(B_C);
    B_D=B_D/norm(B_D);
    lambda_A_1=lambda_A_1/norm(lambda_A_1);
    lambda_A_2=lambda_A_2/norm(lambda_A_2);
    lambda_A_3=lambda_A_3/norm(lambda_A_3);
    lambda_B_1=lambda_B_1/norm(lambda_B_1);
    lambda_B_2=lambda_B_2/norm(lambda_B_2);
    lambda_B_3=lambda_B_3/norm(lambda_B_3);
    lambda_C_1=lambda_C_1/norm(lambda_C_1);
    lambda_C_2=lambda_C_2/norm(lambda_C_2);
    lambda_C_3=lambda_C_3/norm(lambda_C_3);
    lambda_D_1=lambda_D_1/norm(lambda_D_1);
    lambda_D_2=lambda_D_2/norm(lambda_D_2);
    lambda_D_3=lambda_D_3/norm(lambda_D_3);

    return B_A, B_B, B_C, B_D, T_A, T_B, T_C, T_D, lambda_A_1, lambda_A_2, lambda_A_3, lambda_B_1, lambda_B_2, lambda_B_3, lambda_C_1, lambda_C_2, lambda_C_3, lambda_D_1, lambda_D_2, lambda_D_3
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



function itebd(parameters, B_A, B_B, B_C, B_D, T_A, T_B, T_C, T_D, lambda_A_1, lambda_A_2, lambda_A_3, lambda_B_1, lambda_B_2, lambda_B_3, lambda_C_1, lambda_C_2, lambda_C_3, lambda_D_1, lambda_D_2, lambda_D_3,  tau, dt, trun_tol)
    # println("one step")
    # println(space(T_u))
    # println(space(T_d))

    gates_ru_ld_rd=gate_RU_LD_RD(parameters,dt, typeof(space(T_A,1)));

    for ct=1:Int(round(tau/abs(dt)))
        println("iteration "*string(ct));flush(stdout)
        B_A, B_B, B_C, B_D, T_A, T_B, T_C, T_D, lambda_A_1, lambda_A_2, lambda_A_3, lambda_B_1, lambda_B_2, lambda_B_3, lambda_C_1, lambda_C_2, lambda_C_3, lambda_D_1, lambda_D_2, lambda_D_3= triangle_update(B_A, B_B, B_C, B_D, T_A, T_B, T_C, T_D, lambda_A_1, lambda_A_2, lambda_A_3, lambda_B_1, lambda_B_2, lambda_B_3, lambda_C_1, lambda_C_2, lambda_C_3, lambda_D_1, lambda_D_2, lambda_D_3, gates_ru_ld_rd, trun_tol);
    end
    return B_A, B_B, B_C, B_D, T_A, T_B, T_C, T_D, lambda_A_1, lambda_A_2, lambda_A_3, lambda_B_1, lambda_B_2, lambda_B_3, lambda_C_1, lambda_C_2, lambda_C_3, lambda_D_1, lambda_D_2, lambda_D_3
end

