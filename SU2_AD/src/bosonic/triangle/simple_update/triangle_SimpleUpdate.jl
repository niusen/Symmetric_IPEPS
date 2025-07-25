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





function get_triangles_PBC(Lx,Ly)
    #coordinate of 3 sites in a triangle: 
    #(px+1,py)
    #(px,py+1)
    #(px+1,py+1)
    function distant_triangles(T1::Array,T2::Array)
        #if they share any common index, return false
        for cc=1:length(T1)
            if T1[cc] in T2
                return false
            end
        end
        return true
    end

    triangle_order=reshape(1:Lx*Ly,Lx,Ly);
    list_all_triangles=zeros(Int,Lx*Ly,2);
    for c1=1:Lx
        for c2=1:Ly
            list_all_triangles[triangle_order[c1,c2],1:2]=[c1 c2];
        end
    end

    list_triangle_sites=zeros(Int,Lx*Ly,3);
    for cc=1:Lx*Ly
        coord=list_all_triangles[cc,:];
        list_triangle_sites[cc,1:3]=[triangle_order[mod1(coord[1]+1,Lx),coord[2]] triangle_order[coord[1],mod1(coord[2]+1,Ly)] triangle_order[mod1(coord[1]+1,Lx),mod1(coord[2]+1,Ly)]];
    end
    
    triangle_groups=Matrix{Vector}(undef,2+mod(Lx,2),2+mod(Ly,2));
    for ca=1:size(triangle_groups,1)
        for cb=1:size(triangle_groups,2)
            group=Vector{Int64}(undef,0);
            # push!(group,1);
            triangle_groups[ca,cb]=group;
        end
    end
    
    
    for cc in axes(list_all_triangles,1)
        px,py=list_all_triangles[cc,:];
        if (mod(Lx,2)==1)&&(px==Lx)
            tx=3;
        else
            if mod(px,2)==1
                tx=1;
            elseif mod(px,2)==0
                tx=2;
            end
        end
        if (mod(Ly,2)==1)&&(py==Ly)
            ty=3;
        else
            if mod(py,2)==1
                ty=1;
            elseif mod(py,2)==0
                ty=2;
            end
        end
        group=triangle_groups[tx,ty];
        push!(group,cc);
        triangle_groups[tx,ty]=group;
    end

    #verify in each group triangles has no overlap
    for group in triangle_groups
        for c1 in eachindex(group)
            for c2=c1+1:length(group)
                @assert distant_triangles(list_triangle_sites[group[c1],:],list_triangle_sites[group[c2],:])==true; 
            end
        end
    end

    #verify all triangles are included
    vec=[];
    for cc in triangle_groups
        vec=vcat(vec,cc);
    end
    @assert sum(abs.(sort(vec)[:]-Vector(1:Lx*Ly)))<1e-10;


    triangle_groups_=Vector{Vector{Tuple}}(undef,length(triangle_groups));
    for cc in eachindex(triangle_groups)
        group=triangle_groups[cc];
        group_=Vector{Tuple}(undef,length(group));
        for tt in eachindex(group)
            group_[tt]=(list_all_triangles[group[tt],1],list_all_triangles[group[tt],2],)
        end
        triangle_groups_[cc]=group_;
    end

    return triangle_groups_
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



function gate_triangle(energy_setting,J1,J_ijk,J_kji,dt, space_type)
    #anti-clockwise and satisfy C3 symmetry
    # @assert mod(Lx,energy_setting.Magnetic_cell)==0;
    if space_type==GradedSpace{ProductSector{Tuple{SU2Irrep, SU2Irrep}}, TensorKit.SortedVectorDict{ProductSector{Tuple{SU2Irrep, SU2Irrep}}, Int64}}
        Hamiltonian_terms=Hamiltonian_SU2_SU2;
    end
    P_ij, P_ijk, P_kji = @ignore_derivatives Hamiltonian_terms();
    
   
    Id=unitary(space(P_ij,4)',space(P_ij,4)');
    @tensor hh_12[:]:=P_ij[-1,-2,-4,-5]*Id[-3,-6];
    @tensor hh_13[:]:=P_ij[-1,-3,-4,-6]*Id[-2,-5];
    @tensor hh_23[:]:=P_ij[-2,-3,-5,-6]*Id[-1,-4];
    
    hh_12=permute(hh_12,(1,2,3,),(4,5,6,));
    hh_13=permute(hh_13,(1,2,3,),(4,5,6,));
    hh_23=permute(hh_23,(1,2,3,),(4,5,6,));
    #################
    hh=J1*(hh_12+hh_13+hh_23)+J_ijk*P_ijk+J_kji*P_kji;
    #################

    hh=permute(hh,(1,2,3,),(4,5,6,));
    hh_rotate=permute(hh,(2,3,1,),(5,6,4,));
    @assert norm(hh-hh_rotate)/norm(hh)<1e-10;
    @assert norm(hh-hh')/norm(hh)<1e-10;
    eu,ev=eigh(hh);
    gate=ev*exp(-dt*eu)*ev';

    return gate
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






function gate_RU_LD_RD(parameters,energy_setting, dt, space_type)
    Lx=energy_setting.Lx;
    Ly=energy_setting.Ly;

    @assert energy_setting.model in ("Heisenberg",)
    if space_type==GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}
        #Ident_set, N_occu_set, n_double_set, Cdag_set, C_set=special_Hamiltonians_spinful_SU2();
    elseif space_type==GradedSpace{SU4Irrep, TensorKit.SortedVectorDict{SU4Irrep, Int64}} 
        Sa,Sb, Id_SS,  P_123_a,P_123_b,P_123_c,  P_132_a,P_132_b,P_132_c,   Id_P123_ab, Id_P123_bc,   chirality_123_a,chirality_123_b,chirality_123_c, Id_chirality_ab,Id_chirality_bc=SUN_spin(energy_setting.N);
    end
    
    J=parameters["J"];
    K=parameters["K"];
    Φ=parameters["Φ"];

    @assert K==0;#Heisenberg model

    Id=unitary(space(Sa,3),space(Sa,3));
    @tensor SS[:]:=Sa[1,2,-1,-2]*Sb[2,1,-3,-4];

    @tensor SS12[:]:=SS[-1,-4,-2,-5]*Id[-3,-6];
    @tensor SS13[:]:=SS[-1,-4,-3,-6]*Id[-2,-5];
    @tensor SS23[:]:=SS[-2,-5,-3,-6]*Id[-1,-4];


    gate_set=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Lx;
        for cy=1:Ly
            ####################
            @tensor hh=J*(SS12+SS23+SS13);
            #################
            hh=permute(hh,(1,2,3,),(4,5,6,));#
            @assert norm(hh-hh')/norm(hh)<1e-12;
            @assert norm(hh-permute(hh,(2,3,1,),(5,6,4,)))/norm(hh)<1e-12;
            eu,ev=eigh(hh);
            gate=ev*exp(-dt*eu)*ev';
            gate_set[cx,cy]=gate;
        end
    end
    return gate_set
end