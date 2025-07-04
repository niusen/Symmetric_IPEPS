using LinearAlgebra: diagind

Base.@kwdef mutable struct grad_CTMRG_settings
    CTM_conv_tol :: Float64 = 1e-6
    CTM_ite_nums :: Int64 =200
    CTM_trun_tol :: Float64 =1e-8
    svd_lanczos_tol :: Float64 =1e-8
    projector_strategy :: String ="4x4";#"4x4" or "4x2"
    conv_check :: String ="singular_value";
    CTM_ite_info :: Bool = true
    CTM_conv_info :: Bool = true
    CTM_trun_svd :: Bool = false
    construct_double_layer :: Bool=true
    grad_checkpoint :: Bool = false #use check point to save memory, i.e., only store CTM tensors in each iteration and disgard larger intermidiate tensors.
end

Base.@kwdef mutable struct LS_CTMRG_settings
    CTM_conv_tol :: Float64 = 1e-6
    CTM_ite_nums :: Int64 =200
    CTM_trun_tol :: Float64 =1e-8
    svd_lanczos_tol :: Float64 =1e-8
    projector_strategy :: String ="4x4";#"4x4" or "4x2"
    conv_check :: String ="singular_value";
    CTM_ite_info :: Bool = true
    CTM_conv_info :: Bool = true
    CTM_trun_svd :: Bool = false
    construct_double_layer :: Bool=true
    grad_checkpoint :: Bool = false #use check point to save memory, i.e., only store CTM tensors in each iteration and disgard larger intermidiate tensors.
end


Base.@kwdef mutable struct Optim_settings
    init_statenm :: String ="nothing"
    init_noise :: Float64 =0
    grad_CTM_method :: String ="restart"; # "restart" or "from_converged_CTM"
    linesearch_CTM_method :: String ="restart"; # "restart" or "from_converged_CTM"
end




Base.@kwdef mutable struct Backward_settings
    grad_inverse_tol :: Float64 = 1e-8
    grad_regulation_epsilon :: Float64 = 1e-12
    show_ite_grad_norm :: Bool = false;
end






struct Cset_struc #avoid using mutable, which could induce mistake for AD
    C1::TensorMap
    C2::TensorMap
    C3::TensorMap
    C4::TensorMap

end

struct Tset_struc #avoid using mutable, which could induce mistake for AD
    T1::TensorMap
    T2::TensorMap
    T3::TensorMap
    T4::TensorMap

end

struct CTM_struc #avoid using mutable, which could induce mistake for AD
    Cset::Cset_struc
    Tset::Tset_struc

end


Base.@kwdef mutable struct initial_condition
    
    init_type :: String = "PBC";
    reconstruct_CTM :: Bool =false;
    reconstruct_AA :: Bool =false


end






function get_Cset(Cset,direction)
    if direction==1
        return Cset.C1
    elseif direction==2
        return Cset.C2
    elseif direction==3
        return Cset.C3
    elseif direction==4
        return Cset.C4
    end

end

function get_Tset(Tset,direction)
    if direction==1
        return Tset.T1
    elseif direction==2
        return Tset.T2
    elseif direction==3
        return Tset.T3
    elseif direction==4
        return Tset.T4
    end

end

function set_Cset(Cset,M,direction)
    if direction==1 
        Cset_new=Cset_struc(M,Cset.C2,Cset.C3,Cset.C4);
    elseif direction==2
        Cset_new=Cset_struc(Cset.C1,M,Cset.C3,Cset.C4); 
    elseif direction==3
        Cset_new=Cset_struc(Cset.C1,Cset.C2,M,Cset.C4); 
    elseif direction==4
        Cset_new=Cset_struc(Cset.C1,Cset.C2,Cset.C3,M);
    end
    return Cset_new
end

function set_Tset(Tset,M,direction)
    if direction==1 
        Tset_new=Tset_struc(M,Tset.T2,Tset.T3,Tset.T4);
    elseif direction==2
        Tset_new=Tset_struc(Tset.T1,M,Tset.T3,Tset.T4);
    elseif direction==3
        Tset_new=Tset_struc(Tset.T1,Tset.T2,M,Tset.T4);
    elseif direction==4
        Tset_new=Tset_struc(Tset.T1,Tset.T2,Tset.T3,M);
    end
    return Tset_new
end
# function set_Tset_tuple(Tset,M,direction)
#     if direction==1 
#         Tset_new=(T1=M,T2=Tset.T2,T3=Tset.T3,T4=Tset.T4);
#     elseif direction==2
#         Tset_new=(T1=Tset.T1,T2=M,T3=Tset.T3,T4=Tset.T4);
#     elseif direction==3
#         Tset_new=(T1=Tset.T1,T2=Tset.T2,T3=M,T4=Tset.T4);
#     elseif direction==4
#         Tset_new=(T1=Tset.T1,T2=Tset.T2,T3=Tset.T3,T4=M);
#     end
#     return Tset_new
# end


struct Elementary_tensors
    A_set :: Vector{Any}
    B_set :: Vector{Any}
    A1_set :: Vector{Any}
    A2_set :: Vector{Any}
    A1_has_odd :: Vector{Float64}
    A2_has_odd :: Vector{Float64}

end

Base.@kwdef struct IPESS_IRREP
    Bond_irrep :: String = "A"; #"A", "B", "A+iB"
    Triangle_irrep :: String = "A1+iA2"; #"A1", "A2", "A1+iA2"
    nonchiral :: String = "No"; #"No", "A1_even", "A1_odd"

end




function remove_small_elements(T::TensorMap,tol=1e-20)
    #delete super small elements, otherwise svd could send error
    Norm=norm(T);
    # for cb in eachindex(T.data.values)
    #     mm=T.data.values[cb];
    #     mm1=deepcopy(mm);
    #     for ci in eachindex(mm1)
    #         if abs(mm1[ci])/Norm<tol;
    #             mm1[ci]=0;
    #         end
    #     end 
    #     T.data.values[cb]=mm1;
    # end
    mm=T.data;
    mm1=deepcopy(mm);
    for ele in eachindex(mm1)
        if (abs(mm1[ele])/Norm)<tol;
            mm1[ele]=0;
        end
    end
    T.data.=mm1;
    return T
end


function TensorMap_to_DiagonalTensorMap(t::TensorMap) 
    isa(t, DiagonalTensorMap) && return t
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("DiagonalTensorMap requires equal domain and codomain"))
    # A = storagetype(t)
    t_dense=convert(Array,t);
    @assert norm(diagm(diag(t_dense))-t_dense)/norm(t_dense)<1e-15
    d = DiagonalTensorMap(ones(sum(space(t, 1).dims.values)), space(t, 1))
    for (c, b) in blocks(d)
        bt = block(t, c)
        # TODO: rewrite in terms of `diagview` from MatrixAlgebraKit.jl
        copy!(b.diag, view(bt, diagind(bt)))
    end
    return d
end





function distribute_workers(N_terms,ntask)
    if N_terms<ntask
        group_ind=Matrix{Int}(undef,N_terms,2);
        for cc=1:N_terms
            group_ind[cc,1]=cc;
            group_ind[cc,2]=cc;
        end
        return group_ind
    else

        length_set=Int(floor(N_terms/ntask))*ones(Int64,ntask);
        extra=N_terms-sum(length_set);
        for cc=1:extra
            length_set[cc]=length_set[cc]+1
        end
        @assert sum(length_set)==N_terms;

        group_ind=Matrix{Int64}(undef,ntask,2);
        total=0;
        for cc=1:ntask
            group_ind[cc,:]=[sum(length_set[1:cc-1])+1,sum(length_set[1:cc])];
        end
        #verify
        total=Vector{Int}(undef,0);
        for cc=1:size(group_ind,1)
            @assert group_ind[cc,1]<=group_ind[cc,2]
            total=vcat(total,Vector(group_ind[cc,1]:group_ind[cc,2]));
        end
        @assert sort(total)==sort(1:N_terms);
        return group_ind
    end
end