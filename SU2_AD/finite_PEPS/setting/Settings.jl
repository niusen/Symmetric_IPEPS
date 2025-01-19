# Base.@kwdef mutable struct grad_CTMRG_settings
#     CTM_conv_tol :: Float64 = 1e-6
#     CTM_ite_nums :: Int64 =200
#     CTM_trun_tol :: Float64 =1e-8
#     svd_lanczos_tol :: Float64 =1e-8
#     projector_strategy :: String ="4x4";#"4x4" or "4x2"
#     conv_check :: String ="singular_value";
#     CTM_ite_info :: Bool = true
#     CTM_conv_info :: Bool = true
#     CTM_trun_svd :: Bool = false
#     construct_double_layer :: Bool=true
#     grad_checkpoint :: Bool = false #use check point to save memory, i.e., only store CTM tensors in each iteration and disgard larger intermidiate tensors.
# end

# Base.@kwdef mutable struct LS_CTMRG_settings
#     CTM_conv_tol :: Float64 = 1e-6
#     CTM_ite_nums :: Int64 =200
#     CTM_trun_tol :: Float64 =1e-8
#     svd_lanczos_tol :: Float64 =1e-8
#     projector_strategy :: String ="4x4";#"4x4" or "4x2"
#     conv_check :: String ="singular_value";
#     CTM_ite_info :: Bool = true
#     CTM_conv_info :: Bool = true
#     CTM_trun_svd :: Bool = false
#     construct_double_layer :: Bool=true
#     grad_checkpoint :: Bool = false #use check point to save memory, i.e., only store CTM tensors in each iteration and disgard larger intermidiate tensors.
# end


# Base.@kwdef mutable struct Optim_settings
#     init_statenm :: String ="nothing"
#     init_noise :: Float64 =0
#     grad_CTM_method :: String ="restart"; # "restart" or "from_converged_CTM"
#     linesearch_CTM_method :: String ="restart"; # "restart" or "from_converged_CTM"
# end

function Rank(T::TensorMap)
    #number of indices
    return length((domain(T)*codomain(T)).spaces)
end 

Base.@kwdef mutable struct Svd_settings
    svd_trun_method :: String = "chi";#"chi" or "tol"
    chi_max :: Int64 = 500
    tol :: Float64 = 1e-5;
end


Base.@kwdef mutable struct Backward_settings
    grad_inverse_tol :: Float64 = 1e-8
    grad_regulation_epsilon :: Float64 = 1e-12
    show_ite_grad_norm :: Bool = false;
end


Base.@kwdef mutable struct finite_PEPS_with_coe
    Tset :: Matrix{TensorMap} 
    logcoe :: Float64
end

abstract type Contract_History end

struct disk_contract_history <:Contract_History
    config :: Vector;
    mps_top_set::Matrix{TensorMap}
    mps_bot_set::Matrix{TensorMap}
end


struct Final_mps_range #used for contract PBC MPS on torus
    left_range::Matrix{Int}
    right_range::Matrix{Int}
    length::Vector{Int}
end
# struct torus_contract_history <:Contract_History
#     config :: Vector;
#     mps_all_set::Matrix{TensorMap}
#     final_mps_set::Matrix{TensorMap}
#     final_mps_range::Final_mps_range
# end
struct torus_contract_history <:Contract_History
    config :: Vector;
    mps_all_set::Matrix{TensorMap}
end

function coordinate_2d_to_1d(L1::Int,L2::Int,pos::Vector{Int})
    px,py=pos;
    return (px-1)*L1+py
end

function coordinate_1d_to_2d(L1::Int,L2::Int,ind::Int)
    px=div(ind-1,L1)+1;
    py=rem(ind-1,L1)+1;
    # @assert ind==((px)*Lx)
    return [px,py]
end




function convert_to_dense(T::Tensor)
    function convert_to_dense_space(V)
        if V.dual
            return ComplexSpace(dim(V))';
        else
            return ComplexSpace(dim(V));
        end
    end

    if Rank(T)==5;
        T=permute(T,(1,2,3,4,5,));
        V1=convert_to_dense_space(space(T,1));
        V2=convert_to_dense_space(space(T,2));
        V3=convert_to_dense_space(space(T,3));
        V4=convert_to_dense_space(space(T,4));
        V5=convert_to_dense_space(space(T,5));
        T_dense=convert(Array,T);
        T_new=TensorMap(T_dense,V1*V2*V3*V4,V5');
        T_new=permute(T_new,(1,2,3,4,5,))
    end
    return T_new
end


# struct Cset_struc #avoid using mutable, which could induce mistake for AD
#     C1::TensorMap
#     C2::TensorMap
#     C3::TensorMap
#     C4::TensorMap

# end

# struct Tset_struc #avoid using mutable, which could induce mistake for AD
#     T1::TensorMap
#     T2::TensorMap
#     T3::TensorMap
#     T4::TensorMap

# end

# struct CTM_struc #avoid using mutable, which could induce mistake for AD
#     Cset::Cset_struc
#     Tset::Tset_struc

# end


# Base.@kwdef mutable struct initial_condition
    
#     init_type :: String = "PBC";
#     reconstruct_CTM :: Bool =false;
#     reconstruct_AA :: Bool =false


# end






# function get_Cset(Cset,direction)
#     if direction==1
#         return Cset.C1
#     elseif direction==2
#         return Cset.C2
#     elseif direction==3
#         return Cset.C3
#     elseif direction==4
#         return Cset.C4
#     end

# end

# function get_Tset(Tset,direction)
#     if direction==1
#         return Tset.T1
#     elseif direction==2
#         return Tset.T2
#     elseif direction==3
#         return Tset.T3
#     elseif direction==4
#         return Tset.T4
#     end

# end

# function set_Cset(Cset,M,direction)
#     if direction==1 
#         Cset_new=Cset_struc(M,Cset.C2,Cset.C3,Cset.C4);
#     elseif direction==2
#         Cset_new=Cset_struc(Cset.C1,M,Cset.C3,Cset.C4); 
#     elseif direction==3
#         Cset_new=Cset_struc(Cset.C1,Cset.C2,M,Cset.C4); 
#     elseif direction==4
#         Cset_new=Cset_struc(Cset.C1,Cset.C2,Cset.C3,M);
#     end
#     return Cset_new
# end

# function set_Tset(Tset,M,direction)
#     if direction==1 
#         Tset_new=Tset_struc(M,Tset.T2,Tset.T3,Tset.T4);
#     elseif direction==2
#         Tset_new=Tset_struc(Tset.T1,M,Tset.T3,Tset.T4);
#     elseif direction==3
#         Tset_new=Tset_struc(Tset.T1,Tset.T2,M,Tset.T4);
#     elseif direction==4
#         Tset_new=Tset_struc(Tset.T1,Tset.T2,Tset.T3,M);
#     end
#     return Tset_new
# end
# # function set_Tset_tuple(Tset,M,direction)
# #     if direction==1 
# #         Tset_new=(T1=M,T2=Tset.T2,T3=Tset.T3,T4=Tset.T4);
# #     elseif direction==2
# #         Tset_new=(T1=Tset.T1,T2=M,T3=Tset.T3,T4=Tset.T4);
# #     elseif direction==3
# #         Tset_new=(T1=Tset.T1,T2=Tset.T2,T3=M,T4=Tset.T4);
# #     elseif direction==4
# #         Tset_new=(T1=Tset.T1,T2=Tset.T2,T3=Tset.T3,T4=M);
# #     end
# #     return Tset_new
# # end


# struct Elementary_tensors
#     A_set :: Vector{Any}
#     B_set :: Vector{Any}
#     A1_set :: Vector{Any}
#     A2_set :: Vector{Any}
#     A1_has_odd :: Vector{Float64}
#     A2_has_odd :: Vector{Float64}

# end

# Base.@kwdef struct IPESS_IRREP
#     Bond_irrep :: String = "A"; #"A", "B", "A+iB"
#     Triangle_irrep :: String = "A1+iA2"; #"A1", "A2", "A1+iA2"
#     nonchiral :: String = "No"; #"No", "A1_even", "A1_odd"

# end