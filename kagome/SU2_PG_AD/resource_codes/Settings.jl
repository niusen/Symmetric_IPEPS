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

Base.@kwdef mutable struct Energy_settings
    kagome_method :: String = "E_single_triangle";# "E_single_triangle", "E_triangle", "J2J3", "E_bond"
    E_up_method :: String = "1x1";#"1x1", "2x2"
    E_dn_method :: String = "simplified";#"open_leg", "simplfied"
    cal_chiral_order :: Bool = false;

end


Base.@kwdef mutable struct Backward_settings
    grad_inverse_tol :: Float64 = 1e-8
    grad_regulation_epsilon :: Float64 = 1e-12
    show_ite_grad_norm :: Bool = false;
end






mutable struct Cset_struc
    C1::TensorMap
    C2::TensorMap
    C3::TensorMap
    C4::TensorMap

end

mutable struct Tset_struc
    T1::TensorMap
    T2::TensorMap
    T3::TensorMap
    T4::TensorMap

end

mutable struct CTM_struc
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