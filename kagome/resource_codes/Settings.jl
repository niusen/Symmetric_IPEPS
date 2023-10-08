Base.@kwdef mutable struct CTMRG_settings
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
end


Base.@kwdef mutable struct Optim_settings
    init_statenm :: String ="nothing"
    init_noise :: Float64 =0
    grad_CTM_method :: String ="from_converged_CTM"; # "restart" or "from_converged_CTM"
    linesearch_CTM_method :: String ="from_converged_CTM"; # "restart" or "from_converged_CTM"
end

Base.@kwdef mutable struct Energy_settings
    kagome_method :: String = "E_single_triangle";# "E_single_triangle", "E_triangle", "J2J3", "E_bond"
    E_up_method :: String = "1x1";#"1x1", "2x2"
    cal_chiral_order :: Bool = false;

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


Base.@kwdef mutable struct initial_CTM
    CTM :: CTM_struc 
    init_type :: String = "PBC";
    reconstruct :: Bool =false;



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
        Cset.C1=M;
    elseif direction==2
        Cset.C2=M; 
    elseif direction==3
        Cset.C3=M; 
    elseif direction==4
        Cset.C4=M; 
    end
    return Cset
end

function set_Tset(Tset,M,direction)
    if direction==1 
        Tset.T1=M;
    elseif direction==2
        Tset.T2=M; 
    elseif direction==3
        Tset.T3=M; 
    elseif direction==4
        Tset.T4=M; 
    end
    return Tset
end

