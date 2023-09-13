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


