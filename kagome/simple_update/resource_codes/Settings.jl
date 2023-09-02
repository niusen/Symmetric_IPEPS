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
end




