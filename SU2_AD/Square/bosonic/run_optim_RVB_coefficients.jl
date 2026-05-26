using Random

cd(@__DIR__)

include("optim_RVB_coefficients.jl")



D = 3
chi = 54

J1 = 2 * cos(0.06 * pi) * cos(0.14 * pi)
J2 = 2 * cos(0.06 * pi) * sin(0.14 * pi)
Jchi = 2 * sin(0.06 * pi) * 2
parameters = Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)])

grad_ctm_setting = grad_CTMRG_settings()
grad_ctm_setting.CTM_conv_tol = 1e-6
grad_ctm_setting.CTM_ite_nums = 50
grad_ctm_setting.CTM_trun_tol = 1e-8
grad_ctm_setting.svd_lanczos_tol = 1e-8
grad_ctm_setting.projector_strategy = "4x4"
grad_ctm_setting.conv_check = "singular_value"
grad_ctm_setting.CTM_ite_info = true
grad_ctm_setting.CTM_conv_info = true
grad_ctm_setting.CTM_trun_svd = false
grad_ctm_setting.construct_double_layer = true
grad_ctm_setting.grad_checkpoint = true
dump(grad_ctm_setting)

energy_setting = Square_Energy_settings()
energy_setting.model = "triangle_J1_J2_Jchi"
dump(energy_setting)

global chi, parameters, energy_setting, grad_ctm_setting
global multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = grad_ctm_setting.CTM_trun_tol

Random.seed!(555);
init_statenm = "nothing"
p0 = [1.9450618725500934, -0.5911130954483106]

result = optimize_RVB_coefficients(
    p0;
    seed=1234,
    init_statenm=init_statenm,
    chi=chi,
    parameters=parameters,
    ctm_setting=grad_ctm_setting,
    energy_setting=energy_setting,
    chiral_phase=im,
    maxiter=50,
    grad_tol=1e-8,
    fd_step=1e-4,
    step0=1.0,
    save_filenm="Optim_RVB_coefficients_D$(D)_chi_$(chi).jld2",
)

println("Best energy = " * string(result.energy))
println("Best coefficients = " * string(result.coefficients))
