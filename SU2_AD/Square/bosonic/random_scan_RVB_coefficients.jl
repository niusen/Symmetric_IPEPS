using Random
using JLD2

cd(@__DIR__)

include("optim_RVB_coefficients.jl")

Random.seed!(2024)

D = 3
chi = 54

J1 = 2 * cos(0.06 * pi) * cos(0.14 * pi)
J2 = 2 * cos(0.06 * pi) * sin(0.14 * pi)
Jchi = 2 * sin(0.06 * pi) * 2
parameters = Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)])

ctm_setting = grad_CTMRG_settings()
ctm_setting.CTM_conv_tol = 1e-6
ctm_setting.CTM_ite_nums = 20
ctm_setting.CTM_trun_tol = 1e-8
ctm_setting.svd_lanczos_tol = 1e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = false
ctm_setting.CTM_conv_info = false
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true
ctm_setting.grad_checkpoint = true

energy_setting = Square_Energy_settings()
energy_setting.model = "triangle_J1_J2_Jchi"

global chi, parameters, energy_setting
global multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = ctm_setting.CTM_trun_tol

sample_num = 100
coefficient_range = 2.0
seed = 1234
save_filenm = "Random_scan_RVB_coefficients_D$(D)_chi_$(chi).jld2"

basis = D2_point_group_symmetric_tensors(; seed=seed)
rng = MersenneTwister(2024)

records = Vector{NamedTuple}(undef, 0)
best_E = Inf
best_p = nothing
best_obs = nothing

for sample_id = 1:sample_num
    p = coefficient_range .* (2 .* rand(rng, 2) .- 1)

    E = RVB_coefficients_cost(
        p,
        basis;
        chi=chi,
        parameters=parameters,
        ctm_setting=ctm_setting,
        energy_setting=energy_setting,
        chiral_phase=im,
        verbose=false,
    )

    obs = RVB_coefficients_observables(
        p,
        basis;
        chi=chi,
        parameters=parameters,
        ctm_setting=ctm_setting,
        chiral_phase=im,
    )

    record = (
        sample_id=sample_id,
        variables=copy(p),
        coefficients=full_RVB_coefficients(p),
        energy=E,
        chirality=obs.chirality,
        weighted_chirality=obs.weighted_chirality,
        nn=obs.nn,
        nnn=obs.nnn,
        weighted_nn=obs.weighted_nn,
        weighted_nnn=obs.weighted_nnn,
        weighted_total=obs.weighted_total,
    )
    push!(records, record)

    println("sample " * string(sample_id) *
            ", E=" * string(E) *
            ", variables=" * string(p) *
            ", coefficients=" * string(full_RVB_coefficients(p)) *
            ", chirality=" * string(obs.chirality) *
            ", Jchi contribution=" * string(obs.weighted_chirality))
    flush(stdout)

    if E < best_E
        best_E = E
        best_p = copy(p)
        best_obs = obs

        best_A = RVB_tensor_from_coefficients(
            best_p,
            basis;
            chiral_phase=im,
        )
        jldsave(
            save_filenm;
            rvb_variables=best_p,
            rvb_coefficients=full_RVB_coefficients(best_p),
            variables=best_p,
            coefficients=full_RVB_coefficients(best_p),
            energy=best_E,
            chirality=best_obs.chirality,
            weighted_chirality=best_obs.weighted_chirality,
            A=best_A,
            records=records,
        )

        println("  new best: E=" * string(best_E) *
                ", variables=" * string(best_p) *
                ", coefficients=" * string(full_RVB_coefficients(best_p)) *
                ", chirality=" * string(best_obs.chirality) *
                ", saved to " * save_filenm)
        flush(stdout)
    end
end

println("Random scan finished.")
println("Best energy = " * string(best_E))
println("Best variables = " * string(best_p))
println("Best coefficients = " * string(full_RVB_coefficients(best_p)))
println("Best chirality = " * string(best_obs.chirality))
println("Best Jchi contribution = " * string(best_obs.weighted_chirality))
