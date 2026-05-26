using TensorKit
using LinearAlgebra: diag, I, diagm, norm
using JLD2, ChainRulesCore
using KrylovKit
using JSON
using Random
using Dates

include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_RVB_ansatz.jl"))

function _rvb_get_global(name::Symbol, value)
    if value !== nothing
        return value
    elseif isdefined(Main, name)
        return getfield(Main, name)
    else
        error("Please pass keyword `$name=...`, or define global `$name` before calling this function.")
    end
end

function _rvb_get_ctm_setting(ctm_setting)
    if ctm_setting !== nothing
        return ctm_setting
    elseif isdefined(Main, :grad_ctm_setting)
        return getfield(Main, :grad_ctm_setting)
    elseif isdefined(Main, :LS_ctm_setting)
        return getfield(Main, :LS_ctm_setting)
    else
        error("Please pass `ctm_setting=...`, or define global `grad_ctm_setting`/`LS_ctm_setting`.")
    end
end

function _rvb_quiet_ctm_setting(ctm_setting)
    setting = deepcopy(ctm_setting)
    fields = fieldnames(typeof(setting))
    if :CTM_ite_info in fields
        setfield!(setting, :CTM_ite_info, false)
    end
    if :CTM_conv_info in fields
        setfield!(setting, :CTM_conv_info, false)
    end
    return setting
end

function _rvb_ctmrg(A, chi, init, init_CTM, ctm_setting)
    out = CTMRG(A, chi, init, init_CTM, ctm_setting)
    if length(out) == 8
        return out
    elseif length(out) == 6
        CTM, AA, U_L, U_D, U_R, U_U = out
        return CTM, AA, U_L, U_D, U_R, U_U, missing, missing
    else
        error("Unexpected CTMRG return length: " * string(length(out)))
    end
end

function full_RVB_coefficients(p::AbstractVector)
    if length(p) == 2
        return [1.0, p[1], p[2]]
    elseif length(p) == 3
        @assert abs(p[1] - 1) < 1e-12 "For this optimizer the first RVB coefficient is fixed to 1."
        return Float64.(collect(p))
    else
        error("RVB coefficient variables must be [c2, c3], with c1 fixed to 1.")
    end
end

function RVB_tensor_from_coefficients(
    p::AbstractVector,
    basis;
    chiral_phase=im,
)
    q = full_RVB_coefficients(p)
    A1a, A1b, A2 = basis
    return q[1] * A1a + q[2] * A1b + (chiral_phase * q[3]) * A2
end

function RVB_tensor_from_coefficients(
    p::AbstractVector;
    seed::Integer=1234,
    chiral_phase=im,
)
    basis = D2_point_group_symmetric_tensors(; seed=seed)
    return RVB_tensor_from_coefficients(
        p,
        basis;
        chiral_phase=chiral_phase,
    )
end

function RVB_variables_from_tensor(A::TensorMap, basis; chiral_phase=im)
    A1a, A1b, A2 = basis
    A = A / norm(A)

    c1 = dot(A1a, A)
    @assert abs(c1) > 1e-12 "Cannot fix c1=1 because loaded tensor has nearly zero A1a component."

    c2 = dot(A1b, A) / c1
    c3 = dot(chiral_phase * A2, A) / c1

    @assert abs(imag(c2)) < 1e-8 "Loaded tensor gives complex c2=$(c2); this optimizer expects real coefficients."
    @assert abs(imag(c3)) < 1e-8 "Loaded tensor gives complex c3=$(c3); this optimizer expects real coefficients."
    return [real(c2), real(c3)]
end

function initial_RVB_variables(init_statenm, basis; default=[1.0, 1.0], chiral_phase=im)
    if init_statenm in (nothing, "nothing", "")
        return Float64.(collect(default))
    end

    data = load(init_statenm)
    if haskey(data, "rvb_variables")
        variables = data["rvb_variables"]
        @assert length(variables) == 2 "Saved `rvb_variables` must be [c2, c3]."
        return Float64.(collect(variables))
    elseif haskey(data, "variables")
        variables = data["variables"]
        @assert length(variables) == 2 "Saved `variables` must be [c2, c3]."
        return Float64.(collect(variables))
    elseif haskey(data, "rvb_coefficients")
        coefficients = data["rvb_coefficients"]
        @assert length(coefficients) == 3 "Saved `rvb_coefficients` must be [c1, c2, c3]."
        @assert abs(coefficients[1]) > 1e-12 "Cannot fix c1=1 because saved coefficient c1 is nearly zero."
        return Float64.([real(coefficients[2] / coefficients[1]), real(coefficients[3] / coefficients[1])])
    elseif haskey(data, "coefficients")
        coefficients = data["coefficients"]
        @assert length(coefficients) == 3 "Saved `coefficients` must be [c1, c2, c3]."
        @assert abs(coefficients[1]) > 1e-12 "Cannot fix c1=1 because saved coefficient c1 is nearly zero."
        return Float64.([real(coefficients[2] / coefficients[1]), real(coefficients[3] / coefficients[1])])
    else
        error("Initial state file must contain saved RVB parameters: `rvb_variables`, `variables`, `rvb_coefficients`, or `coefficients`.")
    end
end

function RVB_coefficients_cost(
    p::AbstractVector,
    basis;
    chi=nothing,
    parameters=nothing,
    ctm_setting=nothing,
    energy_setting=nothing,
    chiral_phase=im,
    verbose::Bool=false,
)
    chi_value = _rvb_get_global(:chi, chi)
    parameters_value = _rvb_get_global(:parameters, parameters)
    ctm_setting_value = _rvb_get_ctm_setting(ctm_setting)
    if !verbose
        ctm_setting_value = _rvb_quiet_ctm_setting(ctm_setting_value)
    end
    energy_setting_value = _rvb_get_global(:energy_setting, energy_setting)

    @assert energy_setting_value.model == "triangle_J1_J2_Jchi" "Only triangle_J1_J2_Jchi is implemented here."

    A = RVB_tensor_from_coefficients(
        p,
        basis;
        chiral_phase=chiral_phase,
    )
    A = A / norm(A)

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err = _rvb_ctmrg(A, chi_value, init, [], ctm_setting_value)

    H_Heisenberg, H123chiral, H12, H31, H23 = Hamiltonians(space(A, 1))
    J1 = parameters_value["J1"]
    J2 = parameters_value["J2"]
    Jchi = parameters_value["Jchi"]
    H_triangle = J1 / 4 * (H12 + H31) + J2 / 2 * H23 + Jchi * H123chiral

    E_T1, E_T2, E_T3, E_T4 = evaluate_triangle(
        H_triangle,
        A,
        AA,
        U_L,
        U_D,
        U_R,
        U_U,
        CTM,
        ctm_setting_value,
    )
    E = real(E_T1 + E_T2 + E_T3 + E_T4)
    if verbose
        println("E= " * string(E) * ", coefficients= " * string(full_RVB_coefficients(p)) *
                ", E_triangle= " * string(real.([E_T1, E_T2, E_T3, E_T4])) *
                ", ctm_ite_num= " * string(ite_num) *
                ", ctm_ite_err= " * string(ite_err))
        flush(stdout)
    end
    return E
end

function _rvb_triangle_expectations(rho, U_s_s, H12, H31, H23, H123chiral)
    @tensor rho[:]:=rho[1,2,3]*U_s_s[-1,-4,1]*U_s_s[-2,-5,2]*U_s_s[-3,-6,3]
    norm_rho = @tensor rho[1,2,3,1,2,3]
    e12 = @tensor rho[1,2,3,4,5,6]*H12[1,2,3,4,5,6]
    e31 = @tensor rho[1,2,3,4,5,6]*H31[1,2,3,4,5,6]
    e23 = @tensor rho[1,2,3,4,5,6]*H23[1,2,3,4,5,6]
    echi = @tensor rho[1,2,3,4,5,6]*H123chiral[1,2,3,4,5,6]
    return e12 / norm_rho, e31 / norm_rho, e23 / norm_rho, echi / norm_rho
end

function RVB_coefficients_observables(
    p::AbstractVector,
    basis;
    chi=nothing,
    parameters=nothing,
    ctm_setting=nothing,
    chiral_phase=im,
)
    chi_value = _rvb_get_global(:chi, chi)
    parameters_value = _rvb_get_global(:parameters, parameters)
    ctm_setting_value = _rvb_quiet_ctm_setting(_rvb_get_ctm_setting(ctm_setting))

    A = RVB_tensor_from_coefficients(
        p,
        basis;
        chiral_phase=chiral_phase,
    )
    A = A / norm(A)

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    CTM, AA, U_L, U_D, U_R, U_U, ite_num, ite_err = _rvb_ctmrg(A, chi_value, init, [], ctm_setting_value)

    AA_open, U_s_s = build_double_layer_open(A)
    H_Heisenberg, H123chiral, H12, H31, H23 = Hamiltonians(space(A, 1))

    rho_LU_RU_LD = ob_LU_RU_LD(CTM, AA, AA_open, AA_open, AA_open)
    t1 = _rvb_triangle_expectations(rho_LU_RU_LD, U_s_s, H12, H31, H23, H123chiral)

    rho_LD_RU_RD = ob_LD_RU_RD(CTM, AA, AA_open, AA_open, AA_open)
    rho_LD_RU_RD = permute(rho_LD_RU_RD, (3,1,2,))
    t2 = _rvb_triangle_expectations(rho_LD_RU_RD, U_s_s, H12, H31, H23, H123chiral)

    rho_LU_LD_RD = ob_LU_LD_RD(CTM, AA, AA_open, AA_open, AA_open)
    rho_LU_LD_RD = permute(rho_LU_LD_RD, (2,1,3,))
    t3 = _rvb_triangle_expectations(rho_LU_LD_RD, U_s_s, H12, H31, H23, H123chiral)

    rho_LU_RU_RD = ob_LU_RU_RD(CTM, AA, AA_open, AA_open, AA_open)
    rho_LU_RU_RD = permute(rho_LU_RU_RD, (2,3,1,))
    t4 = _rvb_triangle_expectations(rho_LU_RU_RD, U_s_s, H12, H31, H23, H123chiral)

    e12_terms = real.([t1[1], t2[1], t3[1], t4[1]])
    e31_terms = real.([t1[2], t2[2], t3[2], t4[2]])
    e23_terms = real.([t1[3], t2[3], t3[3], t4[3]])
    chiral_terms = real.([t1[4], t2[4], t3[4], t4[4]])

    nn = sum(e12_terms .+ e31_terms) / 4
    nnn = sum(e23_terms) / 2
    chirality = sum(chiral_terms)

    J1 = parameters_value["J1"]
    J2 = parameters_value["J2"]
    Jchi = parameters_value["Jchi"]

    weighted_nn = J1 * nn
    weighted_nnn = J2 * nnn
    weighted_chirality = Jchi * chirality
    weighted_total = weighted_nn + weighted_nnn + weighted_chirality

    println("  observables:")
    println("    <NN Heisenberg>      = " * string(nn) *
            ", triangle H12=" * string(e12_terms) *
            ", triangle H31=" * string(e31_terms) *
            ", J1 contribution=" * string(weighted_nn))
    println("    <NNN Heisenberg>     = " * string(nnn) *
            ", triangle H23=" * string(e23_terms) *
            ", J2 contribution=" * string(weighted_nnn))
    println("    <scalar chirality>   = " * string(chirality) *
            ", triangles=" * string(chiral_terms) *
            ", Jchi contribution=" * string(weighted_chirality))
    println("    term contribution sum= " * string(weighted_total) *
            ", ctm_ite_num=" * string(ite_num) *
            ", ctm_ite_err=" * string(ite_err))
    flush(stdout)

    return (
        nn=nn,
        e12_terms=e12_terms,
        e31_terms=e31_terms,
        nnn=nnn,
        e23_terms=e23_terms,
        chirality=chirality,
        chiral_terms=chiral_terms,
        weighted_nn=weighted_nn,
        weighted_nnn=weighted_nnn,
        weighted_chirality=weighted_chirality,
        weighted_total=weighted_total,
    )
end

function RVB_coefficients_value_gradient(
    p::AbstractVector,
    basis;
    fd_step::Real=1e-4,
    chi=nothing,
    parameters=nothing,
    ctm_setting=nothing,
    energy_setting=nothing,
    chiral_phase=im,
)
    E0 = RVB_coefficients_cost(
        p,
        basis;
        chi=chi,
        parameters=parameters,
        ctm_setting=ctm_setting,
        energy_setting=energy_setting,
        chiral_phase=chiral_phase,
        verbose=false,
    )
    grad = zeros(Float64, length(p))

    for cc in eachindex(p)
        dp = zeros(Float64, length(p))
        dp[cc] = fd_step * max(1.0, abs(p[cc]))

        E_plus = RVB_coefficients_cost(
            p .+ dp,
            basis;
            chi=chi,
            parameters=parameters,
            ctm_setting=ctm_setting,
            energy_setting=energy_setting,
            chiral_phase=chiral_phase,
            verbose=false,
        )
        grad[cc] = (E_plus - E0) / dp[cc]
    end

    return E0, grad
end

function optimize_RVB_coefficients(
    p0::AbstractVector;
    seed::Integer=1234,
    init_statenm=nothing,
    chi=nothing,
    parameters=nothing,
    ctm_setting=nothing,
    energy_setting=nothing,
    chiral_phase=im,
    maxiter::Integer=50,
    grad_tol::Real=1e-8,
    fd_step::Real=1e-4,
    step0::Real=1.0,
    backtrack::Real=0.5,
    armijo::Real=1e-4,
    min_step::Real=1e-8,
    save_filenm="Optim_RVB_coefficients.jld2",
    verbose_energy::Bool=false,
    print_observables::Bool=true,
)
    @assert length(p0) == 2 "Initial variables must be [c2, c3]; c1 is fixed to 1."
    @assert 0 < backtrack < 1 "backtrack must be between 0 and 1."

    basis = D2_point_group_symmetric_tensors(; seed=seed)
    p = initial_RVB_variables(init_statenm, basis; default=p0, chiral_phase=chiral_phase)

    history_E = Float64[]
    history_p = Vector{Vector{Float64}}()
    history_observables = Any[]

    E, grad = RVB_coefficients_value_gradient(
        p,
        basis;
        chi=chi,
        parameters=parameters,
        ctm_setting=ctm_setting,
        energy_setting=energy_setting,
        chiral_phase=chiral_phase,
        fd_step=fd_step,
    )

    best_E = E
    best_p = copy(p)
    push!(history_E, E)
    push!(history_p, copy(p))

    for iter = 1:maxiter
        grad_norm = norm(grad)
        println("RVB coefficient iteration " * string(iter) * " start" *
                ", E= " * string(E) *
                ", variables= " * string(p) *
                ", coefficients= " * string(full_RVB_coefficients(p)) *
                ", grad= " * string(grad) *
                ", grad_norm= " * string(grad_norm))
        flush(stdout)

        if grad_norm < grad_tol
            break
        end

        direction = -grad
        alpha = step0
        accepted = false
        trial_p = copy(p)
        trial_E = E

        while alpha >= min_step
            trial_p = p .+ alpha .* direction
            trial_E = RVB_coefficients_cost(
                trial_p,
                basis;
                chi=chi,
                parameters=parameters,
                ctm_setting=ctm_setting,
                energy_setting=energy_setting,
                chiral_phase=chiral_phase,
                verbose=false,
            )

            if trial_E <= E - armijo * alpha * grad_norm^2 || trial_E < E
                accepted = true
                break
            end
            alpha *= backtrack
        end

        if !accepted
            println("Line search stopped: no acceptable step above min_step=" * string(min_step))
            flush(stdout)
            break
        end

        p = trial_p
        E = trial_E
        E, grad = RVB_coefficients_value_gradient(
            p,
            basis;
            chi=chi,
            parameters=parameters,
            ctm_setting=ctm_setting,
            energy_setting=energy_setting,
            chiral_phase=chiral_phase,
            fd_step=fd_step,
        )

        push!(history_E, E)
        push!(history_p, copy(p))

        println("RVB coefficient iteration " * string(iter) * " accepted" *
                ", E= " * string(E) *
                ", variables= " * string(p) *
                ", coefficients= " * string(full_RVB_coefficients(p)))
        flush(stdout)

        if print_observables
            obs = RVB_coefficients_observables(
                p,
                basis;
                chi=chi,
                parameters=parameters,
                ctm_setting=ctm_setting,
                chiral_phase=chiral_phase,
            )
            push!(history_observables, obs)
        end

        if verbose_energy
            RVB_coefficients_cost(
                p,
                basis;
                chi=chi,
                parameters=parameters,
                ctm_setting=ctm_setting,
                energy_setting=energy_setting,
                chiral_phase=chiral_phase,
                verbose=true,
            )
        end

        if E < best_E
            best_E = E
            best_p = copy(p)
            if save_filenm !== nothing
                best_A = RVB_tensor_from_coefficients(
                    best_p,
                    basis;
                    chiral_phase=chiral_phase,
                )
                jldsave(
                    save_filenm;
                    rvb_variables=best_p,
                    rvb_coefficients=full_RVB_coefficients(best_p),
                    variables=best_p,
                    coefficients=full_RVB_coefficients(best_p),
                    energy=best_E,
                    A=best_A,
                )
                println("Saved best RVB coefficients to " * string(save_filenm))
                flush(stdout)
            end
        end
    end

    best_A = RVB_tensor_from_coefficients(
        best_p,
        basis;
        chiral_phase=chiral_phase,
    )

    return (
        energy=best_E,
        variables=best_p,
        coefficients=full_RVB_coefficients(best_p),
        tensor=best_A,
        state=Square_iPEPS(best_A),
        history_E=history_E,
        history_p=history_p,
        history_observables=history_observables,
    )
end
