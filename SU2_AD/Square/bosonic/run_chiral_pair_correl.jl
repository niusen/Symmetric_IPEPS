using Revise, TensorKit, Zygote
using LinearAlgebra: I, norm
using JLD2, ChainRulesCore
using MAT
using KrylovKit
using Random
using Zygote: @ignore_derivatives

cd(@__DIR__)

include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_spin_operator.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "iPEPS_ansatz.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "AD_lib.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "CTMRG.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "Settings.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_chiral_pair_model.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "bosonic", "square", "square_chiral_pair_correl.jl"))

Random.seed!(555)


###########################
import LinearAlgebra.BLAS as BLAS
n_cpu=10;
BLAS.set_num_threads(n_cpu);
println("number of cpus: "*string(BLAS.get_num_threads()))
Base.Sys.set_process_title("C"*string(n_cpu)*"_ob")
pid=getpid();
println("pid="*string(pid));
@show num_logical_cores = Sys.CPU_THREADS
@show hostnm=gethostname()
###########################

chi_set = [40,80,120,160]
distance = 40

init_statenm = "OptimKit_SU2_chiral_pair_D9_chi_54.jld2"

J1 = 2 * cos(0.06 * pi) * cos(0.14 * pi)
J2 = 2 * cos(0.06 * pi) * sin(0.14 * pi)
Jchi = 2 * sin(0.06 * pi) * 2
parameters = Dict([("J1", J1), ("J2", J2), ("Jchi", Jchi)])

ctm_setting = LS_CTMRG_settings()
ctm_setting.CTM_conv_tol = 1e-6
ctm_setting.CTM_ite_nums = 50
ctm_setting.CTM_trun_tol = 1e-12
ctm_setting.svd_lanczos_tol = 1e-8
ctm_setting.projector_strategy = "4x4"
ctm_setting.conv_check = "singular_value"
ctm_setting.CTM_ite_info = false
ctm_setting.CTM_conv_info = true
ctm_setting.CTM_trun_svd = false
ctm_setting.construct_double_layer = true
ctm_setting.grad_checkpoint = true
dump(ctm_setting)

global multiplet_tol, projector_trun_tol
multiplet_tol = 1e-5
projector_trun_tol = ctm_setting.CTM_trun_tol

data = load(init_statenm)
@assert haskey(data, "A") "State file must contain tensor `A`."
state = Square_iPEPS(data["A"])
D = dim(space(state.T, 1))
Vv = space(state.T, 1)
Vp_pair = chiral_pair_physical_space()
@assert space(state.T, 5) == Vp_pair' "Loaded physical space is not the chiral-pair physical leg."

println("correlation initial state virtual spaces: " *
        string((space(state.T, 1), space(state.T, 2), space(state.T, 3), space(state.T, 4))))
println("correlation initial state physical space: " * string(space(state.T, 5)))
flush(stdout)

for chi_value in chi_set
    global chi
    chi = chi_value

    println("\nCompute chiral-pair correlations with chi = " * string(chi))
    flush(stdout)

    init = initial_condition(init_type="PBC", reconstruct_CTM=true, reconstruct_AA=true)
    result = chiral_pair_spinspin_correl_x(
        state,
        chi,
        ctm_setting;
        distance=distance,
        parameters=parameters,
        init=init,
        init_CTM=[],
    )

    println("chiral-pair spin-spin correlations along x:")
    println("  distance = " * string(result.distance))
    println("  <S_A(0).S_A(r)> = " * string(real.(result.spin_AA)))
    println("  <S_B(0).S_B(r)> = " * string(real.(result.spin_BB)))
    println("  <S_A(0).S_B(r)> = " * string(real.(result.spin_AB)))
    println("  <S_B(0).S_A(r)> = " * string(real.(result.spin_BA)))
    println("  E = " * string(result.energy))
    print_chiral_pair_observables(result.observables)
    println("  ctm_ite_num = " * string(result.ctm_ite_num))
    println("  ctm_ite_err = " * string(result.ctm_ite_err))
    flush(stdout)

    mat_filenm = "correl_chiral_pair_D$(D)_chi_$(chi).mat"
    matwrite(
        mat_filenm,
        Dict(
            "distance" => result.distance,
            "norm" => result.norm,
            "spin_AA" => result.spin_AA,
            "spin_BB" => result.spin_BB,
            "spin_AB" => result.spin_AB,
            "spin_BA" => result.spin_BA,
            "E" => result.energy,
            "E_triangles" => result.E_triangles,
            "J1" => J1,
            "J2" => J2,
            "Jchi" => Jchi,
            "NN_A" => result.observables.nn_A,
            "NN_B" => result.observables.nn_B,
            "NNN_A" => result.observables.nnn_A,
            "NNN_B" => result.observables.nnn_B,
            "chirality_A" => result.observables.chirality_A,
            "chirality_B" => result.observables.chirality_B,
            "J1_contribution" => result.observables.weighted_nn,
            "J2_contribution" => result.observables.weighted_nnn,
            "Jchi_contribution" => result.observables.weighted_chirality,
            "term_contribution_sum" => result.observables.weighted_nn +
                result.observables.weighted_nnn +
                result.observables.weighted_chirality,
            "e12_A" => result.observables.e12_A,
            "e31_A" => result.observables.e31_A,
            "e23_A" => result.observables.e23_A,
            "chi_A" => result.observables.chi_A,
            "e12_B" => result.observables.e12_B,
            "e31_B" => result.observables.e31_B,
            "e23_B" => result.observables.e23_B,
            "chi_B" => result.observables.chi_B,
            "D" => D,
            "chi" => chi,
            "chi_set" => chi_set,
            "distance_max" => distance,
            "init_statenm" => init_statenm,
            "virtual_space" => string(Vv),
            "physical_space" => string(space(state.T, 5)),
            "ctm_ite_num" => ismissing(result.ctm_ite_num) ? -1 : result.ctm_ite_num,
            "ctm_ite_err" => ismissing(result.ctm_ite_err) ? NaN : result.ctm_ite_err,
        );
        compress=false,
    )
    println("Saved chiral-pair correlations to " * mat_filenm)
    flush(stdout)
end

nothing
