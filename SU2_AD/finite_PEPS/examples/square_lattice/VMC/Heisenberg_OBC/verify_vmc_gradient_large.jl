using LinearAlgebra: I, diagm, diag, dot
using Random
using Statistics
using TensorKit
using JLD2

import TensorKit: permute, AbstractTensorMap

# The production sources use TensorKit's former three-argument permutation
# spelling. This local forwarding method also works with current TensorKit.
permute(tensor::AbstractTensorMap, codomain_axes::Tuple, domain_axes::Tuple; kwargs...) =
    permute(tensor, (codomain_axes, domain_axes); kwargs...)

const scalar_name = get(ENV, "PEPS_SCALAR", "Float64")
const global_eltype = scalar_name == "ComplexF64" ? ComplexF64 : Float64
scalar_name in ("Float64", "ComplexF64") || error("PEPS_SCALAR must be Float64 or ComplexF64")

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
env_bool(name, default) = lowercase(get(ENV, name, string(default))) in ("1", "true", "yes", "on")

const Lattice = "square"
const Lx = env_int("PEPS_LX", 10)
const Ly = env_int("PEPS_LY", 10)
const D = env_int("PEPS_D", 3)
const chi = env_int("PEPS_CHI", 2 * D)
const use_mps_sweep = env_bool("PEPS_USE_MPS_SWEEP", false)
const n_mps_sweep = env_int("PEPS_MPS_SWEEPS", 0)
const L = Lx * Ly
const Nbra = L
const Ne = L
const GC_spacing = env_int("GRADCHECK_GC_SPACING", 200)

const sample_count = env_int("GRADCHECK_SAMPLES", 16)
const burnin_sweeps = env_int("GRADCHECK_BURNIN_SWEEPS", 50)
const spacing_sweeps = env_int("GRADCHECK_SPACING_SWEEPS", 5)
const direction_count = env_int("GRADCHECK_DIRECTIONS", 2)
const amplitude_check_count = env_int("GRADCHECK_AMPLITUDE_SAMPLES", 4)
const block_size = env_int("GRADCHECK_BLOCK_SIZE", 4)
const random_seed = env_int("GRADCHECK_SEED", 20260824)
const normalize_once = env_bool("GRADCHECK_NORMALIZE", true)
const run_energy_fd = env_bool("GRADCHECK_ENERGY_FD", true)
const fd_steps = parse.(Float64, split(get(ENV, "GRADCHECK_STEPS", "1e-3,3e-4"), ','))
const state_name = get(ENV, "PEPS_STATE", "Heisenberg_SU_$(Lx)x$(Ly)_D$(D)")
const output_file = abspath(get(ENV, "GRADCHECK_OUTPUT", "large_gradient_check_$(Lx)x$(Ly)_D$(D)_chi$(chi).jld2"))

sample_count > 1 || error("GRADCHECK_SAMPLES must exceed one")
direction_count > 0 || error("GRADCHECK_DIRECTIONS must be positive")
all(fd_steps .> 0) || error("all GRADCHECK_STEPS must be positive")

cd(@__DIR__)

include("../../../../state/iPEPS_ansatz.jl")
include("../../../../setting/Settings.jl")
include("../../../../setting/linearalgebra.jl")
include("../../../../setting/tuple_methods.jl")
include("../../../../environment/MC/finite_clusters.jl")
include("../../../../environment/MC/contract_disk.jl")
include("../../../../environment/MC/sampling.jl")
include("../../../../environment/MC/mps_sweep.jl")

global contract_fun = contract_whole_disk
global ite_num = 0

# The checked-out development TensorKit removed truncdim/truncbelow. This
# fallback is only selected in that API and keeps the full SVD; it is exact for
# the local 4x4 D=2, chi=10 smoke test. Production/server environments with the
# legacy truncation API retain the original chi-truncated implementation.
if !isdefined(Main, :truncdim)
    function normalized_tsvd(tensor::TensorMap, ::Int)
        tensor_norm = norm(tensor)
        u, s, v = TensorKit.svd_compact(tensor / tensor_norm)
        return u, s * tensor_norm, v
    end
end


function validation_decompose_physical_legs(fpeps::Matrix{TensorMap}, vp)
    if !(vp isa ComplexSpace)
        return decompose_physical_legs(fpeps, vp)
    end

    decomposed = Array{TensorMap}(undef, Lx, Ly, TensorKit.dim(vp))
    for physical in 1:TensorKit.dim(vp), cx in 1:Lx, cy in 1:Ly
        tensor = fpeps[cx, cy]
        virtual_rank = Rank(tensor) - 1
        inds = ntuple(_ -> Colon(), virtual_rank)
        data = Array(tensor[inds..., physical])
        virtual_space = space(tensor, 1)
        for leg in 2:virtual_rank
            virtual_space *= space(tensor, leg)
        end
        decomposed[cx, cy, physical] = TensorMap(
            data, virtual_space, ProductSpace{ComplexSpace,0}()
        )
    end
    return decomposed
end


# Make the one optional normalize_PEPS! call use the same dense-copy path on
# current TensorKit. On legacy TensorKit this simply replaces an equivalent
# ComplexSpace method inside this validation script.
function apply_sampling_projector(
    fpeps::Matrix{TensorMap}, config::Array, sample::Matrix{TensorMap}, vp::ComplexSpace
)
    decomposed = validation_decompose_physical_legs(fpeps, vp)
    return pick_sample(decomposed, vec(config), sample)
end


function validation_set_grad_config(holes, config, psi, vp)
    !(vp isa ComplexSpace) && return set_grad_config(holes, config, psi)

    result = similar(psi)
    for cx in 1:Lx, cy in 1:Ly
        physical = Int((-config[cx + Lx * (cy - 1)] + 1) / 2 + 1)
        template = psi[cx, cy]
        data = zeros(eltype(template), size(convert(Array, template)))
        selectdim(data, ndims(data), physical) .= convert(Array, holes[cx, cy])
        result[cx, cy] = TensorMap(data, codomain(template), domain(template))
    end
    return result
end


function peps_norm(x::Matrix{TensorMap})
    return sqrt(sum(norm(tensor)^2 for tensor in x))
end


function random_tensor_like(tensor::TensorMap)
    # Legacy TensorKit accepts TensorMap(randn, spaces), which preserves
    # symmetry blocks. Current dense TensorKit uses an explicit array.
    try
        return TensorMap(randn, codomain(tensor), domain(tensor))
    catch error
        sectortype(space(tensor, 1)) == Trivial || rethrow(error)
        data = randn(size(convert(Array, tensor)))
        return TensorMap(data, codomain(tensor), domain(tensor))
    end
end


function random_tangent_direction(psi::Matrix{TensorMap})
    direction = similar(psi)
    for site in eachindex(psi)
        tensor = psi[site]
        noise = random_tensor_like(tensor)
        if eltype(tensor) <: Complex
            noise = noise + im * random_tensor_like(tensor)
        end

        # Remove the local complex rescaling gauge A -> c*A. This improves the
        # signal without changing the physical tangent space being tested.
        coefficient = dot(tensor, noise) / dot(tensor, tensor)
        direction[site] = noise - coefficient * tensor
    end
    scale = peps_norm(psi) / peps_norm(direction)
    for site in eachindex(direction)
        direction[site] = scale * direction[site]
    end
    return direction
end


function perturb_peps(psi::Matrix{TensorMap}, direction::Matrix{TensorMap}, step)
    result = similar(psi)
    for site in eachindex(psi)
        result[site] = psi[site] + step * direction[site]
    end
    return result
end


function directional_log_derivative(log_gradient, direction)
    value = zero(ComplexF64)
    for site in eachindex(log_gradient)
        # This is a holomorphic contraction: sum O_i * v_i, with no complex
        # conjugation. LinearAlgebra.dot would conjugate its first argument.
        value += sum(convert(Array, log_gradient[site]) .* convert(Array, direction[site]))
    end
    return value
end


function amplitude_only(decomposed, config, vp)
    sample = Matrix{TensorMap}(undef, Lx, Ly)
    amplitude, sample, _ = contract_sample(
        decomposed, Lx, Ly, config, sample, vp, contract_fun
    )
    return amplitude
end


function local_energy_from_amplitude(decomposed, config, vp, amplitude, sample, nn_reduced)
    energy = zero(global_eltype)
    for site in 1:L
        for neighbor in nn_reduced[site]
            if config[site] == config[neighbor]
                energy += 0.25
            else
                flipped = flip_config(config, site, neighbor)
                amplitude_flip, sample, _ = contract_sample(
                    decomposed, Lx, Ly, flipped, sample, vp, contract_fun
                )
                energy += 0.5 * amplitude_flip / amplitude - 0.25
            end
        end
    end
    return energy
end


function local_energy_only(decomposed, config, vp, nn_reduced)
    sample = Matrix{TensorMap}(undef, Lx, Ly)
    amplitude, sample, _ = contract_sample(
        decomposed, Lx, Ly, config, sample, vp, contract_fun
    )
    energy = local_energy_from_amplitude(
        decomposed, config, vp, amplitude, sample, nn_reduced
    )
    return energy, amplitude
end


function local_energy_and_log_gradient(psi, decomposed, config, vp, nn_reduced)
    sample = Matrix{TensorMap}(undef, Lx, Ly)
    amplitude, sample, _ = contract_sample(
        decomposed, Lx, Ly, config, sample, vp, contract_fun
    )
    holes = contract_disk_derivative(sample, config, chi)
    log_gradient = validation_set_grad_config(holes, config, psi, vp) / amplitude
    energy = local_energy_from_amplitude(
        decomposed, config, vp, amplitude, sample, nn_reduced
    )
    return energy, amplitude, log_gradient
end


function draw_configurations(decomposed, initial_config, vp)
    _, _, _, nn, _, _, _ = get_neighbours_square(Lx, Ly, "OBC")
    config = deepcopy(initial_config)
    sample = Matrix{TensorMap}(undef, Lx, Ly)
    history = disk_contract_history(
        zeros(Int8, L), Matrix{TensorMap}(undef, Lx, Ly), Matrix{TensorMap}(undef, Lx, Ly)
    )
    amplitude, sample, _, history = partial_contract_sample(
        decomposed, config, sample, vp, history
    )

    configurations = Vector{Vector{Int8}}()
    total_sweeps = burnin_sweeps + sample_count * spacing_sweeps
    for sweep in 1:total_sweeps
        global ite_num = sweep
        for _ in 1:L
            site = rand(1:L)
            neighbor = nn[site][rand(1:length(nn[site]))]
            config[site] == config[neighbor] && continue

            flipped = flip_config(config, site, neighbor)
            amplitude_flip, sample, _, history_flip = partial_contract_sample(
                decomposed, flipped, sample, vp, history
            )
            if rand() < abs2(amplitude_flip / amplitude)
                config = deepcopy(flipped)
                amplitude = amplitude_flip
                history = deepcopy(history_flip)
            end
        end

        if sweep > burnin_sweeps && mod(sweep - burnin_sweeps, spacing_sweeps) == 0
            push!(configurations, deepcopy(config))
            println("sampled configuration ", length(configurations), "/", sample_count)
            flush(stdout)
        end
    end
    @assert length(configurations) == sample_count
    return configurations
end


function covariance_prediction(energies, directional_logs)
    raw = mean(conj.(energies) .* directional_logs) -
          conj(mean(energies)) * mean(directional_logs)
    return 2 * real(raw), raw
end


function correlated_energy(energies, amplitudes, reference_amplitudes)
    weights = abs2.(amplitudes ./ reference_amplitudes)
    return real(sum(weights .* energies) / sum(weights))
end


function block_jackknife_standard_error(
    energies, directional_logs, plus_energies, minus_energies,
    plus_amplitudes, minus_amplitudes, reference_amplitudes, step,
)
    blocks = [start:min(start + block_size - 1, length(energies))
              for start in 1:block_size:length(energies)]
    blocks = filter(block -> length(block) == block_size, blocks)
    length(blocks) >= 2 || return NaN

    delete_block_estimates = Float64[]
    all_indices = collect(eachindex(energies))
    for block in blocks
        kept = setdiff(all_indices, collect(block))
        isempty(kept) && continue
        predicted, _ = covariance_prediction(energies[kept], directional_logs[kept])
        eplus = correlated_energy(
            plus_energies[kept], plus_amplitudes[kept], reference_amplitudes[kept]
        )
        eminus = correlated_energy(
            minus_energies[kept], minus_amplitudes[kept], reference_amplitudes[kept]
        )
        push!(delete_block_estimates, (eplus - eminus) / (2 * step) - predicted)
    end
    count = length(delete_block_estimates)
    count >= 2 || return NaN
    center = mean(delete_block_estimates)
    return sqrt((count - 1) / count * sum((delete_block_estimates .- center).^2))
end


println("large PEPS VMC gradient validation")
@show Lx Ly D chi global_eltype use_mps_sweep n_mps_sweep
@show state_name sample_count burnin_sweeps spacing_sweeps
@show direction_count fd_steps run_energy_fd random_seed
flush(stdout)

Random.seed!(random_seed)
psi, Vp = load_fPEPS(Lx, Ly, state_name)
if normalize_once
    config_start = normalize_PEPS!(psi, Vp, contract_fun)
else
    config_start = initial_Neel_config_square(Lx, Ly, 1)
end
psi_ref = deepcopy(psi)
decomposed_ref = validation_decompose_physical_legs(psi_ref, Vp)

directions = [random_tangent_direction(psi_ref) for _ in 1:direction_count]
plus_decomposed = Matrix{Any}(undef, direction_count, length(fd_steps))
minus_decomposed = similar(plus_decomposed)
for direction_index in 1:direction_count, step_index in eachindex(fd_steps)
    step = fd_steps[step_index]
    plus_decomposed[direction_index, step_index] = validation_decompose_physical_legs(
        perturb_peps(psi_ref, directions[direction_index], step), Vp
    )
    minus_decomposed[direction_index, step_index] = validation_decompose_physical_legs(
        perturb_peps(psi_ref, directions[direction_index], -step), Vp
    )
end

configurations = draw_configurations(decomposed_ref, config_start, Vp)
_, _, _, _, _, nn_reduced, _ = get_neighbours_square(Lx, Ly, "OBC")

energies = zeros(ComplexF64, sample_count)
reference_amplitudes = zeros(ComplexF64, sample_count)
directional_logs = zeros(ComplexF64, sample_count, direction_count)
for sample_index in 1:sample_count
    energy, amplitude, log_gradient = local_energy_and_log_gradient(
        psi_ref, decomposed_ref, configurations[sample_index], Vp, nn_reduced
    )
    energies[sample_index] = energy
    reference_amplitudes[sample_index] = amplitude
    for direction_index in 1:direction_count
        directional_logs[sample_index, direction_index] = directional_log_derivative(
            log_gradient, directions[direction_index]
        )
    end
    println("reference gradient sample ", sample_index, "/", sample_count)
    flush(stdout)
end

plus_energies = zeros(ComplexF64, sample_count, direction_count, length(fd_steps))
minus_energies = similar(plus_energies)
plus_amplitudes = similar(plus_energies)
minus_amplitudes = similar(plus_energies)

evaluation_count = run_energy_fd ? sample_count : min(sample_count, amplitude_check_count)
for sample_index in 1:evaluation_count
    config = configurations[sample_index]
    for direction_index in 1:direction_count, step_index in eachindex(fd_steps)
        if run_energy_fd
            plus_energies[sample_index, direction_index, step_index],
            plus_amplitudes[sample_index, direction_index, step_index] = local_energy_only(
                plus_decomposed[direction_index, step_index], config, Vp, nn_reduced
            )
            minus_energies[sample_index, direction_index, step_index],
            minus_amplitudes[sample_index, direction_index, step_index] = local_energy_only(
                minus_decomposed[direction_index, step_index], config, Vp, nn_reduced
            )
        else
            plus_amplitudes[sample_index, direction_index, step_index] = amplitude_only(
                plus_decomposed[direction_index, step_index], config, Vp
            )
            minus_amplitudes[sample_index, direction_index, step_index] = amplitude_only(
                minus_decomposed[direction_index, step_index], config, Vp
            )
        end
    end
    println("finite-difference sample ", sample_index, "/", evaluation_count)
    flush(stdout)
end

amplitude_relative_errors = fill(NaN, direction_count, length(fd_steps))
amplitude_absolute_errors = fill(NaN, direction_count, length(fd_steps))
checked = min(sample_count, amplitude_check_count)
for direction_index in 1:direction_count, step_index in eachindex(fd_steps)
    step = fd_steps[step_index]
    finite_logs = (
        plus_amplitudes[1:checked, direction_index, step_index] .-
        minus_amplitudes[1:checked, direction_index, step_index]
    ) ./ (2 * step .* reference_amplitudes[1:checked])
    analytic_logs = directional_logs[1:checked, direction_index]
    absolute = abs.(finite_logs .- analytic_logs)
    relative = absolute ./ max.(abs.(analytic_logs), 1e-12)
    amplitude_absolute_errors[direction_index, step_index] = maximum(absolute)
    amplitude_relative_errors[direction_index, step_index] = maximum(relative)
end

predicted_derivatives = zeros(Float64, direction_count)
raw_covariances = zeros(ComplexF64, direction_count)
finite_derivatives = fill(NaN, direction_count, length(fd_steps))
difference_standard_errors = fill(NaN, direction_count, length(fd_steps))
z_scores = fill(NaN, direction_count, length(fd_steps))

for direction_index in 1:direction_count
    predicted_derivatives[direction_index], raw_covariances[direction_index] =
        covariance_prediction(energies, directional_logs[:, direction_index])
    for step_index in eachindex(fd_steps)
        if run_energy_fd
            step = fd_steps[step_index]
            eplus = correlated_energy(
                plus_energies[:, direction_index, step_index],
                plus_amplitudes[:, direction_index, step_index],
                reference_amplitudes,
            )
            eminus = correlated_energy(
                minus_energies[:, direction_index, step_index],
                minus_amplitudes[:, direction_index, step_index],
                reference_amplitudes,
            )
            finite_derivatives[direction_index, step_index] = (eplus - eminus) / (2 * step)
            difference_standard_errors[direction_index, step_index] = block_jackknife_standard_error(
                energies,
                directional_logs[:, direction_index],
                plus_energies[:, direction_index, step_index],
                minus_energies[:, direction_index, step_index],
                plus_amplitudes[:, direction_index, step_index],
                minus_amplitudes[:, direction_index, step_index],
                reference_amplitudes,
                step,
            )
            difference = finite_derivatives[direction_index, step_index] -
                         predicted_derivatives[direction_index]
            standard_error = difference_standard_errors[direction_index, step_index]
            z_scores[direction_index, step_index] = difference / standard_error
        end
    end
end

println()
println("deterministic amplitude directional checks")
for direction_index in 1:direction_count, step_index in eachindex(fd_steps)
    println(
        "direction=", direction_index,
        " h=", fd_steps[step_index],
        " max_abs=", amplitude_absolute_errors[direction_index, step_index],
        " max_rel=", amplitude_relative_errors[direction_index, step_index],
    )
end

println()
println("VMC energy directional checks")
for direction_index in 1:direction_count
    println("direction=", direction_index, " covariance prediction=", predicted_derivatives[direction_index])
    if run_energy_fd
        for step_index in eachindex(fd_steps)
            println(
                "  h=", fd_steps[step_index],
                " correlated_FD=", finite_derivatives[direction_index, step_index],
                " difference=", finite_derivatives[direction_index, step_index] - predicted_derivatives[direction_index],
                " block_SE=", difference_standard_errors[direction_index, step_index],
                " z=", z_scores[direction_index, step_index],
            )
        end
    end
end
flush(stdout)

mkpath(dirname(output_file))
jldsave(
    output_file;
    Lx,
    Ly,
    D,
    chi,
    global_eltype,
    state_name,
    sample_count,
    burnin_sweeps,
    spacing_sweeps,
    random_seed,
    fd_steps,
    run_energy_fd,
    configurations,
    directions,
    energies,
    reference_amplitudes,
    directional_logs,
    raw_covariances,
    predicted_derivatives,
    finite_derivatives,
    difference_standard_errors,
    z_scores,
    amplitude_absolute_errors,
    amplitude_relative_errors,
)
println("wrote ", output_file)
