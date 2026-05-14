module CleanVMCPEPS

using LinearAlgebra
using Random
using Printf

export PEPS, random_peps, neel_config, all_sz0_configs, amplitude, amplitude_grad,
       energy_and_grad_exact, finite_difference_check, mc_energy_and_grad,
       markov_energy_and_grad, run_all_checks

struct PEPS
    tensors::Matrix{Array{Float64,5}}
    Lx::Int
    Ly::Int
    D::Int
end

left_dim(x, D) = x == 1 ? 1 : D
right_dim(x, Lx, D) = x == Lx ? 1 : D
down_dim(y, D) = y == 1 ? 1 : D
up_dim(y, Ly, D) = y == Ly ? 1 : D

function random_peps(Lx::Int=4, Ly::Int=4, D::Int=2; seed::Int=1)
    rng = MersenneTwister(seed)
    tensors = Matrix{Array{Float64,5}}(undef, Lx, Ly)
    for x in 1:Lx, y in 1:Ly
        dims = (left_dim(x, D), down_dim(y, D), right_dim(x, Lx, D), up_dim(y, Ly, D), 2)
        tensors[x, y] = 0.2 .* randn(rng, dims...)
    end
    return PEPS(tensors, Lx, Ly, D)
end

function copy_peps(psi::PEPS)
    tensors = Matrix{Array{Float64,5}}(undef, psi.Lx, psi.Ly)
    for x in 1:psi.Lx, y in 1:psi.Ly
        tensors[x, y] = copy(psi.tensors[x, y])
    end
    return PEPS(tensors, psi.Lx, psi.Ly, psi.D)
end

function neel_config(Lx::Int, Ly::Int)
    return [isodd(x + y) ? 1 : 2 for x in 1:Lx, y in 1:Ly]
end

function all_sz0_configs(Lx::Int, Ly::Int)
    L = Lx * Ly
    @assert iseven(L)
    configs = Vector{Matrix{Int}}()
    for bits in UInt64(0):(UInt64(1) << L)-UInt64(1)
        count_ones(bits) == L ÷ 2 || continue
        cfg = Matrix{Int}(undef, Lx, Ly)
        for site in 1:L
            x = mod1(site, Lx)
            y = div(site - 1, Lx) + 1
            cfg[x, y] = ((bits >> (site - 1)) & UInt64(1)) == 1 ? 1 : 2
        end
        push!(configs, cfg)
    end
    return configs
end

function tuples_of_dims(dims::Vector{Int})
    out = Vector{Vector{Int}}()
    cur = ones(Int, length(dims))
    function rec(k)
        if k > length(dims)
            push!(out, copy(cur))
        else
            for v in 1:dims[k]
                cur[k] = v
                rec(k + 1)
            end
        end
    end
    rec(1)
    return out
end

function boundary_tuples(psi::PEPS, y::Int, dir::Symbol)
    dims = if dir == :down
        [down_dim(y, psi.D) for _ in 1:psi.Lx]
    elseif dir == :up
        [up_dim(y, psi.Ly, psi.D) for _ in 1:psi.Lx]
    else
        error("dir must be :down or :up")
    end
    return tuples_of_dims(dims)
end

function row_transfer(psi::PEPS, cfg::Matrix{Int}, y::Int)
    downs = boundary_tuples(psi, y, :down)
    ups = boundary_tuples(psi, y, :up)
    T = zeros(length(downs), length(ups))
    for (id, dtuple) in pairs(downs), (iu, utuple) in pairs(ups)
        vals = [1.0]
        for x in 1:psi.Lx
            A = psi.tensors[x, y]
            p = cfg[x, y]
            newvals = zeros(size(A, 3))
            for l in 1:length(vals), r in 1:size(A, 3)
                newvals[r] += vals[l] * A[l, dtuple[x], r, utuple[x], p]
            end
            vals = newvals
        end
        T[id, iu] = vals[1]
    end
    return T
end

function row_transfers(psi::PEPS, cfg::Matrix{Int})
    return [row_transfer(psi, cfg, y) for y in 1:psi.Ly]
end

function amplitude_from_rows(rows::Vector{Matrix{Float64}})
    v = [1.0]
    for T in rows
        v = vec(v' * T)
    end
    return v[1]
end

amplitude(psi::PEPS, cfg::Matrix{Int}) = amplitude_from_rows(row_transfers(psi, cfg))

function row_prefix_suffix(psi::PEPS, cfg::Matrix{Int}, y::Int, dtuple::Vector{Int}, utuple::Vector{Int})
    prefix = Vector{Vector{Float64}}(undef, psi.Lx + 1)
    suffix = Vector{Vector{Float64}}(undef, psi.Lx + 1)
    prefix[1] = [1.0]
    for x in 1:psi.Lx
        A = psi.tensors[x, y]
        p = cfg[x, y]
        v = zeros(size(A, 3))
        for l in 1:length(prefix[x]), r in 1:size(A, 3)
            v[r] += prefix[x][l] * A[l, dtuple[x], r, utuple[x], p]
        end
        prefix[x + 1] = v
    end
    suffix[psi.Lx + 1] = [1.0]
    for x in psi.Lx:-1:1
        A = psi.tensors[x, y]
        p = cfg[x, y]
        v = zeros(size(A, 1))
        for l in 1:size(A, 1), r in 1:length(suffix[x + 1])
            v[l] += A[l, dtuple[x], r, utuple[x], p] * suffix[x + 1][r]
        end
        suffix[x] = v
    end
    return prefix, suffix
end

function amplitude_grad(psi::PEPS, cfg::Matrix{Int})
    rows = row_transfers(psi, cfg)
    amp = amplitude_from_rows(rows)
    grads = zero_like(psi)

    left_env = Vector{Vector{Float64}}(undef, psi.Ly + 1)
    right_env = Vector{Vector{Float64}}(undef, psi.Ly + 1)
    left_env[1] = [1.0]
    for y in 1:psi.Ly
        left_env[y + 1] = vec(left_env[y]' * rows[y])
    end
    right_env[psi.Ly + 1] = [1.0]
    for y in psi.Ly:-1:1
        right_env[y] = rows[y] * right_env[y + 1]
    end

    for y in 1:psi.Ly
        downs = boundary_tuples(psi, y, :down)
        ups = boundary_tuples(psi, y, :up)
        for (id, dtuple) in pairs(downs), (iu, utuple) in pairs(ups)
            row_weight = left_env[y][id] * right_env[y + 1][iu]
            row_weight == 0 && continue
            prefix, suffix = row_prefix_suffix(psi, cfg, y, dtuple, utuple)
            for x in 1:psi.Lx
                p = cfg[x, y]
                for l in eachindex(prefix[x]), r in eachindex(suffix[x + 1])
                    grads[x, y][l, dtuple[x], r, utuple[x], p] += row_weight * prefix[x][l] * suffix[x + 1][r]
                end
            end
        end
    end
    return amp, grads
end

function neighbours_reduced(Lx::Int, Ly::Int)
    bonds = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
    for x in 1:Lx, y in 1:Ly
        x < Lx && push!(bonds, ((x, y), (x + 1, y)))
        y < Ly && push!(bonds, ((x, y), (x, y + 1)))
    end
    return bonds
end

function flipped(cfg::Matrix{Int}, a, b)
    out = copy(cfg)
    out[a...] = cfg[b...]
    out[b...] = cfg[a...]
    return out
end

function config_key(cfg::Matrix{Int})
    key = UInt64(0)
    site = 1
    for y in 1:size(cfg, 2), x in 1:size(cfg, 1)
        cfg[x, y] == 1 && (key |= UInt64(1) << (site - 1))
        site += 1
    end
    return key
end

function amplitudes(psi::PEPS, configs::Vector{Matrix{Int}})
    return Dict(config_key(cfg) => amplitude(psi, cfg) for cfg in configs)
end

function local_energy(cfg::Matrix{Int}, amps::Dict{UInt64,Float64}, bonds)
    amp = amps[config_key(cfg)]
    e = 0.0
    for (a, b) in bonds
        if cfg[a...] == cfg[b...]
            e += 0.25
        else
            e += 0.5 * amps[config_key(flipped(cfg, a, b))] / amp - 0.25
        end
    end
    return e
end

function zero_like(psi::PEPS)
    grads = Matrix{Array{Float64,5}}(undef, psi.Lx, psi.Ly)
    for x in 1:psi.Lx, y in 1:psi.Ly
        grads[x, y] = zeros(size(psi.tensors[x, y]))
    end
    return grads
end

function add_scaled!(dst, a, src)
    for i in eachindex(dst)
        dst[i] .+= a .* src[i]
    end
end

function sub_grads(a, b)
    out = similar(a)
    for i in eachindex(a)
        out[i] = a[i] .- b[i]
    end
    return out
end

function scale_grads(a, g)
    out = similar(g)
    for i in eachindex(g)
        out[i] = a .* g[i]
    end
    return out
end

function energy_and_grad_exact(psi::PEPS)
    configs = all_sz0_configs(psi.Lx, psi.Ly)
    amps = amplitudes(psi, configs)
    bonds = neighbours_reduced(psi.Lx, psi.Ly)
    weights = [amps[config_key(cfg)]^2 for cfg in configs]
    Z = sum(weights)
    E = 0.0
    Omean = zero_like(psi)
    EOmean = zero_like(psi)
    for (cfg, w) in zip(configs, weights)
        p = w / Z
        eloc = local_energy(cfg, amps, bonds)
        amp, grad_amp = amplitude_grad(psi, cfg)
        O = scale_grads(1 / amp, grad_amp)
        E += p * eloc
        add_scaled!(Omean, p, O)
        add_scaled!(EOmean, p * eloc, O)
    end
    grad = scale_grads(2.0, sub_grads(EOmean, scale_grads(E, Omean)))
    return E, grad
end

function finite_difference_check(psi::PEPS, grad; checks=8, dt=1e-6, seed=2)
    rng = MersenneTwister(seed)
    E0, _ = energy_and_grad_exact(psi)
    worst = 0.0
    for _ in 1:checks
        x = rand(rng, 1:psi.Lx)
        y = rand(rng, 1:psi.Ly)
        idx = rand(rng, eachindex(psi.tensors[x, y]))
        psi2 = copy_peps(psi)
        psi2.tensors[x, y][idx] += dt
        E1, _ = energy_and_grad_exact(psi2)
        fd = (E1 - E0) / dt
        pred = grad[x, y][idx]
        rel = abs(fd - pred) / max(abs(fd), abs(pred), 1e-14)
        worst = max(worst, rel)
        @printf("FD check site (%d,%d) idx %s: fd %.12e grad %.12e rel %.3e\n",
                x, y, string(idx), fd, pred, rel)
    end
    return worst
end

function mc_energy_and_grad(psi::PEPS; nsamples::Int=1000, seed::Int=3)
    rng = MersenneTwister(seed)
    configs = all_sz0_configs(psi.Lx, psi.Ly)
    amps = amplitudes(psi, configs)
    bonds = neighbours_reduced(psi.Lx, psi.Ly)
    weights = [amps[config_key(cfg)]^2 for cfg in configs]
    cdf = cumsum(weights ./ sum(weights))
    cdf[end] = 1.0

    E = 0.0
    Omean = zero_like(psi)
    EOmean = zero_like(psi)
    for _ in 1:nsamples
        cfg = configs[searchsortedfirst(cdf, rand(rng))]
        eloc = local_energy(cfg, amps, bonds)
        amp, grad_amp = amplitude_grad(psi, cfg)
        O = scale_grads(1 / amp, grad_amp)
        E += eloc
        add_scaled!(Omean, 1.0, O)
        add_scaled!(EOmean, eloc, O)
    end
    E /= nsamples
    Omean = scale_grads(1 / nsamples, Omean)
    EOmean = scale_grads(1 / nsamples, EOmean)
    grad = scale_grads(2.0, sub_grads(EOmean, scale_grads(E, Omean)))
    return E, grad
end

function markov_energy_and_grad(psi::PEPS; nsamples::Int=1000, therm::Int=1000,
                                sweeps_between::Int=1, seed::Int=4)
    rng = MersenneTwister(seed)
    cfg = neel_config(psi.Lx, psi.Ly)
    bonds = neighbours_reduced(psi.Lx, psi.Ly)
    amp = amplitude(psi, cfg)

    function sweep!(cfg, amp)
        for _ in 1:length(bonds)
            a, b = bonds[rand(rng, 1:length(bonds))]
            cfg[a...] == cfg[b...] && continue
            proposed = flipped(cfg, a, b)
            amp_new = amplitude(psi, proposed)
            ratio = (amp_new / amp)^2
            if rand(rng) < min(1.0, ratio)
                cfg = proposed
                amp = amp_new
            end
        end
        return cfg, amp
    end

    for _ in 1:therm
        cfg, amp = sweep!(cfg, amp)
    end

    E = 0.0
    Omean = zero_like(psi)
    EOmean = zero_like(psi)
    for _ in 1:nsamples
        for _ in 1:sweeps_between
            cfg, amp = sweep!(cfg, amp)
        end

        # Local energy only needs amplitudes of nearest-neighbour flips.
        eloc = 0.0
        for (a, b) in bonds
            if cfg[a...] == cfg[b...]
                eloc += 0.25
            else
                amp_flip = amplitude(psi, flipped(cfg, a, b))
                eloc += 0.5 * amp_flip / amp - 0.25
            end
        end

        amp_check, grad_amp = amplitude_grad(psi, cfg)
        @assert abs(amp_check - amp) / max(abs(amp), 1e-14) < 1e-10
        O = scale_grads(1 / amp, grad_amp)
        E += eloc
        add_scaled!(Omean, 1.0, O)
        add_scaled!(EOmean, eloc, O)
    end
    E /= nsamples
    Omean = scale_grads(1 / nsamples, Omean)
    EOmean = scale_grads(1 / nsamples, EOmean)
    grad = scale_grads(2.0, sub_grads(EOmean, scale_grads(E, Omean)))
    return E, grad
end

function grad_norm(g)
    return sqrt(sum(sum(abs2, gi) for gi in g))
end

function grad_overlap(a, b)
    num = sum(sum(a[i] .* b[i]) for i in eachindex(a))
    den = grad_norm(a) * grad_norm(b)
    return num / den
end

function run_all_checks()
    psi = random_peps(4, 4, 2; seed=11)
    cfg = neel_config(psi.Lx, psi.Ly)
    amp, grad_amp = amplitude_grad(psi, cfg)
    println("Single-config amplitude = ", amp)

    # Single amplitude derivative check.
    for (x, y, idx) in ((1, 1, CartesianIndex(1, 1, 1, 1, 1)),
                        (2, 2, CartesianIndex(2, 2, 1, 2, 1)),
                        (4, 4, CartesianIndex(1, 2, 1, 1, 2)))
        psi2 = copy_peps(psi)
        dt = 1e-6
        psi2.tensors[x, y][idx] += dt
        fd = (amplitude(psi2, cfg) - amp) / dt
        pred = grad_amp[x, y][idx]
        @printf("Amplitude grad check (%d,%d) %s: fd %.12e grad %.12e rel %.3e\n",
                x, y, string(idx), fd, pred,
                abs(fd - pred) / max(abs(fd), abs(pred), 1e-14))
    end

    E, grad = energy_and_grad_exact(psi)
    println("Exact VMC energy = ", E)
    worst = finite_difference_check(psi, grad; checks=8)
    println("Worst exact-gradient FD relative error = ", worst)

    for n in (200, 1000, 10000)
        Emc, gmc = mc_energy_and_grad(psi; nsamples=n, seed=20)
        @printf("MC nsamples=%5d: E %.8f, grad overlap %.6f, relnorm %.3e\n",
                n, Emc, grad_overlap(gmc, grad), grad_norm(sub_grads(gmc, grad)) / grad_norm(grad))
    end

    for n in (200, 1000, 10000)
        Emarkov, gmarkov = markov_energy_and_grad(psi; nsamples=n, therm=1000,
                                                  sweeps_between=2, seed=30)
        @printf("Markov nsamples=%5d: E %.8f, grad overlap %.6f, relnorm %.3e\n",
                n, Emarkov, grad_overlap(gmarkov, grad), grad_norm(sub_grads(gmarkov, grad)) / grad_norm(grad))
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    CleanVMCPEPS.run_all_checks()
end
