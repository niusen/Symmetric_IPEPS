# Version-independent numerical diagnostics for comparing old/new TensorKit.
# Usage: julia --project=<environment> compatibility_probe.jl report.toml
using TensorKit, LinearAlgebra, Random, TOML
BLAS.set_num_threads(10)
if !isdefined(@__MODULE__,:KagomeSU2Cells)
    include(joinpath(@__DIR__,"..","..","..","..","src","bosonic","kagome","simple_update_cells.jl"))
end
using .KagomeSU2Cells

function compatibility_report(path)
    H = heisenberg_triangle()
    report = Dict{String,Any}("TensorKit"=>string(pkgversion(TensorKit)),
                          "julia"=>string(VERSION))
for pattern in (:rectangular,:sqrt3xsqrt3)
    state = initialize_cells(KagomeSUCellGeometry(3,3;pattern);
                             rng=MersenneTwister(92),complex=true)
    # Fill reduced SU(2) blocks deterministically, independent of RNG order
    # and package-specific random-constructor implementations.
    for (r,c) in enumerate(state.cells), (f,name) in enumerate((:T_u,:T_d,:B_a,:B_b,:B_c))
        tensor = getfield(c,name)
        for (sector,block) in blocks(tensor), j in axes(block,2), i in axes(block,1)
            phase = .37r + .23f + .19dim(sector) + .41i + .29j
            block[i,j] = sin(phase) + .3im*cos(1.7phase)
        end
        setfield!(c,name,tensor/norm(tensor))
    end
    for Dmax in (3,4,6)
        updated = itebd_cells(state,H,.002,.001;Dmax)
        estimate = su_energy_estimates(updated,H)
        spectra = Float64[]
        dimensions = Int[]
        for c in updated.cells, name in (:lambda_u_a,:lambda_u_b,:lambda_u_c,
                                         :lambda_d_a,:lambda_d_b,:lambda_d_c)
            weight = getfield(c,name)
            push!(dimensions,dim(space(weight,1)))
            append!(spectra,sort(abs.(diag(convert(Array,weight)));rev=true))
        end
        report["$(pattern)_D$(Dmax)"] = Dict(
            "up_energy"=>estimate.up,"down_energy"=>estimate.down,
            "dimensions"=>dimensions,"lambda_spectra"=>spectra)
    end
    end
    open(path,"w") do io
    TOML.print(io,report;sorted=true)
    end
    println("Wrote compatibility diagnostics: ",path)
    return report
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 1 || error("provide an output TOML path")
    compatibility_report(ARGS[1])
end
