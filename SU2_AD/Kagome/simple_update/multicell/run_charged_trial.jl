# Tiny trial of a polarized auxiliary spin, not a production plateau scan.
using TensorKit, LinearAlgebra, Random, TOML, JLD2
if !isdefined(@__MODULE__,:KagomeChargedSU2)
    include(joinpath(@__DIR__,"..","..","..","src","bosonic","kagome",
                     "simple_update_charged_cell.jl"))
end
using .KagomeChargedSU2
const ChargedBase = KagomeChargedSU2.BaseSU

function finite_sector_ground_energy()
    # Nine-site Sz=-1/2 subspace has only binomial(9,5)=126 basis states.
    indices = [i for i in 0:511 if count_ones(i) == 5]
    positions = Dict(value=>i for (i,value) in enumerate(indices))
    matrix = zeros(length(indices),length(indices))
    for (column,index) in enumerate(indices), (i,j) in KagomeChargedSU2._torus_bonds()
        bi,bj = (index >> (i-1)) & 1,(index >> (j-1)) & 1
        matrix[column,column] += (.5-bi)*(.5-bj)
        if bi != bj
            target = xor(index,(1 << (i-1)) | (1 << (j-1)))
            matrix[positions[target],column] += .5
        end
    end
    return eigmin(Symmetric(matrix))/9
end

function charged_trial(out;Dmax=3,seed=1234,h=.38,threads=10)
    ispath(out) && error("use a NEW output directory")
    BLAS.set_num_threads(threads)
    H = ChargedBase.heisenberg_triangle()
    state = initialize_charged(;rng=MersenneTwister(seed))
    initial = exact_torus_observables(state;h)
    mkpath(out)
    schedule = [(dt=.01,tau=.02),(dt=.003,tau=.006),(dt=.001,tau=.002)]
    metadata = Dict{String,Any}("Dmax"=>Dmax,"seed"=>seed,"h"=>h,"schedule"=>schedule,
        "auxiliary_spin"=>"1/2","auxiliary_component"=>1,"completed_stages"=>0,
        "TensorKit"=>string(pkgversion(TensorKit)),"production_converged"=>false)
    save_charged_checkpoint(joinpath(out,"initial.jld2"),state,H;metadata)
    for (k,stage) in enumerate(schedule)
        println("stage $k/3: dt=$(stage.dt), tau=$(stage.tau), full steps=2")
        flush(stdout)
        state = itebd_charged(state,H,stage.tau,stage.dt;Dmax)
        metadata["completed_stages"] = k
        save_charged_checkpoint(joinpath(out,"stage_$k.jld2"),state,H;metadata)
    end
    final = exact_torus_observables(state;h)
    ground = finite_sector_ground_energy()
    final.exchange_energy_per_site >= ground-1e-11 || error("energy below exact finite-size bound")
    metadata["measurement"] = final.measurement
    save_charged_checkpoint(joinpath(out,"final.jld2"),state,H;metadata)
    # This explicit projection is what a physical polarized CTMRG must use.
    # Summing the raw auxiliary leg would describe a different, unpolarized state.
    cells = polarized_dense_cells(state)
    site_map = state.raw.geometry.site_map
    jldsave(joinpath(out,"polarized_dense.jld2");cells,site_map,metadata)
    report = Dict("initial"=>Dict(string(k)=>v for (k,v) in pairs(initial)),
                  "final"=>Dict(string(k)=>v for (k,v) in pairs(final)),
                  "Dmax"=>Dmax,"seed"=>seed,"h"=>h,
                  "TensorKit"=>string(pkgversion(TensorKit)),
                  "finite_9_site_sector_ground_energy_per_site"=>ground,
                  "production_converged"=>false,
                  "note"=>"Fixed magnetization by construction; no plateau width demonstrated.")
    open(joinpath(out,"report.toml"),"w") do io
        TOML.print(io,report;sorted=true)
    end
    println("initial: ",initial)
    println("final:   ",final)
    println("exact finite-sector ground energy per site: ",ground)
    println("saved: ",abspath(out))
    return state,report
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) == 1 || error("usage: run_charged_trial.jl NEW_OUTPUT_DIRECTORY")
    charged_trial(ARGS[1];Dmax=parse(Int,get(ENV,"KAGOME_CHARGED_DMAX","3")),
                  seed=parse(Int,get(ENV,"KAGOME_CHARGED_SEED","1234")))
end
