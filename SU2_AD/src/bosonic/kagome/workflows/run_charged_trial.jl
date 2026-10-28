# Tiny trial of a polarized auxiliary spin, not a production plateau scan.
using TensorKit, LinearAlgebra, Random, TOML, JLD2
if !isdefined(@__MODULE__,:KagomeChargedSU2)
    include(joinpath(@__DIR__,"..","simple_update_charged_cell.jl"))
end
using .KagomeChargedSU2
if !isdefined(@__MODULE__,:KagomeU1Cells)
    include(joinpath(@__DIR__,"..","simple_update_u1_cells.jl"))
end
using .KagomeU1Cells
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

function charged_trial(out;Dmax=3,seed=1234,h=.38,threads=10,symmetry=:su2,initial_state=nothing,schedule=nothing)
    ispath(out) && error("use a NEW output directory")
    symmetry in (:su2,:u1) || error("symmetry must be su2 or u1")
    BLAS.set_num_threads(threads)
    if isnothing(initial_state)
        state=initialize_charged(;rng=MersenneTwister(seed))
    else
        data=JLD2.load(initial_state)
        state=data["format"]=="kagome_u1_cells_v1" ? first(load_u1_checkpoint(initial_state)) :
              first(load_charged_checkpoint(initial_state))
    end
    if symmetry==:u1
        state=state isa U1KagomeState ? state : to_u1(state)
        H=u1_heisenberg_triangle()
        observe=(s)->u1_torus_observables(s;h)
        evolve=(s,tau,dt)->itebd_u1(s,H,tau,dt;Dmax)
        save_state=(p,s,m)->save_u1_checkpoint(p,s,H;metadata=m)
    else
        state isa U1KagomeState && error("cannot convert a general U1 state back to SU2")
        H=ChargedBase.heisenberg_triangle()
        observe=(s)->exact_torus_observables(s;h)
        evolve=(s,tau,dt)->itebd_charged(s,H,tau,dt;Dmax)
        save_state=(p,s,m)->save_charged_checkpoint(p,s,H;metadata=m)
    end
    initial=observe(state)
    mkpath(out)
    schedule=isnothing(schedule) ? [(dt=.01,tau=.02),(dt=.003,tau=.006),(dt=.001,tau=.002)] : schedule
    all(isfinite(s.dt) && s.dt>0 && isfinite(s.tau) && s.tau>=0 for s in schedule) || error("invalid schedule")
    metadata = Dict{String,Any}("Dmax"=>Dmax,"seed"=>seed,"h"=>h,"schedule"=>schedule,
        "auxiliary_spin"=>"1/2","auxiliary_component"=>1,"completed_stages"=>0,
        "TensorKit"=>string(pkgversion(TensorKit)),"production_converged"=>false,
        "symmetry"=>string(symmetry),"initialization"=>isnothing(initial_state) ? "SU2 charge projection" : abspath(initial_state))
    save_state(joinpath(out,"initial.jld2"),state,metadata)
    for (k,stage) in enumerate(schedule)
        println("stage $k/$(length(schedule)): dt=$(stage.dt), tau=$(stage.tau), full steps=$(round(Int,stage.tau/stage.dt))")
        flush(stdout)
        state = evolve(state,stage.tau,stage.dt)
        metadata["completed_stages"] = k
        save_state(joinpath(out,"stage_$k.jld2"),state,metadata)
    end
    final=observe(state)
    ground = finite_sector_ground_energy()
    final.exchange_energy_per_site >= ground-1e-11 || error("energy below exact finite-size bound")
    metadata["measurement"] = final.measurement
    save_state(joinpath(out,"final.jld2"),state,metadata)
    # This explicit projection is what a physical polarized CTMRG must use.
    # Summing the raw auxiliary leg selects a different auxiliary boundary ket;
    # it is not the fixed-Sz projection used here.
    if symmetry==:u1
        portable=export_u1_portable(state)
        open(joinpath(out,"u1_state.toml"),"w") do io
            TOML.print(io,portable;sorted=true)
        end
    else
        cells=polarized_dense_cells(state)
        site_map=state.raw.geometry.site_map
        jldsave(joinpath(out,"polarized_dense.jld2");cells,site_map,metadata)
    end
    report = Dict("initial"=>Dict(string(k)=>v for (k,v) in pairs(initial)),
                  "final"=>Dict(string(k)=>v for (k,v) in pairs(final)),
                  "Dmax"=>Dmax,"seed"=>seed,"h"=>h,"symmetry"=>string(symmetry),
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
    length(ARGS)>=1 || error("usage: run_charged_trial.jl NEW_OUT [--symmetry su2|u1] [--init CHECKPOINT]")
    options=Dict("symmetry"=>get(ENV,"KAGOME_CHARGED_SYMMETRY","su2"),"init"=>"",
        "Dmax"=>get(ENV,"KAGOME_CHARGED_DMAX","3"),"seed"=>get(ENV,"KAGOME_CHARGED_SEED","1234"),
        "h"=>"0.38","schedule"=>"")
    for i in 2:2:length(ARGS)
        key=startswith(ARGS[i],"--") ? ARGS[i][3:end] : ""
        haskey(options,key) && i<length(ARGS) || error("invalid option")
        options[key]=ARGS[i+1]
    end
    schedule=isempty(options["schedule"]) ? nothing : [begin
        dt,tau=parse.(Float64,split(item,':'));(dt=dt,tau=tau)
    end for item in split(options["schedule"],',')]
    charged_trial(ARGS[1];Dmax=parse(Int,options["Dmax"]),seed=parse(Int,options["seed"]),
        h=parse(Float64,options["h"]),symmetry=Symbol(lowercase(options["symmetry"])),schedule,
        initial_state=isempty(options["init"]) ? nothing : options["init"])
end
