using TensorKit, LinearAlgebra, Random, TOML, JLD2, Dates
if !isdefined(Main,:PAIR_SOURCE)
    const PAIR_SOURCE=normpath(joinpath(@__DIR__,"..","..","..","..","..","src","fermionic","kagome"))
end
isdefined(Main,:KagomeFermionicZ2) || include(joinpath(PAIR_SOURCE,"KagomeFermionicZ2.jl"))
using .KagomeFermionicZ2
isdefined(Main,:KagomeFermionicCTM) || include(joinpath(PAIR_SOURCE,"ctmrg.jl"))
using .KagomeFermionicCTM
isdefined(Main,:KagomeFermionicOptimization) || include(joinpath(PAIR_SOURCE,"optimization","KagomeFermionicOptimization.jl"))
using .KagomeFermionicOptimization
BLAS.set_num_threads(1)
isdefined(Main,:pairing_reference) || include("reference.jl")
if !isdefined(Main,:PAIR_OUTPUT)
    const PAIR_OUTPUT = normpath(joinpath(@__DIR__,"..","..","results","pairing_D4"))
end
mkpath(PAIR_OUTPUT)
pair_name(delta) = "delta"*replace(string(delta),"."=>"p")
asdict(record)=Dict(string(k)=>v for (k,v) in pairs(record))
write_toml(path,data)=open(io -> TOML.print(io,data),path,"w")
function note(directory,message)
    line="$(Dates.now()) $message"
    println(line); flush(stdout)
    open(io -> println(io,line),joinpath(directory,"progress.log"),"a")
end
function sectors(state)
    [Dict("site"=>b,"up"=>[dim(space(state.B[1,1,b],2),Z2Irrep(q)) for q in (0,1)],
          "down"=>[dim(space(state.B[1,1,b],1),Z2Irrep(q)) for q in (0,1)]) for b in 1:3]
end

function pair_measure(directory,label,state,parameters,exact; settings=((16,1e-8),(32,1e-8),(32,1e-10)))
    records = []
    for (chi,tolerance) in settings
        elapsed = @elapsed result = measure_ctmrg(state,parameters;chi,maxiter=80,tol=1e-8,trun_tol=tolerance,verbose=false)
        ob = result.observables
        row = Dict("label"=>label,"chi"=>chi,"projector_tol"=>tolerance,"energy"=>ob.energy_per_site,
            "energy_error"=>ob.energy_per_site-exact["energy_per_site"],"density"=>ob.density,
            "pair_real"=>real(ob.pair_amplitude),"pair_imag"=>imag(ob.pair_amplitude),
            "density_site_spread"=>maximum(real,ob.n)-minimum(real,ob.n),
            "pair_site_deviation"=>maximum(abs.(ob.pair_amplitude_set .- ob.pair_amplitude)),
            "double_occupancy"=>ob.double_occupancy,"ctm_converged"=>result.converged,
            "ctm_error"=>result.error,"ctm_iterations"=>result.iterations,"seconds"=>elapsed)
        push!(records,row)
        note(directory,"MEASURE $row")
        write_toml(joinpath(directory,"$(label)_measurements.toml"),Dict("measurements"=>records))
        jldsave(joinpath(directory,"$(label)_chi$(chi)_tol$(tolerance)_ob.jld2");observables=ob,parameters,record=row)
    end
    records
end
