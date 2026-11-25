using TensorKit, LinearAlgebra, Random, TOML, JLD2, Dates
const SOURCE = normpath(joinpath(@__DIR__,"..","..","..","..","..","src","fermionic","kagome"))
include(joinpath(SOURCE,"KagomeFermionicZ2.jl"))
using .KagomeFermionicZ2
include(joinpath(SOURCE,"ctmrg.jl"))
using .KagomeFermionicCTM
include(joinpath(SOURCE,"optimization","KagomeFermionicOptimization.jl"))
using .KagomeFermionicOptimization
const KF = KagomeFermionicZ2
const KO = KagomeFermionicOptimization
BLAS.set_num_threads(1)
const OUTPUT = normpath(joinpath(@__DIR__,"..","..","results","gapped_D4"))
mkpath(OUTPUT)
const PARAMETERS = Dict{String,Any}("t_up"=>1.0,"t_dn"=>0.2,"U"=>0.0,"μ"=>-0.6,"centered"=>true)
const EXACT_ENERGY = -0.9457489221413491

function note(message)
    line = "$(Dates.now()) $message"
    println(line)
    flush(stdout)
    open(io -> println(io,line),joinpath(OUTPUT,"progress.log"),"a")
end

function measure(label,state;chis=(12,24),maxiter=150,tol=1e-9)
    records = []
    for chi in chis
        elapsed = @elapsed result = measure_ctmrg(state,PARAMETERS;chi,maxiter,tol,verbose=false)
        ob = result.observables
        record = Dict("label"=>label,"chi"=>chi,"energy"=>ob.energy_per_site,
                      "energy_error"=>ob.energy_per_site-EXACT_ENERGY,
                      "density"=>ob.density,"double_occupancy"=>ob.double_occupancy,
                      "ctm_error"=>result.error,"ctm_iterations"=>result.iterations,
                      "ctm_converged"=>result.converged,"seconds"=>elapsed)
        push!(records,record)
        note("measurement $record")
        jldsave(joinpath(OUTPUT,"$(label)_chi$(chi)_ob.jld2");parameters=PARAMETERS,observables=ob,record)
    end
    open(io -> TOML.print(io,Dict("measurements"=>records,"exact_energy"=>EXACT_ENERGY)),joinpath(OUTPUT,"$(label)_measurements.toml"),"w")
    return records
end

function run_su(;resume=false,schedule=((5.0,0.1),(3.0,0.02),(1.0,0.005),(0.2,0.001),(2.0,0.005),(0.5,0.001)))
    checkpoint = joinpath(OUTPUT,"su.jld2")
    state = resume ? load_checkpoint(checkpoint)[1] : initial_iPESS(1,1,Rep[ℤ₂](0=>2,1=>2);rng=MersenneTwister(888))
    history = []
    for (stage,(tau,dt)) in enumerate(schedule)
        note("SU stage=$stage tau=$tau dt=$dt")
        elapsed = @elapsed itebd!(state,PARAMETERS,tau,dt,4;trun_tol=1e-12)
        estimate = su_observables(state,PARAMETERS)
        record = (;stage,tau,dt,seconds=elapsed,su_energy=estimate.energy_per_site,density=estimate.density)
        push!(history,record)
        note("SU $record")
        note("SU virtual spaces $(map(b -> (space(b,1),space(b,2)),state.B))")
        save_checkpoint(checkpoint,state;parameters=PARAMETERS,metadata=Dict("method"=>"SU","history"=>history,"D_max"=>4))
        measure("su_stage$(stage)",state;chis=(12,))
    end
    measure("su",state)
    return state
end

function run_fu(;resume=false,schedule=((0.1,0.02),),chi=12,iterations=50,n_sweep=4,ctm_tol=1e-8,projector_tol=1e-6)
    state = load_checkpoint(joinpath(OUTPUT,resume ? "fu.jld2" : "su.jld2"))[1]
    history = []
    for (stage,(tau,dt)) in enumerate(schedule)
        note("FU stage=$stage tau=$tau dt=$dt chi=$chi iterations=$iterations n_sweep=$n_sweep ctm_tol=$ctm_tol projector_tol=$projector_tol")
        callback = function(s,record)
            note("FU $record")
            save_checkpoint(joinpath(OUTPUT,"fu_latest.jld2"),s;parameters=PARAMETERS,metadata=Dict("method"=>"FU","record"=>record))
        end
        elapsed = @elapsed records = full_update!(state,PARAMETERS,tau,dt,4;chi,ctm_iterations=iterations,
                         ctm_tol,projector_tol,n_sweep,use_gauge=true,solve_rtol=1e-10,ridge=1e-12,callback)
        append!(history,records)
        note("FU stage=$stage seconds=$elapsed")
        save_checkpoint(joinpath(OUTPUT,"fu.jld2"),state;parameters=PARAMETERS,
                        metadata=Dict("method"=>"FU","history"=>history,"chi"=>chi,"iterations"=>iterations,"n_sweep"=>n_sweep,"ctm_tol"=>ctm_tol,"projector_tol"=>projector_tol))
        measure("fu_stage$(stage)",state;chis=(12,))
    end
    measure("fu",state)
    return state
end

function run_ad(;resume=false,chi=12,iterations=8,maxiter=30,step_size=0.2,gradcheck=true,projector_tol=1e-6)
    state = load_checkpoint(joinpath(OUTPUT,resume ? "ad.jld2" : "su.jld2"))[1]
    note("AD chi=$chi iterations=$iterations maxiter=$maxiter step_size=$step_size projector_tol=$projector_tol")
    if gradcheck
        v,_ = pack_state(state)
        problem = VariationalProblem(state,PARAMETERS;chi,ctm_iterations=iterations,projector_tol)
        report = check_gradient(v,problem;epsilon=1e-5)
        note("AD gradient check $report")
        report.relative_error<1e-3 || report.absolute_error<1e-7 || error("gradient check failed")
    end
    callback = function(s,record)
        note("AD $record")
        save_checkpoint(joinpath(OUTPUT,"ad_latest.jld2"),s;parameters=PARAMETERS,metadata=Dict("method"=>"AD","record"=>record))
    end
    elapsed = @elapsed result = optimize_variational(state,PARAMETERS;chi,ctm_iterations=iterations,
                             maxiter,memory=8,step_size,grad_tol=1e-7,projector_tol,callback)
    note("AD finished status=$(result.status) E=$(result.energy) gradient_norm=$(result.gradient_norm) seconds=$elapsed")
    save_checkpoint(joinpath(OUTPUT,"ad.jld2"),result.state;parameters=PARAMETERS,
                    metadata=Dict("method"=>"AD","history"=>result.history,"chi"=>chi,"iterations"=>iterations,"projector_tol"=>projector_tol,"status"=>String(result.status)))
    measure("ad",result.state)
    return result
end

if !isempty(ARGS)
    action = only(ARGS)
    if action=="all"
        include("exact_reference.jl")
        open(io -> TOML.print(io,Dict("reference"=>[exact_reference(mesh=n) for n in (300,600)])),joinpath(OUTPUT,"exact.toml"),"w")
        run_su()
        run_fu(schedule=((0.1,0.05),),chi=12,iterations=100,n_sweep=4,projector_tol=1e-10)
        run_fu(resume=true,schedule=((0.2,0.02),(0.05,0.005)),chi=12,iterations=100,n_sweep=6,projector_tol=1e-6)
        run_ad(chi=8,iterations=4,maxiter=50,step_size=1.0,projector_tol=1e-6)
        include("summarize.jl")
    else
        action=="su" ? run_su() : action=="fu" ? run_fu() : action=="ad" ? run_ad() : error("use su, fu, ad or all")
    end
end
