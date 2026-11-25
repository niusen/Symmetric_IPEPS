using TensorKit, LinearAlgebra, Random, TOML, JLD2, Dates
const SOURCE = normpath(joinpath(@__DIR__,"..","..","..","..","..","src","fermionic","kagome"))
include(joinpath(SOURCE,"KagomeFermionicZ2.jl"))
using .KagomeFermionicZ2
include(joinpath(SOURCE,"ctmrg.jl"))
using .KagomeFermionicCTM
include(joinpath(SOURCE,"optimization","KagomeFermionicOptimization.jl"))
using .KagomeFermionicOptimization
include(joinpath(@__DIR__,"..","gapped_D4","exact_reference.jl"))
BLAS.set_num_threads(1)
const OUTPUT = normpath(joinpath(@__DIR__,"..","..","results","parameter_scan_D4"))
const SU_SCHEDULE = ((5.0,0.1),(3.0,0.02),(1.0,0.005),(0.2,0.001),(2.0,0.005),(0.5,0.001))
const FU_SCHEDULE = ((0.1,0.05,4,1e-10),(0.2,0.02,6,1e-6),(0.05,0.005,6,1e-6))
mkpath(OUTPUT)
point_name(r) = "r"*replace(string(r),"."=>"p")
function note(directory,message)
    line = "$(Dates.now()) $message"
    println(line); flush(stdout)
    open(io -> println(io,line),joinpath(directory,"progress.log"),"a")
end
asdict(record) = Dict(string(k)=>v for (k,v) in pairs(record))
function write_toml(path,data)
    open(io -> TOML.print(io,data),path,"w")
end
function sectors(state)
    [Dict("site"=>b,"up"=>[dim(space(state.B[1,1,b],2),Z2Irrep(q)) for q in (0,1)],
          "down"=>[dim(space(state.B[1,1,b],1),Z2Irrep(q)) for q in (0,1)]) for b in 1:3]
end
function measure(directory,label,state,parameters,exact;chis=(12,24))
    records = []
    for chi in chis
        elapsed = @elapsed result = measure_ctmrg(state,parameters;chi,maxiter=200,tol=1e-9,verbose=false)
        ob = result.observables
        record = Dict("label"=>label,"chi"=>chi,"energy"=>ob.energy_per_site,
                      "energy_error"=>ob.energy_per_site-exact["energy_per_site"],
                      "density"=>ob.density,"double_occupancy"=>ob.double_occupancy,
                      "ctm_error"=>result.error,"ctm_iterations"=>result.iterations,
                      "ctm_converged"=>result.converged,"seconds"=>elapsed)
        push!(records,record)
        note(directory,"MEASURE $record")
        jldsave(joinpath(directory,"$(label)_chi$(chi)_ob.jld2");parameters,observables=ob,record)
    end
    write_toml(joinpath(directory,"$(label)_measurements.toml"),Dict("measurements"=>records))
    records
end
function run_point(r;ad_maxiter=50)
    directory = joinpath(OUTPUT,point_name(r)); mkpath(directory)
    parameters = Dict{String,Any}("t_up"=>1.0,"t_dn"=>r,"U"=>0.0,"μ"=>-(1+r)/2,"centered"=>true)
    refs = [exact_reference(t_dn=r,mu=parameters["μ"],mesh=n) for n in (300,600)]
    exact = refs[end]
    @assert abs(refs[1]["energy_per_site"]-exact["energy_per_site"])<1e-10
    write_toml(joinpath(directory,"settings.toml"),Dict("parameters"=>parameters,"reference"=>refs,
        "D_max"=>4,"seed"=>888,"unit_cell"=>[1,1],"su_schedule_tau_dt"=>[collect(s) for s in SU_SCHEDULE],
        "fu_schedule_tau_dt_sweeps_projector_tol"=>[collect(s) for s in FU_SCHEDULE],"ad_maxiter"=>ad_maxiter,
        "initialization"=>"FU and AD independently start from the same SU state"))
    note(directory,"BEGIN r=$r exact=$exact")
    sufile = joinpath(directory,"su.jld2")
    if isfile(sufile)
        su,_,_ = load_checkpoint(sufile)
        note(directory,"reuse completed SU checkpoint")
    else
        su = initial_iPESS(1,1,Rep[ℤ₂](0=>2,1=>2);rng=MersenneTwister(888))
        history = []
        for (stage,(tau,dt)) in enumerate(SU_SCHEDULE)
            elapsed = @elapsed itebd!(su,parameters,tau,dt,4;trun_tol=1e-12)
            push!(history,(;stage,tau,dt,seconds=elapsed))
            note(directory,"SU stage=$stage tau=$tau dt=$dt seconds=$elapsed sectors=$(sectors(su))")
            save_checkpoint(joinpath(directory,"su_latest.jld2"),su;parameters,metadata=Dict("history"=>history))
        end
        save_checkpoint(sufile,su;parameters,metadata=Dict("method"=>"SU","history"=>history,"sectors"=>sectors(su)))
    end
    measure(directory,"su",su,parameters,exact)
    fufile = joinpath(directory,"fu.jld2")
    if !isfile(fufile)
        fu = deepcopy(su); history = []
        for (stage,(tau,dt,sweeps,projector_tol)) in enumerate(FU_SCHEDULE)
            counter = Ref(0)
            callback = function(state,record)
                counter[] += 1
                counter[]%3==0 && note(directory,"FU stage=$stage step=$(record.step) fidelity=$(record.fidelity) env_error=$(record.environment.ctm_error)")
            end
            elapsed = @elapsed records = full_update!(fu,parameters,tau,dt,4;chi=12,ctm_iterations=100,
                ctm_tol=1e-8,projector_tol,n_sweep=sweeps,use_gauge=true,solve_rtol=1e-10,ridge=1e-12,callback)
            append!(history,records)
            note(directory,"FU stage=$stage seconds=$elapsed")
            save_checkpoint(joinpath(directory,"fu_latest.jld2"),fu;parameters,metadata=Dict("stage"=>stage,"history"=>history))
        end
        save_checkpoint(fufile,fu;parameters,metadata=Dict("method"=>"FU","history"=>history,"sectors"=>sectors(fu)))
    else
        fu = load_checkpoint(fufile)[1]
    end
    measure(directory,"fu",fu,parameters,exact)
    adfile = joinpath(directory,"ad.jld2")
    if !isfile(adfile)
        v,_ = pack_state(su)
        chi,iterations,projector_tol = 8,4,1e-6
        # Increase unrolled sweeps if their energy has not reached the measured environment.
        measured = TOML.parsefile(joinpath(directory,"su_measurements.toml"))["measurements"][end]["energy"]
        diagnostics = []
        for sweeps in (4,8,12)
            problem = VariationalProblem(su,parameters;chi,ctm_iterations=sweeps,projector_tol)
            energy = variational_energy(v,problem)
            push!(diagnostics,Dict("sweeps"=>sweeps,"energy"=>energy,"difference_from_measurement"=>energy-measured))
            note(directory,"AD objective sweeps=$sweeps E=$energy measured=$measured")
            iterations = sweeps
            abs(energy-measured)<1e-7 && break
        end
        problem = VariationalProblem(su,parameters;chi,ctm_iterations=iterations,projector_tol)
        report = check_gradient(v,problem;epsilon=1e-5)
        note(directory,"AD gradient check $report")
        write_toml(joinpath(directory,"ad_gradient_check.toml"),Dict("initial"=>asdict(report),"objective_checks"=>diagnostics))
        report.relative_error<1e-3 || report.absolute_error<1e-7 || error("gradient check failed for r=$r")
        callback = function(state,record)
            note(directory,"AD $record")
            if record.iteration%10==0
                save_checkpoint(joinpath(directory,"ad_latest.jld2"),state;parameters,metadata=Dict("record"=>record))
            end
        end
        elapsed = @elapsed result = optimize_variational(su,parameters;chi,ctm_iterations=iterations,maxiter=ad_maxiter,
            memory=8,step_size=1.0,grad_tol=1e-7,projector_tol,callback)
        note(directory,"AD finished status=$(result.status) E=$(result.energy) gradient_norm=$(result.gradient_norm) seconds=$elapsed")
        save_checkpoint(adfile,result.state;parameters,metadata=Dict("method"=>"AD","history"=>result.history,
            "chi"=>chi,"ctm_iterations"=>iterations,"projector_tol"=>projector_tol,"status"=>String(result.status),
            "gradient_norm"=>result.gradient_norm,"sectors"=>sectors(result.state)))
        ad = result.state
    else
        ad = load_checkpoint(adfile)[1]
    end
    measure(directory,"ad",ad,parameters,exact)
    note(directory,"COMPLETE r=$r")
end

if abspath(PROGRAM_FILE)==@__FILE__
    ratios = isempty(ARGS) ? (0.1,0.4,0.6,0.0) : parse.(Float64,ARGS)
    for ratio in ratios
        run_point(ratio)
        GC.gc()
    end
end
