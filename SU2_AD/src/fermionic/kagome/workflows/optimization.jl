using TensorKit, LinearAlgebra, Random, TOML, JLD2
const SOURCE = normpath(joinpath(@__DIR__,".."))
if !isdefined(@__MODULE__,:KagomeFermionicZ2)
    include(joinpath(SOURCE,"KagomeFermionicZ2.jl"))
end
using .KagomeFermionicZ2
if !isdefined(@__MODULE__,:KagomeFermionicCTM)
    include(joinpath(SOURCE,"ctmrg.jl"))
end
using .KagomeFermionicCTM
if !isdefined(@__MODULE__,:KagomeFermionicOptimization)
    include(joinpath(SOURCE,"optimization","KagomeFermionicOptimization.jl"))
end
using .KagomeFermionicOptimization

function run_optimization(method,config_path)
    method in (:full_update,:variational) || throw(ArgumentError("unknown optimization method"))
    config_path = abspath(config_path)
    config = TOML.parsefile(config_path)
    base = dirname(config_path)
    run,model = config["run"],config["model"]
    parameters = Dict{String,Any}("t_up"=>get(model,"t_up",1.0),"t_dn"=>get(model,"t_dn",1.0),
                                  "U"=>get(model,"U",0.0),"μ"=>get(model,"mu",0.0),
                                  "centered"=>get(model,"centered",true))
    haskey(model,"pairing") && (parameters["pairing"] = model["pairing"])
    BLAS.set_num_threads(get(run,"blas_threads",1))
    resume = get(run,"resume","")
    if isempty(resume)
        V = Rep[ℤ₂](0=>run["D_even"],1=>run["D_odd"])
        state = initial_iPESS(run["Lx"],run["Ly"],V;rng=MersenneTwister(get(run,"seed",888)))
    else
        state,saved_parameters,_ = load_checkpoint(normpath(joinpath(base,resume)))
        size(state.T_u)==(run["Lx"],run["Ly"]) || throw(DimensionMismatch("resume unit cell"))
        saved_parameters==parameters || get(run,"allow_parameter_change",false) ||
            throw(ArgumentError("resume model differs; set allow_parameter_change=true for a deliberate change"))
    end
    Dmax = get(run,"D_max",dim(space(state.B[1,1,1],1)))
    warmup = get(run,"simple_update_tau",0.0)
    if warmup>0
        itebd!(state,parameters,warmup,get(run,"simple_update_dt",0.01),Dmax)
    end
    output = normpath(joinpath(base,run["output"]))
    mkpath(output)
    filename = joinpath(output,"$(method)_kagome_iPESS_Z2.jld2")
    println("method=$method cell=$(size(state.T_u)) parameters=$parameters")
    opts,ctm = config["optimization"],config["ctm"]
    history = NamedTuple[]
    status = :completed_schedule
    if method==:full_update
        for (stage,schedule) in enumerate(config["schedule"])
            callback = function (s,record)
                println("stage=$stage step=$(record.step) $(record.orientation) ($(record.x),$(record.y)) fidelity=$(record.fidelity) loss=$(record.loss) CTM_error=$(record.environment.ctm_error)")
                flush(stdout)
            end
            records = full_update!(state,parameters,schedule["tau"],schedule["dt"],Dmax;
                                    chi=ctm["chi"],ctm_iterations=ctm["iterations"],ctm_tol=get(ctm,"tol",1e-7),
                                    projector_tol=get(ctm,"projector_tol",1e-8),
                                    n_sweep=get(opts,"n_sweep",10),use_gauge=get(opts,"use_gauge",true),
                                    solve_rtol=get(opts,"solve_rtol",1e-10),ridge=get(opts,"ridge",1e-12),callback)
            append!(history,records)
            save_checkpoint(filename,state;parameters,metadata=Dict("method"=>String(method),"stage"=>stage,"history"=>history,"config"=>config))
        end
    else
        if get(opts,"check_gradient",false)
            v,_ = pack_state(state)
            problem = VariationalProblem(state,parameters;chi=ctm["chi"],ctm_iterations=ctm["iterations"],
                                         projector_tol=get(ctm,"projector_tol",1e-8))
            report = check_gradient(v,problem)
            println("gradient check: $report")
            report.relative_error<get(opts,"gradient_check_tol",1e-3) || error("gradient check failed")
        end
        callback = function (s,record)
            println("iteration=$(record.iteration) E/site=$(record.energy) |g|=$(record.gradient_norm) step=$(record.step)")
            flush(stdout)
            save_checkpoint(filename,s;parameters,metadata=Dict("method"=>String(method),"record"=>record,"config"=>config))
        end
        result = optimize_variational(state,parameters;chi=ctm["chi"],ctm_iterations=ctm["iterations"],
                                       projector_tol=get(ctm,"projector_tol",1e-8),
                                       maxiter=get(opts,"maxiter",50),memory=get(opts,"memory",8),
                                       grad_tol=get(opts,"grad_tol",1e-6),step_size=get(opts,"step_size",1.0),callback)
        state,history,status = result.state,result.history,result.status
        println("optimizer status=$status E/site=$(result.energy) |g|=$(result.gradient_norm)")
    end
    save_checkpoint(filename,state;parameters,metadata=Dict("method"=>String(method),"status"=>String(status),"history"=>history,"config"=>config))
    if get(ctm,"measure",true)
        result = measure_ctmrg(state,parameters;chi=ctm["chi"],maxiter=get(ctm,"measurement_iterations",100),
                               tol=get(ctm,"tol",1e-7),verbose=false)
        println("Final CTMRG E/site=$(result.observables.energy_per_site), converged=$(result.converged), error=$(result.error)")
        jldsave(joinpath(output,"observables.jld2");parameters,observables=result.observables,
                 chi=result.chi,iterations=result.iterations,error=result.error,converged=result.converged)
    end
    println("Saved $filename")
    return (;state,history,status,filename)
end
