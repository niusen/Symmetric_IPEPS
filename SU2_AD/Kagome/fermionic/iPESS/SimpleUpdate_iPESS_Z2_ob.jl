using TensorKit, LinearAlgebra, Random, TOML, JLD2

const SOURCE = normpath(joinpath(@__DIR__,"..","..","..","src","fermionic","kagome"))
include(joinpath(SOURCE,"KagomeFermionicZ2.jl"))
using .KagomeFermionicZ2
include(joinpath(SOURCE,"ctmrg.jl"))
using .KagomeFermionicCTM

function main(config_path=joinpath(@__DIR__,"config.toml"))
    config_path = abspath(config_path)
    config = TOML.parsefile(config_path)
    base = dirname(config_path)
    run = config["run"]
    model = config["model"]
    parameters = Dict{String,Any}("t_up"=>get(model,"t_up",1.0),"t_dn"=>get(model,"t_dn",1.0),
                                  "U"=>get(model,"U",0.0),"μ"=>get(model,"mu",0.0),
                                  "centered"=>get(model,"centered",true))
    D_max = run["D_max"]
    BLAS.set_num_threads(get(run,"blas_threads",1))
    resume = get(run,"resume","")
    if isempty(resume)
        V = Rep[ℤ₂](0=>run["D_even"],1=>run["D_odd"])
        dim(V) <= D_max || throw(ArgumentError("initial bond dimension exceeds D_max"))
        state = initial_iPESS(run["Lx"],run["Ly"],V;rng=MersenneTwister(get(run,"seed",888)))
    else
        state,saved_parameters,_ = load_checkpoint(normpath(joinpath(base,resume)))
        size(state.T_u) == (run["Lx"],run["Ly"]) || throw(DimensionMismatch("resume unit cell"))
        saved_parameters == parameters ||
            get(run,"allow_parameter_change",false) ||
            throw(ArgumentError("resume parameters differ; set allow_parameter_change=true for a deliberate quench"))
    end
    output = normpath(joinpath(base,get(run,"output","results")))
    mkpath(output)
    filename = joinpath(output,"SU_kagome_iPESS_Z2_D$(D_max).jld2")
    println("Kagome spinful Hubbard, Z2 parity, cell=$(size(state.T_u)), D_max=$D_max")
    println("parameters=$parameters")
    println("Initial SU estimate: $(su_observables(state,parameters).energy_per_site)")
    history = NamedTuple[]
    for (stage,schedule) in enumerate(config["schedule"])
        callback = function (_,record)
            if record.step == 1 || record.step % get(run,"print_every",20) == 0
                println("stage=$stage step=$(record.step) tau=$(record.tau) discarded_weight=$(record.discarded_weight)")
                flush(stdout)
            end
        end
        append!(history,itebd!(state,parameters,schedule["tau"],schedule["dt"],D_max;
                               trun_tol=get(run,"trun_tol",1e-10),callback))
        ob = su_observables(state,parameters)
        println("stage=$stage SU estimate E/site=$(ob.energy_per_site), n=$(ob.density)")
        save_checkpoint(filename,state;parameters,metadata=Dict("stage"=>stage,"history"=>history,"config"=>config))
    end
    # Also save zero-stage runs, before starting the more expensive CTMRG.
    save_checkpoint(filename,state;parameters,metadata=Dict("history"=>history,"config"=>config))
    ctm = get(config,"ctm",Dict())
    if get(ctm,"enabled",true)
        result = measure_ctmrg(state,parameters;chi=get(ctm,"chi",20),
                               maxiter=get(ctm,"maxiter",100),tol=get(ctm,"tol",1e-6),
                               verbose=get(ctm,"verbose",true))
        println("CTMRG E/site=$(result.observables.energy_per_site), n=$(result.observables.density)")
        println("CTMRG converged=$(result.converged), iterations=$(result.iterations), error=$(result.error)")
        result.converged || @warn "CTMRG has not converged; increase maxiter/chi and check convergence before interpreting this energy."
        jldsave(joinpath(output,"observables.jld2");parameters,observables=result.observables,
                 chi=result.chi,iterations=result.iterations,error=result.error,converged=result.converged)
    end
    println("Saved $filename")
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? joinpath(@__DIR__,"config.toml") : only(ARGS))
end
