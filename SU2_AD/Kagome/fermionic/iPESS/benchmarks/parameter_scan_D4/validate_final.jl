isdefined(Main,:KagomeFermionicOptimization) || include("run.jl")

function validate_final(r)
    directory = joinpath(OUTPUT,point_name(r))
    label = isfile(joinpath(directory,"ad_strict.jld2")) ? "ad_strict" : "ad"
    state,parameters,metadata = load_checkpoint(joinpath(directory,"$label.jld2"))
    v,_ = pack_state(state)
    problem = VariationalProblem(state,parameters;chi=metadata["chi"],
        ctm_iterations=metadata["ctm_iterations"],projector_tol=metadata["projector_tol"])
    gradient = check_gradient(v,problem;epsilon=1e-5)
    note(directory,"FINAL_GRADIENT $gradient")
    original = measure_ctmrg(state,parameters;chi=24,maxiter=200,tol=1e-9,trun_tol=1e-10,verbose=false)
    rp = merge(parameters,Dict("t_up"=>parameters["t_dn"],"t_dn"=>parameters["t_up"]))
    reflected = measure_ctmrg(reverse_triangles(state),rp;chi=24,maxiter=200,tol=1e-9,trun_tol=1e-10,verbose=false)
    comparison = Dict("original_energy"=>original.observables.energy_per_site,
        "reflected_energy"=>reflected.observables.energy_per_site,
        "difference"=>reflected.observables.energy_per_site-original.observables.energy_per_site,
        "original_converged"=>original.converged,"reflected_converged"=>reflected.converged,
        "original_ctm_error"=>original.error,"reflected_ctm_error"=>reflected.error,
        "chi"=>24,"projector_tol"=>1e-10)
    note(directory,"FINAL_REBLOCKING $comparison")
    write_toml(joinpath(directory,"final_validation.toml"),Dict("gradient"=>asdict(gradient),"reblocking"=>comparison))
end

if abspath(PROGRAM_FILE)==@__FILE__
    for ratio in (isempty(ARGS) ? [0.1,0.4,0.6] : parse.(Float64,ARGS))
        validate_final(ratio)
    end
end
