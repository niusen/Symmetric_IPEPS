isdefined(Main,:KagomeFermionicOptimization) || include("run.jl")

function run_strict_ad(r;maxiter=50)
    directory = joinpath(OUTPUT,point_name(r))
    state,parameters,_ = load_checkpoint(joinpath(directory,"su.jld2"))
    v,_ = pack_state(state)
    chi,iterations,projector_tol = 12,4,1e-8
    problem = VariationalProblem(state,parameters;chi,ctm_iterations=iterations,projector_tol)
    initial_check = check_gradient(v,problem;epsilon=1e-5)
    initial_check.relative_error<1e-3 || initial_check.absolute_error<1e-7 || error("strict AD gradient check failed")
    note(directory,"STRICT_AD initial gradient $initial_check")
    callback = function(s,record)
        note(directory,"STRICT_AD $record")
        record.iteration%10==0 && save_checkpoint(joinpath(directory,"ad_strict_latest.jld2"),s;parameters,metadata=Dict("record"=>record))
    end
    result = optimize_variational(state,parameters;chi,ctm_iterations=iterations,maxiter,
        memory=8,step_size=1.0,grad_tol=1e-7,projector_tol,callback)
    note(directory,"STRICT_AD finished status=$(result.status) E=$(result.energy) gradient_norm=$(result.gradient_norm)")
    save_checkpoint(joinpath(directory,"ad_strict.jld2"),result.state;parameters,
        metadata=Dict("method"=>"AD","history"=>result.history,"chi"=>chi,"ctm_iterations"=>iterations,
                      "projector_tol"=>projector_tol,"status"=>String(result.status),"sectors"=>sectors(result.state)))
    final_v,_ = pack_state(result.state)
    final_check = check_gradient(final_v,problem;epsilon=1e-5)
    note(directory,"STRICT_AD final gradient $final_check")
    write_toml(joinpath(directory,"ad_strict_gradient_check.toml"),Dict("initial"=>asdict(initial_check),"final"=>asdict(final_check)))
    measure(directory,"ad_strict",result.state,parameters,exact_reference(t_dn=r,mu=parameters["μ"]))
    result
end

if abspath(PROGRAM_FILE)==@__FILE__
    for ratio in (isempty(ARGS) ? [0.1] : parse.(Float64,ARGS))
        run_strict_ad(ratio)
    end
end
