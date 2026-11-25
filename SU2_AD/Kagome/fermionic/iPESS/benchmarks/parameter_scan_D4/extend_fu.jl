isdefined(Main,:KagomeFermionicOptimization) || include("run.jl")

function extend_fu(r)
    directory = joinpath(OUTPUT,point_name(r))
    state,parameters,_ = load_checkpoint(joinpath(directory,"fu.jld2"))
    history = []
    for (stage,(tau,dt)) in enumerate(((0.5,0.05),(0.1,0.01)))
        counter = Ref(0)
        callback = function(s,record)
            counter[] += 1
            counter[]%3==0 && note(directory,"EXTENDED_FU stage=$stage step=$(record.step) loss=$(record.loss)")
        end
        records = full_update!(state,parameters,tau,dt,4;chi=12,ctm_iterations=100,
            ctm_tol=1e-8,projector_tol=1e-6,n_sweep=10,use_gauge=true,solve_rtol=1e-10,ridge=1e-12,callback)
        append!(history,records)
        save_checkpoint(joinpath(directory,"fu_extended.jld2"),state;parameters,metadata=Dict("history"=>history,"stage"=>stage))
        measure(directory,"fu_extended_stage$(stage)",state,parameters,exact_reference(t_dn=r,mu=parameters["μ"]);chis=(24,))
    end
    measure(directory,"fu_extended",state,parameters,exact_reference(t_dn=r,mu=parameters["μ"]))
    state
end

function refine_fu_step(r)
    directory = joinpath(OUTPUT,point_name(r))
    state,parameters,metadata = load_checkpoint(joinpath(directory,"fu_extended.jld2"))
    callback = (s,record) -> record.orientation==:up && note(directory,"EXTENDED_FU fine step=$(record.step) loss=$(record.loss)")
    records = full_update!(state,parameters,0.05,0.005,4;chi=12,ctm_iterations=100,
        ctm_tol=1e-8,projector_tol=1e-6,n_sweep=10,use_gauge=true,solve_rtol=1e-10,ridge=1e-12,callback)
    append!(metadata["history"],records); metadata["final_dt"] = 0.005
    save_checkpoint(joinpath(directory,"fu_extended_fine.jld2"),state;parameters,metadata)
    measure(directory,"fu_extended_fine",state,parameters,exact_reference(t_dn=r,mu=parameters["μ"]))
end

if abspath(PROGRAM_FILE)==@__FILE__
    ratio = isempty(ARGS) ? 0.4 : parse(Float64,only(ARGS))
    extend_fu(ratio)
    refine_fu_step(ratio)
end
