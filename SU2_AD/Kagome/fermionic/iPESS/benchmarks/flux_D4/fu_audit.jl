isdefined(Main,:FLUX_OUTPUT) || include("run.jl")

"""Audit ALS convergence on one gated triangle without changing the saved state."""
function audit_flux_fu(;directory=joinpath(FLUX_OUTPUT,"seed999"),method="su")
    state,p,_=load_checkpoint(joinpath(directory,"$method.jld2"))
    env=triangle_environment(state,1,1;chi=16,ctm_iterations=80,
        ctm_tol=1e-8,projector_tol=1e-8)
    gate,_=trotter_gate(Hamiltonian_triangle(p;orientation=:up),.005)
    target=KagomeFermionicZ2._apply_gate(env.W0,gate)
    # Compare the normalized full external-leg metric with the product of
    # its one-leg marginals. Julia's first tensor index is fastest varying.
    dimension=prod(dim(space(env.metric,i)) for i in 1:3)
    rho=reshape(convert(Array,env.metric),dimension,dimension)
    rho/=real(tr(rho))
    gauges=KagomeFermionicOptimization._gauges(env.metric)
    marginals=[convert(Array,g.g*g.g') for g in gauges]
    product=kron(marginals[3],kron(marginals[2],marginals[1]))
    nonproduct=norm(rho-product)/norm(rho)
    reports=[]
    for sweeps in (0,12,40)
        fit=fit_triangle(target,env;Dmax=4,n_sweep=sweeps,use_gauge=true,
                         solve_rtol=1e-10,ridge=1e-12)
        row=Dict("max_sweeps"=>sweeps,"actual_sweeps"=>length(fit.losses)-1,
            "fidelity"=>fit.fidelity,"losses"=>fit.losses,"selected_seed"=>String(fit.seed),
            "candidates"=>[Dict("seed"=>String(c.seed),"initial_loss"=>c.initial_loss,
                "final_loss"=>c.final_loss,"fidelity"=>c.fidelity) for c in fit.candidates])
        push!(reports,row); note(directory,"FU_AUDIT $method $row")
    end
    write_toml(joinpath(directory,"fu_audit_$method.toml"),Dict("dt"=>.005,
        "metric_nonproduct_relative_norm"=>nonproduct,
        "environment"=>asdict(env.diagnostics),"checks"=>reports))
    reports
end
