isdefined(Main,:FLUX_OUTPUT) || include("run.jl")

"""Check fixed-sweep convergence and two independent real directional derivatives."""
function audit_flux_ad(;directory=joinpath(FLUX_OUTPUT,"seed999"))
    state,p,metadata=load_checkpoint(joinpath(directory,"ad.jld2"))
    v,_=pack_state(state)
    reports=[]
    previous_gradient=nothing
    for sweeps in unique((metadata["ctm_iterations"],8,12))
        problem=VariationalProblem(state,p;chi=metadata["chi"],ctm_iterations=sweeps,
                                   projector_tol=metadata["projector_tol"])
        energy,g=value_and_gradient(v,problem)
        gradient_change=isnothing(previous_gradient) ? 0.0 :
            norm(g-previous_gradient)/max(norm(g),norm(previous_gradient),eps())
        for seed in (123,456)
            direction=randn(MersenneTwister(seed),ComplexF64,length(v))
            direction/=norm(direction)
            ad=real(dot(g,direction))
            for epsilon in (1e-5,3e-6)
                fd=(variational_energy(v+epsilon*direction,problem)-
                    variational_energy(v-epsilon*direction,problem))/(2epsilon)
                absolute=abs(ad-fd); relative=absolute/max(abs(ad),abs(fd),1e-10)
                row=Dict("sweeps"=>sweeps,"seed"=>seed,"epsilon"=>epsilon,
                    "energy"=>energy,"gradient_norm"=>norm(g),"ad"=>ad,
                    "relative_gradient_change_from_previous_sweeps"=>gradient_change,
                    "finite_difference"=>fd,"absolute_error"=>absolute,
                    "relative_error"=>relative,"passed"=>(relative<1e-3 || absolute<1e-7))
                push!(reports,row); note(directory,"AD_AUDIT $row")
                write_toml(joinpath(directory,"ad_audit.toml"),Dict("checks"=>reports))
            end
        end
        previous_gradient=copy(g)
    end
    reports
end

"""Reproduce failed attempts to tighten the final state's AD contraction."""
function probe_flux_ad_truncation(;directory=joinpath(FLUX_OUTPUT,"seed999"))
    state,p,_=load_checkpoint(joinpath(directory,"ad.jld2"))
    v,_=pack_state(state)
    for (filename,configs) in (
        ("ad_tighten_checks.toml",[(16,c) for c in (1e-6,1e-8,1e-10)]),
        ("ad_chi_checks.toml",[(c,1e-8) for c in (8,10,12)]))
        reports=[]
        for (chi,cutoff) in configs
            problem=VariationalProblem(state,p;chi,ctm_iterations=8,projector_tol=cutoff)
            row=merge(asdict(check_gradient(v,problem;epsilon=1e-5)),
                      Dict("chi"=>chi,"cutoff"=>cutoff,"ctm_iterations"=>8))
            push!(reports,row); note(directory,"AD_TRUNCATION_PROBE $row")
            write_toml(joinpath(directory,filename),Dict("checks"=>reports))
        end
    end
end
