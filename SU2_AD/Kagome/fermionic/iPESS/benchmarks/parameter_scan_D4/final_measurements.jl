isdefined(Main,:KagomeFermionicOptimization) || include("run.jl")

function final_measurements(r;methods=("su","fu","ad"),chis=(24,48),tolerances=(1e-10,1e-12))
    directory = joinpath(OUTPUT,point_name(r)); mkpath(directory)
    source = r==0.2 ? normpath(joinpath(OUTPUT,"..","gapped_D4")) : directory
    for method in methods
        selected = method=="ad" && isfile(joinpath(source,"ad_strict.jld2")) ? "ad_strict" : method
        checkpoint = joinpath(source,"$selected.jld2")
        state,parameters,_ = load_checkpoint(checkpoint)
        exact = exact_reference(t_dn=r,mu=parameters["μ"])
        output_file = joinpath(directory,"$(method)_final_measurements.toml")
        records = isfile(output_file) && mtime(output_file)>=mtime(checkpoint) ? TOML.parsefile(output_file)["measurements"] : []
        for tolerance in tolerances,chi in chis
            any(m -> m["chi"]==chi && m["projector_tol"]==tolerance && m["checkpoint"]==selected,records) && continue
            if any(m -> m["chi"]<chi && m["projector_tol"]==tolerance && !m["ctm_converged"],records)
                note(directory,"SKIP higher chi=$chi at unstable projector_tol=$tolerance for $method; use converged less stringent measurements")
                continue
            end
            elapsed = @elapsed result = measure_ctmrg(state,parameters;chi,maxiter=60,tol=1e-9,trun_tol=tolerance,verbose=false)
            ob = result.observables
            record = Dict("label"=>method,"checkpoint"=>selected,"chi"=>chi,"projector_tol"=>tolerance,
                "energy"=>ob.energy_per_site,"energy_error"=>ob.energy_per_site-exact["energy_per_site"],
                "density"=>ob.density,"double_occupancy"=>ob.double_occupancy,
                "ctm_error"=>result.error,"ctm_iterations"=>result.iterations,"ctm_converged"=>result.converged,"seconds"=>elapsed)
            push!(records,record)
            note(directory,"FINAL_MEASURE $record")
            write_toml(output_file,Dict("measurements"=>records,"source"=>source))
        end
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    for ratio in (isempty(ARGS) ? [0.0,0.1,0.2,0.4,0.6] : parse.(Float64,ARGS))
        final_measurements(ratio)
    end
    if isempty(ARGS) && isfile(joinpath(OUTPUT,"r0p4","fu_extended_fine.jld2"))
        final_measurements(0.4;methods=("fu_extended_fine",))
    end
end
